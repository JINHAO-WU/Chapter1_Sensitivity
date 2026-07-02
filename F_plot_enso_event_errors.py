"""
ENSO event amplitude and peak-month forecast errors.

Reads pickle files with predict_value / real_value arrays.
Edit the configuration block for data sources, leads, or composite window.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from A_basic_sources import (
    FIGURE_ROOT,
    get_dl_sources,
    list_pickle_files_by_year,
    load_prediction_arrays,
)
from plot_style import (
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    TITLE_SIZE,
    add_compact_figure_legend,
    configure_publication_style,
    dataset_color,
    panel_title,
    save_publication_figure,
    style_light_grid,
    style_open_axes,
    validate_data_sources,
)


# =============================================================================
# User configuration
# =============================================================================

FIGURE_ID = "F"
FIGURE_NAME = "enso_event_errors"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
FIGURE_DPI = 300
LEADS = [6]
INPUT_MONTHS = 6
COMPOSITE_MONTHS_BEFORE = 6
COMPOSITE_MONTHS_AFTER = 6
RUN_SELF_TEST = True
DATA_SOURCES = get_dl_sources()

EVENT_CLASSES = ["Strong El Niño", "Weak El Niño", "Strong La Niña", "Weak La Niña"]
EVENT_GROUPS = {"El Niño": EVENT_CLASSES[:2], "La Niña": EVENT_CLASSES[2:]}


# =============================================================================
# Event detection
# =============================================================================

def _classify_peak(value: float) -> str:
    if value >= 1.5:
        return "Strong El Niño"
    if value > 0.5:
        return "Weak El Niño"
    if value <= -1.5:
        return "Strong La Niña"
    if value < -0.5:
        return "Weak La Niña"
    raise ValueError(f"Peak value {value} is not an ENSO event.")


def _polarity(value: float) -> int:
    return 1 if value > 0.5 else (-1 if value < -0.5 else 0)


def _iter_events(real: np.ndarray):
    """Yield (start, end, polarity) for each contiguous ENSO event."""
    start = None
    pol = 0
    for i, v in enumerate(real):
        cur = _polarity(float(v))
        if cur == 0:
            if start is not None:
                yield start, i, pol
            start, pol = None, 0
        elif start is None or cur != pol:
            if start is not None:
                yield start, i, pol
            start, pol = i, cur
    if start is not None:
        yield start, len(real), pol


def _composite_window(pred: np.ndarray, real: np.ndarray, peak: int):
    """Return (pred_window, real_window) centered on peak, or None if out of bounds."""
    lo, hi = peak - COMPOSITE_MONTHS_BEFORE, peak + COMPOSITE_MONTHS_AFTER + 1
    if lo < 0 or hi > len(real):
        return None
    return pred[lo:hi], real[lo:hi]


def _event_metrics(pred: np.ndarray, real: np.ndarray) -> tuple[list[dict], dict]:
    """Compute metrics and collect composite windows for one lead."""
    metrics: list[dict] = []
    composites = {ec: {"pred": [], "real": []} for ec in EVENT_CLASSES}

    for start, end, pol in _iter_events(real):
        re, pe = real[start:end], pred[start:end]
        peak_fn = np.nanargmax if pol > 0 else np.nanargmin
        real_peak_idx = int(peak_fn(re))
        pred_peak_idx = int(peak_fn(pe))
        real_peak_val = float(re[real_peak_idx])
        pred_at_peak = float(pe[real_peak_idx])

        ec = _classify_peak(real_peak_val)
        win = _composite_window(pred, real, start + real_peak_idx)
        if win is not None:
            composites[ec]["pred"].append(win[0])
            composites[ec]["real"].append(win[1])

        amp_diff = (real_peak_val - pred_at_peak if pol > 0
                    else abs(real_peak_val) - abs(pred_at_peak))
        metrics.append({
            "class": ec,
            "amplitude_underestimate": amp_diff,
            "peak_error": pred_peak_idx - real_peak_idx,
        })

    return metrics, composites


def _mean_or_nan(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else math.nan


def _summarize(evts: list[dict]) -> dict:
    """Summarize metrics into per-class and per-group aggregates."""
    pick = lambda cls, key: _mean_or_nan(
        [float(m[key]) for m in evts if m["class"] == cls])

    return {
        "amplitude_by_class": {ec: pick(ec, "amplitude_underestimate")
                               for ec in EVENT_CLASSES},
        "peak_error_by_class": {ec: pick(ec, "peak_error")
                                for ec in EVENT_CLASSES},
        "peak_error_by_group": {grp: _mean_or_nan(
            [float(m["peak_error"]) for m in evts if m["class"] in classes])
            for grp, classes in EVENT_GROUPS.items()},
        "mean_amplitude_underestimate": _mean_or_nan(
            [float(m["amplitude_underestimate"]) for m in evts]),
        "mean_peak_error": _mean_or_nan(
            [float(m["peak_error"]) for m in evts]),
    }


def _analyze_pickle(path: Path) -> tuple[dict, dict]:
    pred, real = load_prediction_arrays(path)
    expected = 360 - INPUT_MONTHS - 18
    if pred.shape[0] != expected:
        raise ValueError(f"{path.name}: expected {expected} samples, got {pred.shape[0]}.")
    if max(LEADS) > pred.shape[1]:
        raise ValueError(f"{path.name}: lead {max(LEADS)} > {pred.shape[1]} available.")

    evts: list[dict] = []
    composites: dict = {ec: {"pred": [], "real": []} for ec in EVENT_CLASSES}
    for lead in LEADS:
        lm, lc = _event_metrics(pred[:, lead - 1], real[:, lead - 1])
        evts.extend(lm)
        for ec in EVENT_CLASSES:
            composites[ec]["pred"].extend(lc[ec]["pred"])
            composites[ec]["real"].extend(lc[ec]["real"])
    return _summarize(evts), composites


def _collect_all(data_sources: list[dict]) -> tuple[list[dict], dict, dict]:
    all_res: list[dict] = []
    comp_by_ds = {s["id"]: {ec: {"pred": [], "real": []} for ec in EVENT_CLASSES}
                  for s in data_sources}
    obs_comp = {ec: {"pred": [], "real": []} for ec in EVENT_CLASSES}

    for src in data_sources:
        files = list_pickle_files_by_year(src["pickle_dir"])
        years = sorted(files)
        print(f"{'='*72}\n{src['label']}: {len(years)} pickle files\n"
              f"Years: {years[0]}-{years[-1]}\n{src['pickle_dir']}\n{'='*72}")

        for yr in years:
            r, c = _analyze_pickle(files[yr])
            r["dataset"] = src["id"]
            r["year"] = yr
            all_res.append(r)
            for ec in EVENT_CLASSES:
                comp_by_ds[src["id"]][ec]["real"].extend(c[ec]["real"])
                obs_comp[ec]["real"].extend(c[ec]["real"])
            print(f"{src['label']:18s} year={yr} "
                  f"amp_under={r['mean_amplitude_underestimate']:.4f} "
                  f"peak_err={r['mean_peak_error']:.4f}")
    return all_res, comp_by_ds, obs_comp


# =============================================================================
# Plotting
# =============================================================================

def _lead_token() -> str:
    return "-".join(str(l) for l in LEADS)


def _sorted_for(results: list[dict], dataset_id: str, key_fn):
    """Return (years, values) for one dataset, sorted by year."""
    rows = sorted((r for r in results if r["dataset"] == dataset_id),
                  key=lambda r: r["year"])
    return np.array([r["year"] for r in rows]), np.array([key_fn(r) for r in rows])


def _source_legend(fig, data_sources, bbox_y):
    handles = [Line2D([0], [0], color=dataset_color(s["id"]), linewidth=2.0,
                      label=s["label"]) for s in data_sources]
    add_compact_figure_legend(fig, handles=handles, ncol=5,
                              bbox_to_anchor=(0.5, bbox_y), fontsize=LEGEND_SIZE,
                              columnspacing=0.40, handlelength=1.20, labelspacing=0.25)


def _style_ax(ax, title: str, panel_idx: int):
    ax.axhline(0, color="#444444", linewidth=1.0, linestyle=(0, (5, 4)))
    title_text = f"$\\mathbf{{({chr(ord('a') + panel_idx)})}}$ {title}"
    ax.set_title(title_text, loc="left", fontsize=TITLE_SIZE)
    style_light_grid(ax, axis="y", linewidth=0.7)
    style_light_grid(ax, axis="x", color="#EEEEEE", linewidth=0.5)
    ax.tick_params(axis="both", direction="in", labelsize=LEGEND_SIZE)
    style_open_axes(ax)


def _plot_amplitude(results: list[dict], data_sources: list[dict], out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), dpi=FIGURE_DPI,
                             sharex=True, sharey=False)
    axes = axes.ravel()
    for i, (ax, ec) in enumerate(zip(axes, EVENT_CLASSES)):
        for src in data_sources:
            yrs, vals = _sorted_for(results, src["id"],
                                    lambda r: r["amplitude_by_class"][ec])
            ax.plot(yrs, vals, color=dataset_color(src["id"]), linewidth=1.9,
                    label=src["label"])
        _style_ax(ax, ec, i)

    for ax in axes[2:]:
        ax.set_xlabel("Test-set start year", fontsize=AXIS_LABEL_SIZE)
    for ax in axes[::2]:
        ax.set_ylabel("Amplitude underestimation", fontsize=AXIS_LABEL_SIZE)

    _source_legend(fig, data_sources, 0.97)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    save_publication_figure(fig, [out], dpi=FIGURE_DPI, pad_inches=0.02)
    plt.close(fig)


def _plot_peak_error(results: list[dict], data_sources: list[dict], out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=FIGURE_DPI,
                             sharex=True, sharey=False)
    axes = axes.ravel()
    for i, (ax, (grp, _)) in enumerate(zip(axes, EVENT_GROUPS.items())):
        for src in data_sources:
            yrs, vals = _sorted_for(results, src["id"],
                                    lambda r: r["peak_error_by_group"][grp])
            ax.plot(yrs, vals, color=dataset_color(src["id"]), linewidth=1.9,
                    label=src["label"])
        _style_ax(ax, grp, i)

    for ax in axes:
        ax.set_xlabel("Test-set start year", fontsize=AXIS_LABEL_SIZE)
    axes[0].set_ylabel("Peak-month error (months)", fontsize=AXIS_LABEL_SIZE)

    _source_legend(fig, data_sources, 0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_publication_figure(fig, [out], dpi=FIGURE_DPI, pad_inches=0.02)
    plt.close(fig)


# =============================================================================
# Self-tests
# =============================================================================

def _run_tests() -> None:
    cases = {-1.5: "Strong La Niña", -0.6: "Weak La Niña",
             0.6: "Weak El Niño", 1.5: "Strong El Niño"}
    for v, exp in cases.items():
        assert _classify_peak(v) == exp, f"_classify_peak({v}) = {_classify_peak(v)} ≠ {exp}"

    global COMPOSITE_MONTHS_BEFORE, COMPOSITE_MONTHS_AFTER
    ob, oa = COMPOSITE_MONTHS_BEFORE, COMPOSITE_MONTHS_AFTER
    COMPOSITE_MONTHS_BEFORE = COMPOSITE_MONTHS_AFTER = 1
    try:
        real = np.array([0.0, 0.7, 1.4, 1.6, 1.2, 0.2, -0.6, -1.7, -1.1, 0.0])
        pred = np.array([0.0, 0.5, 1.0, 1.1, 1.7, 0.1, -0.4, -1.0, -1.8, 0.0])
        metrics, comp = _event_metrics(pred, real)
        assert [m["class"] for m in metrics] == ["Strong El Niño", "Strong La Niña"]
        assert [m["peak_error"] for m in metrics] == [1, 1]
        assert np.allclose(comp["Strong El Niño"]["real"][0], [1.4, 1.6, 1.2])
    finally:
        COMPOSITE_MONTHS_BEFORE, COMPOSITE_MONTHS_AFTER = ob, oa


def main() -> None:
    configure_publication_style()
    validate_data_sources(DATA_SOURCES)
    if RUN_SELF_TEST:
        _run_tests()

    results, comp_by_ds, obs_comp = _collect_all(DATA_SOURCES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    token = _lead_token()

    amp_path = OUTPUT_DIR / f"{FIGURE_ID}_{FIGURE_NAME}_amplitude_underestimate_by_class_lead{token}.png"
    peak_path = OUTPUT_DIR / f"{FIGURE_ID}_{FIGURE_NAME}_peak_month_error_by_phase_lead{token}.png"

    _plot_amplitude(results, DATA_SOURCES, amp_path)
    _plot_peak_error(results, DATA_SOURCES, peak_path)
    print(f"{amp_path}\n{peak_path}")


if __name__ == "__main__":
    main()
