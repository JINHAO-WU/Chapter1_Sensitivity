"""
ENSO transition frequency versus forecast skill.

Each pickle file is one test window. Edit the configuration block below.
"""

from __future__ import annotations

import math
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import stats

from A_basic_sources import FIGURE_ROOT, get_dl_sources, list_pickle_files
from plot_style import (
    DEFAULT_FIGURE_DPI,
    E_TRANSITION_STYLE,
    add_shared_axis_labels,
    configure_publication_style,
    figure_output_paths,
    mm_to_inches,
    save_publication_figure,
    source_panel_grid_5x2,
    style_colorbar,
    style_light_grid,
    style_open_axes,
    style_source_panel_axes_5x2,
    validate_data_sources,
)


# =============================================================================
# User configuration
# =============================================================================

FIGURE_ID = "E"
FIGURE_NAME = "enso_transition_frequency_vs_skill"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
FIGURE_DPI = DEFAULT_FIGURE_DPI

MAKE_SKILL_RELATIONSHIP_PLOT = True
MAKE_TRANSITION_TIME_PLOT = True

LEAD = 6
INPUT_MONTHS = 6
ENSO_THRESHOLD = 0.5
EXTREME_ENSO_THRESHOLD = 1.5

# el_nino_neutral_el_nino | la_nina_neutral_la_nina |
# neutral_el_nino_neutral | neutral_la_nina_neutral
TRANSITION_MODE = "neutral_la_nina_neutral"
ACC_WARNING_TOLERANCE = 0.02

SHOW_FIGURE = False
ANNOTATE_YEAR_RANGE = (1922, 1944)
X_AXIS_PADDING_FRACTION = 0.12
X_AXIS_MIN_PADDING = 0.0008

DATA_SOURCES = get_dl_sources()

PLOT_STYLE = {
    "cmap": "cividis",
    "point_size": E_TRANSITION_STYLE["point_size"],
    "point_alpha": 0.92,
    "point_edge_color": "#303030",
    "fit_line_color": "#1f1f1f",
    "grid_color": "#e6e6e6",
    "highlight_color": "#b2182b",
}

# (label, extreme, event_state, state_run_pattern, arrow_label)
MODE_PROPS = {
    "el_nino_neutral_el_nino":
        ("EN-Neutral-EN", False, 1, (1, 0, 1), "EN -> Neutral -> EN"),
    "la_nina_neutral_la_nina":
        ("LN-Neutral-LN", False, -1, (-1, 0, -1), "LN -> Neutral -> LN"),
    "neutral_el_nino_neutral":
        ("Neutral-EN-Neutral", False, 1, (0, 1, 0), "Neutral -> EN -> Neutral"),
    "neutral_la_nina_neutral":
        ("Neutral-LN-Neutral", False, -1, (0, -1, 0), "Neutral -> LN -> Neutral"),
    "extreme_el_nino_neutral_extreme_el_nino":
        ("ExEN-Neutral-ExEN", True, 1, (1, 0, 1), "ExEN -> Neutral -> ExEN"),
    "extreme_la_nina_neutral_extreme_la_nina":
        ("ExLN-Neutral-ExLN", True, -1, (-1, 0, -1), "ExLN -> Neutral -> ExLN"),
}


@dataclass
class StateRun:
    state: int
    start: int
    end: int


@dataclass
class WindowPoint:
    dataset: str
    dataset_label: str
    pickle_dir: Path
    file: Path
    label: str
    year: float
    transition_frequency: float
    acc: float
    recalculated_acc: float


# =============================================================================
# Helpers
# =============================================================================

def _load_pickle(path):
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected dict, got {type(data).__name__}")
    return data


def _get_array(data, name, path):
    if name not in data:
        raise KeyError(f"{path.name}: missing {name!r}")
    return np.asarray(data[name])


def _check_lead(idx, n_leads, name, path):
    if idx < 0 or idx >= n_leads:
        raise ValueError(
            f"{path.name}: LEAD={LEAD} outside range for {name} (1-{n_leads}).")


def pearson_r(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return math.nan
    xv, yv = x[valid].astype(float), y[valid].astype(float)
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return math.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def pearson_p(r, n):
    if not np.isfinite(r) or n < 3 or abs(r) >= 1:
        return math.nan
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return float(2 * stats.t.sf(abs(t), df=n - 2))


# =============================================================================
# ENSO state classification
# =============================================================================

def _classify_independent(values):
    states = np.zeros(values.shape, dtype=np.int8)
    states[values > ENSO_THRESHOLD] = 1
    states[values < -ENSO_THRESHOLD] = -1
    return states


def _classify_extreme_independent(values):
    states = np.zeros(values.shape, dtype=np.int8)
    states[values >= EXTREME_ENSO_THRESHOLD] = 1
    states[values <= -EXTREME_ENSO_THRESHOLD] = -1
    return states


def _run_states(states, valid):
    runs = []
    start = None
    cur = None
    for i, (s, ok) in enumerate(zip(states, valid)):
        if not ok:
            if start is not None and cur is not None:
                runs.append(StateRun(int(cur), start, i))
            start = cur = None
            continue
        s = int(s)
        if start is None:
            start, cur = i, s
        elif s != cur:
            runs.append(StateRun(int(cur), start, i))
            start, cur = i, s
    if start is not None and cur is not None:
        runs.append(StateRun(int(cur), start, len(states)))
    return runs


def _is_extreme(run, values, event_state):
    seg = values[run.start:run.end]
    return bool(np.nanmax(seg) >= EXTREME_ENSO_THRESHOLD if event_state == 1
                else np.nanmin(seg) <= -EXTREME_ENSO_THRESHOLD)


# =============================================================================
# Transition counting
# =============================================================================

def _count_pattern(runs, pattern):
    n = len(pattern)
    return sum(
        1 for w in zip(*(runs[i:] for i in range(n)))
        if all(r.state == s for r, s in zip(w, pattern))
    )


def _count_extreme_pattern(runs, values, pattern, event_state):
    n = len(pattern)
    count = 0
    for w in zip(*(runs[i:] for i in range(n))):
        rs = list(w)
        if all(r.state == s for r, s in zip(rs, pattern)):
            if _is_extreme(rs[0], values, event_state) and _is_extreme(rs[-1], values, event_state):
                count += 1
    return count


def _get_runs(values, is_extreme):
    valid = np.isfinite(values)
    if is_extreme:
        states = _classify_extreme_independent(values)
    else:
        states = _classify_independent(values)
    return _run_states(states, valid)


def _transition_frequency(values):
    values = np.asarray(values, dtype=float).ravel()
    if values.size < 2:
        return math.nan
    valid = np.isfinite(values)
    if valid[:-1].sum() == 0:
        return math.nan

    label, is_ext, ev_state, pattern, _arrow = MODE_PROPS[TRANSITION_MODE]
    runs = _get_runs(values, is_ext)
    count = (_count_extreme_pattern(runs, values, pattern, ev_state) if is_ext
             else _count_pattern(runs, pattern))
    return float(count / valid[:-1].sum())


def _mode_label():
    return MODE_PROPS[TRANSITION_MODE][0]


def _transition_title():
    arrow = MODE_PROPS[TRANSITION_MODE][4]
    return f"Transition: {arrow}"


def _transition_axis_label():
    if MODE_PROPS[TRANSITION_MODE][1]:
        return "Extreme ENSO-neutral-Extreme ENSO transition frequency"
    if TRANSITION_MODE.startswith("neutral_"):
        return "Neutral-ENSO-Neutral transition frequency"
    return "ENSO event-neutral transition frequency"


def _filename_token():
    return {
        "el_nino_neutral_el_nino": "el_nino",
        "la_nina_neutral_la_nina": "la_nina",
        "neutral_el_nino_neutral": "neutral_el_nino",
        "neutral_la_nina_neutral": "neutral_la_nina",
        "extreme_el_nino_neutral_extreme_el_nino": "extreme_el_nino",
        "extreme_la_nina_neutral_extreme_la_nina": "extreme_la_nina",
    }[TRANSITION_MODE]


def _parse_year(path, fallback):
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", path.stem)
    return (m.group(1), float(m.group(1))) if m else (path.stem, float(fallback))


# =============================================================================
# Data collection
# =============================================================================

def _collect_source_points(source):
    dataset_id = source["id"]
    dataset_label = source["label"]
    pickle_dir = source["pickle_dir"]
    lead_idx = LEAD - 1
    paths = list_pickle_files(pickle_dir)
    points = []

    for fi, p in enumerate(paths):
        data = _load_pickle(p)
        real = _get_array(data, "real_value", p)
        pred = _get_array(data, "predict_value", p)
        acc_vals = _get_array(data, "Pearson", p).reshape(-1)

        if real.shape != pred.shape or real.ndim != 2:
            raise ValueError(f"{p.name}: shape mismatch {real.shape} vs {pred.shape}")

        _check_lead(lead_idx, real.shape[1], "real_value", p)
        _check_lead(lead_idx, pred.shape[1], "predict_value", p)
        _check_lead(lead_idx, acc_vals.size, "Pearson", p)

        obs = real[:, lead_idx].astype(float)
        prd = pred[:, lead_idx].astype(float)
        stored_acc = float(acc_vals[lead_idx])
        recalc = pearson_r(prd, obs)

        if (np.isfinite(stored_acc) and np.isfinite(recalc)
                and abs(stored_acc - recalc) > ACC_WARNING_TOLERANCE):
            warnings.warn(
                f"{p.name}: stored Pearson lead {LEAD} = {stored_acc:.3f} "
                f"!= recalculated ACC = {recalc:.3f}", RuntimeWarning)

        label, year = _parse_year(p, fi)
        points.append(WindowPoint(
            dataset=dataset_id, dataset_label=dataset_label,
            pickle_dir=pickle_dir, file=p, label=label, year=year,
            transition_frequency=_transition_frequency(obs),
            acc=stored_acc, recalculated_acc=recalc,
        ))

    years = [pt.year for pt in points if np.isfinite(pt.year)]
    valid_freq = [pt.transition_frequency for pt in points
                  if np.isfinite(pt.transition_frequency)]
    valid_acc = [pt.acc for pt in points if np.isfinite(pt.acc)]
    print(f"{'='*72}\n{source['label']}: {len(points)} pickle files\n{pickle_dir}")
    if years:
        print(f"Years: {int(min(years))}-{int(max(years))}")
    print(f"Valid frequencies: {len(valid_freq)}; valid ACC: {len(valid_acc)}")
    return points


def _collect_all(data_sources):
    all_pts = []
    for src in data_sources:
        all_pts.extend(_collect_source_points(src))
    return all_pts


def _make_output_paths(fig_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{FIGURE_ID}_{FIGURE_NAME}_{fig_name}_{_filename_token()}_lead{LEAD}"
    return figure_output_paths(OUTPUT_DIR / stem)


def _dataset_order(points):
    seen = []
    for pt in points:
        if pt.dataset not in seen:
            seen.append(pt.dataset)
    return seen


def _points_for(points, ds):
    return [pt for pt in points if pt.dataset == ds]


def _display_label(points, ds_id):
    for pt in points:
        if pt.dataset == ds_id:
            return pt.dataset_label
    raise KeyError(f"No label for {ds_id!r}")


def _padded_limits(vals, lo, hi):
    finite = vals[np.isfinite(vals)]
    rng = float(np.nanmax(finite)) - float(np.nanmin(finite))
    pad = max(rng * X_AXIS_PADDING_FRACTION, X_AXIS_MIN_PADDING)
    return (max(lo, float(np.nanmin(finite)) - pad),
            min(hi, float(np.nanmax(finite)) + pad))


def _clean_ticks(limits, nbins=6):
    loc = MaxNLocator(nbins=nbins)
    ticks = loc.tick_values(*limits)
    ticks = ticks[(ticks >= limits[0]) & (ticks <= limits[1])]
    return ticks if ticks.size >= 2 else np.linspace(*limits, nbins)


def _shared_axes(points):
    freq = np.asarray([pt.transition_frequency for pt in points], dtype=float)
    acc = np.asarray([pt.acc for pt in points], dtype=float)
    valid = np.isfinite(freq) & np.isfinite(acc)
    if valid.sum() < 2:
        raise ValueError("Need >=2 valid points for shared axes.")
    xl = _padded_limits(freq[valid], 0.0, 1.0)
    yl = _padded_limits(acc[valid], -1.0, 1.0)
    return xl, yl, _clean_ticks(xl), _clean_ticks(yl)


def _panel_title(ds, idx):
    return f"$\\mathbf{{({chr(ord('a') + idx)})}}$ {ds}"


def _add_transition_title(fig, show_year_note):
    """Add transition info and optional year-range note below the y-axis label."""
    note_text = _transition_title()
    if show_year_note and ANNOTATE_YEAR_RANGE is not None:
        s, e = ANNOTATE_YEAR_RANGE
        note_text += f"   |   Open red circles: {s}-{e}"
    fig.text(0.5, 0.010, note_text, ha="center", va="bottom",
             fontsize=E_TRANSITION_STYLE["annotation_size"], color="#444444")


def _add_year_colorbar(fig, mappable, axes):
    """Place a longer year colorbar beside the middle rows of the 5x2 layout."""
    upper_pos = axes[3].get_position(fig)
    middle_pos = axes[5].get_position(fig)
    lower_pos = axes[7].get_position(fig)
    cax = fig.add_axes([
        middle_pos.x1 + E_TRANSITION_STYLE["colorbar_pad"],
        lower_pos.y0,
        E_TRANSITION_STYLE["colorbar_width"],
        upper_pos.y1 - lower_pos.y0,
    ])
    colorbar = fig.colorbar(mappable, cax=cax)
    style_colorbar(
        colorbar,
        label="Test-start year",
        fontsize=E_TRANSITION_STYLE["colorbar_label_size"],
        tick_labelsize=E_TRANSITION_STYLE["colorbar_tick_size"],
    )


# ---------------------------------------------------------------------------
# Figure 1: transition frequency vs skill
# ---------------------------------------------------------------------------

def _draw_skill_panel(ax, pts, ds, yr_min, yr_max, xl, yl, xticks, yticks, idx):
    freq = np.asarray([p.transition_frequency for p in pts], dtype=float)
    acc = np.asarray([p.acc for p in pts], dtype=float)
    yrs = np.asarray([p.year for p in pts], dtype=float)
    valid = np.isfinite(freq) & np.isfinite(acc)

    if valid.sum() < 2:
        ax.text(0.5, 0.5, "< 2 valid points", transform=ax.transAxes,
                ha="center", va="center", fontsize=E_TRANSITION_STYLE["tick_label_size"])
        ax.set_title(
            _panel_title(_display_label(pts, ds), idx),
            loc="left",
            fontsize=E_TRANSITION_STYLE["panel_label_size"],
            pad=3,
        )
        return None

    r_val = pearson_r(freq[valid], acc[valid])
    p_val = pearson_p(r_val, int(valid.sum()))

    sc = ax.scatter(freq[valid], acc[valid], c=yrs[valid], cmap=PLOT_STYLE["cmap"],
                    vmin=yr_min, vmax=yr_max, s=PLOT_STYLE["point_size"],
                    edgecolor=PLOT_STYLE["point_edge_color"],
                    linewidth=E_TRANSITION_STYLE["point_edge_width"],
                    alpha=PLOT_STYLE["point_alpha"])

    if np.nanstd(freq[valid]) > 0:
        fit = np.polyfit(freq[valid], acc[valid], 1)
        xl_fit = np.linspace(float(np.nanmin(freq[valid])),
                             float(np.nanmax(freq[valid])), 100)
        ax.plot(xl_fit, fit[0] * xl_fit + fit[1],
                color=PLOT_STYLE["fit_line_color"],
                linewidth=E_TRANSITION_STYLE["fit_line_width"],
                alpha=0.9)

    ax.set_title(
        _panel_title(_display_label(pts, ds), idx),
        loc="left",
        fontsize=E_TRANSITION_STYLE["panel_label_size"],
        pad=3,
    )
    style_open_axes(ax)
    style_light_grid(
        ax,
        axis="both",
        color=PLOT_STYLE["grid_color"],
        linewidth=E_TRANSITION_STYLE["grid_line_width"],
    )
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    ax.tick_params(
        axis="both",
        labelsize=E_TRANSITION_STYLE["tick_label_size"],
        length=E_TRANSITION_STYLE["tick_length"],
        width=E_TRANSITION_STYLE["tick_width"],
        pad=1.8,
    )

    p_text = "NA" if not np.isfinite(p_val) else f"{p_val:.2g}"
    ax.text(0.98, 0.98, f"r = {r_val:.2f}\np = {p_text}",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=E_TRANSITION_STYLE["annotation_size"],
            color="#555555", clip_on=False)

    if ANNOTATE_YEAR_RANGE is not None:
        a0, a1 = ANNOTATE_YEAR_RANGE
        idxs = np.where(valid & (yrs >= a0) & (yrs <= a1))[0]
        ax.scatter(
            freq[idxs],
            acc[idxs],
            s=E_TRANSITION_STYLE["highlight_point_size"],
            facecolors="none",
            edgecolors=PLOT_STYLE["highlight_color"],
            linewidth=E_TRANSITION_STYLE["highlight_edge_width"],
            zorder=5,
        )
    return sc


def _plot_skill_relationship(points):
    ds_list = _dataset_order(points)
    yrs = np.asarray([p.year for p in points], dtype=float)
    valid_yrs = yrs[np.isfinite(yrs)]
    xl, yl, xticks, yticks = _shared_axes(points)

    fig = plt.figure(figsize=(
        mm_to_inches(E_TRANSITION_STYLE["figure_width_mm"]),
        mm_to_inches(E_TRANSITION_STYLE["figure_height_mm"]),
    ))
    axes = source_panel_grid_5x2(fig, left=0.095, right=0.905, bottom=0.083,
                                 top=0.955, wspace=0.08, hspace=0.21)

    sc = None
    for i, ds in enumerate(ds_list):
        sc = _draw_skill_panel(axes[i], _points_for(points, ds), ds,
                               float(np.nanmin(valid_yrs)),
                               float(np.nanmax(valid_yrs)),
                               xl, yl, xticks, yticks, i) or sc

    if sc is not None:
        _add_year_colorbar(fig, sc, axes)

    _add_transition_title(fig, show_year_note=True)
    style_source_panel_axes_5x2(axes, n_visible=len(ds_list))
    add_shared_axis_labels(fig, xlabel=_transition_axis_label(),
                           ylabel=f"ACC at lead {LEAD}-month",
                           xlabel_y=0.035, ylabel_x=0.028,
                           fontsize=E_TRANSITION_STYLE["axis_label_size"])
    save_publication_figure(fig, _make_output_paths("frequency_vs_skill"),
                            dpi=FIGURE_DPI, pad_inches=0.03, print_paths=True)
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: transition frequency over time
# ---------------------------------------------------------------------------

def _draw_time_panel(ax, pts, ds, yr_min, yr_max, f_min, f_max, idx):
    freq = np.asarray([p.transition_frequency for p in pts], dtype=float)
    yrs = np.asarray([p.year for p in pts], dtype=float)
    valid = np.isfinite(yrs) & np.isfinite(freq)

    if valid.sum() < 2:
        ax.text(0.5, 0.5, "< 2 valid points", transform=ax.transAxes,
                ha="center", va="center", fontsize=E_TRANSITION_STYLE["tick_label_size"])
        ax.set_title(
            _panel_title(_display_label(pts, ds), idx),
            loc="left",
            fontsize=E_TRANSITION_STYLE["panel_label_size"],
            pad=3,
        )
        return None

    order = np.argsort(yrs[valid])
    sy, sf = yrs[valid][order], freq[valid][order]

    ax.plot(sy, sf, color=PLOT_STYLE["fit_line_color"],
            linewidth=E_TRANSITION_STYLE["time_line_width"], alpha=0.82, zorder=1)
    sc = ax.scatter(sy, sf, c=sy, cmap=PLOT_STYLE["cmap"],
                    vmin=yr_min, vmax=yr_max, s=PLOT_STYLE["point_size"],
                    edgecolor=PLOT_STYLE["point_edge_color"],
                    linewidth=E_TRANSITION_STYLE["point_edge_width"],
                    alpha=PLOT_STYLE["point_alpha"], zorder=2)

    ax.set_title(
        _panel_title(_display_label(pts, ds), idx),
        loc="left",
        fontsize=E_TRANSITION_STYLE["panel_label_size"],
        pad=3,
    )
    style_open_axes(ax)
    style_light_grid(
        ax,
        axis="both",
        color=PLOT_STYLE["grid_color"],
        linewidth=E_TRANSITION_STYLE["grid_line_width"],
    )
    ax.set_xlim(yr_min, yr_max); ax.set_ylim(f_min, f_max)
    ax.tick_params(
        axis="both",
        labelsize=E_TRANSITION_STYLE["tick_label_size"],
        length=E_TRANSITION_STYLE["tick_length"],
        width=E_TRANSITION_STYLE["tick_width"],
        pad=1.8,
    )
    return sc


def _plot_frequency_over_time(points):
    ds_list = _dataset_order(points)
    yrs = np.asarray([p.year for p in points], dtype=float)
    freq = np.asarray([p.transition_frequency for p in points], dtype=float)
    valid = np.isfinite(yrs) & np.isfinite(freq)

    yr_min = float(np.nanmin(yrs[valid]))
    yr_max = float(np.nanmax(yrs[valid]))
    f_pad = max((float(np.nanmax(freq[valid])) - float(np.nanmin(freq[valid])))
                * X_AXIS_PADDING_FRACTION, X_AXIS_MIN_PADDING)
    f_min = max(0.0, float(np.nanmin(freq[valid])) - f_pad)
    f_max = min(1.0, float(np.nanmax(freq[valid])) + f_pad)

    fig = plt.figure(figsize=(
        mm_to_inches(E_TRANSITION_STYLE["figure_width_mm"]),
        mm_to_inches(E_TRANSITION_STYLE["figure_height_mm"]),
    ))
    axes = source_panel_grid_5x2(fig, left=0.095, right=0.905, bottom=0.083,
                                 top=0.93, wspace=0.08, hspace=0.21)

    sc = None
    for i, ds in enumerate(ds_list):
        sc = _draw_time_panel(axes[i], _points_for(points, ds), ds,
                              yr_min, yr_max, f_min, f_max, i) or sc

    if sc is not None:
        _add_year_colorbar(fig, sc, axes)

    _add_transition_title(fig, show_year_note=False)
    style_source_panel_axes_5x2(axes, n_visible=len(ds_list))
    add_shared_axis_labels(fig, xlabel="Test-start year",
                           ylabel=_transition_axis_label(),
                           xlabel_y=0.035, ylabel_x=0.028,
                           fontsize=E_TRANSITION_STYLE["axis_label_size"])
    save_publication_figure(fig, _make_output_paths("frequency_over_time"),
                            dpi=FIGURE_DPI, pad_inches=0.03, print_paths=True)
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


def main():
    configure_publication_style()
    validate_data_sources(DATA_SOURCES)
    points = _collect_all(DATA_SOURCES)
    if MAKE_SKILL_RELATIONSHIP_PLOT:
        _plot_skill_relationship(points)
    if MAKE_TRANSITION_TIME_PLOT:
        _plot_frequency_over_time(points)

    valid_freq = [p.transition_frequency for p in points
                  if np.isfinite(p.transition_frequency)]
    valid_acc = [p.acc for p in points if np.isfinite(p.acc)]
    print(f"{'='*72}\nProcessed {len(points)} files from {len(_dataset_order(points))} "
          f"sources, lead {LEAD}. Valid frequencies: {len(valid_freq)}; "
          f"valid ACC: {len(valid_acc)}.")


if __name__ == "__main__":
    main()
