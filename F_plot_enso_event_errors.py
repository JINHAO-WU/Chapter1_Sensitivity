"""Analyze ENSO event amplitude and peak-month forecast errors.

The script reads pickle files that contain ``predict_value`` and ``real_value``
arrays, then creates publication-style figures for several model versions.
Edit the configuration block below when changing data sources, leads, or the
composite-event window.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from Basic_sources import get_dl_sources, list_pickle_files_by_year, load_prediction_arrays
from plot_style import (
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    PANEL_LABEL_SIZE,
    TITLE_SIZE,
    configure_publication_style,
    dataset_color as shared_dataset_color,
    style_open_axes,
    validate_data_sources,
)


# =============================================================================
# User configuration
# =============================================================================

FIGURE_ID = "F"
FIGURE_NAME = "enso_event_errors"
FIGURE_ROOT = Path("Figures")
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
FIGURE_DPI = 300

LEADS = [6]
INPUT_MONTHS = 6
COMPOSITE_MONTHS_BEFORE = 6
COMPOSITE_MONTHS_AFTER = 6
RUN_SELF_TEST = True

DATA_SOURCES = get_dl_sources()

EVENT_CLASSES = [
    "Strong El Niño",
    "Weak El Niño",
    "Strong La Niña",
    "Weak La Niña",
]

CLASS_TITLES = {
    "Strong El Niño": "Strong El Niño",
    "Weak El Niño": "Weak El Niño",
    "Weak La Niña": "Weak La Niña",
    "Strong La Niña": "Strong La Niña",
}

EVENT_GROUPS = {
    "El Niño": ["Strong El Niño", "Weak El Niño"],
    "La Niña": ["Strong La Niña", "Weak La Niña"],
}

OBSERVED_COLOR = "#222222"


# =============================================================================
# Data loading and event calculations
# =============================================================================

def load_pickle_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load prediction and observation arrays from one pickle file."""
    pred, real = load_prediction_arrays(path)
    expected_samples = 360 - INPUT_MONTHS - 18
    if pred.shape[0] != expected_samples:
        raise ValueError(
            f"{path.name}: expected {expected_samples} samples, got {pred.shape[0]}."
        )
    if max(LEADS) > pred.shape[1]:
        raise ValueError(
            f"{path.name}: requested lead {max(LEADS)}, but only {pred.shape[1]} leads exist."
        )

    return pred, real


def classify_peak(value: float) -> str:
    """Classify an ENSO event by its observed peak anomaly."""
    if value >= 1.5:
        return "Strong El Niño"
    if value > 0.5:
        return "Weak El Niño"
    if value <= -1.5:
        return "Strong La Niña"
    if value < -0.5:
        return "Weak La Niña"
    raise ValueError(f"Peak value {value} is not an ENSO event.")


def event_polarity(value: float) -> int:
    """Return 1 for warm events, -1 for cold events, and 0 for neutral months."""
    if value < -0.5:
        return -1
    if value > 0.5:
        return 1
    return 0


def iter_real_events(real: np.ndarray):
    """Identify continuous ENSO events from an observed Nino3.4 sequence."""
    start = None
    polarity = 0
    for idx, value in enumerate(real):
        current = event_polarity(float(value))
        if current == 0:
            if start is not None:
                yield start, idx, polarity
            start = None
            polarity = 0
        elif start is None or current != polarity:
            if start is not None:
                yield start, idx, polarity
            start = idx
            polarity = current

    if start is not None:
        yield start, len(real), polarity


def composite_window_for_peak(
    pred: np.ndarray,
    real: np.ndarray,
    peak_index: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return aligned prediction and observation windows around the peak month."""
    window_start = peak_index - COMPOSITE_MONTHS_BEFORE
    window_end = peak_index + COMPOSITE_MONTHS_AFTER + 1
    if window_start < 0 or window_end > len(real):
        return None
    return pred[window_start:window_end], real[window_start:window_end]


def event_metrics_for_lead(pred: np.ndarray, real: np.ndarray) -> tuple[list[dict], dict]:
    """Calculate event metrics and composite samples for one lead."""
    metrics: list[dict] = []
    composite_samples = {
        event_class: {"pred": [], "real": []}
        for event_class in EVENT_CLASSES
    }

    for start, end, polarity in iter_real_events(real):
        real_event = real[start:end]
        pred_event = pred[start:end]

        if polarity > 0:
            real_peak = int(np.nanargmax(real_event))
            pred_peak = int(np.nanargmax(pred_event))
            real_peak_value = float(real_event[real_peak])
            pred_at_real_peak = float(pred_event[real_peak])
            amplitude_underestimate = real_peak_value - pred_at_real_peak
        else:
            real_peak = int(np.nanargmin(real_event))
            pred_peak = int(np.nanargmin(pred_event))
            real_peak_value = float(real_event[real_peak])
            pred_at_real_peak = float(pred_event[real_peak])
            amplitude_underestimate = abs(real_peak_value) - abs(pred_at_real_peak)

        event_class = classify_peak(real_peak_value)
        absolute_peak_index = start + real_peak
        composite_window = composite_window_for_peak(pred, real, absolute_peak_index)
        if composite_window is not None:
            pred_window, real_window = composite_window
            composite_samples[event_class]["pred"].append(pred_window)
            composite_samples[event_class]["real"].append(real_window)

        metrics.append(
            {
                "class": event_class,
                "amplitude_underestimate": amplitude_underestimate,
                "peak_error": pred_peak - real_peak,
            }
        )

    return metrics, composite_samples


def mean_or_nan(values: list[float]) -> float:
    """Return the mean value, or NaN for an empty list."""
    return float(np.mean(values)) if values else math.nan


def empty_composite_samples() -> dict:
    """Create the nested storage used for composite event windows."""
    return {
        event_class: {"pred": [], "real": []}
        for event_class in EVENT_CLASSES
    }


def merge_composite_samples(target: dict, source: dict) -> None:
    """Append composite windows from source into target."""
    for event_class in EVENT_CLASSES:
        target[event_class]["pred"].extend(source[event_class]["pred"])
        target[event_class]["real"].extend(source[event_class]["real"])


def summarize_year_metrics(event_metrics: list[dict]) -> dict:
    """Summarize event metrics for one dataset and test-start year."""
    amplitude_values = [
        float(metric["amplitude_underestimate"])
        for metric in event_metrics
    ]
    peak_errors = [float(metric["peak_error"]) for metric in event_metrics]

    amplitude_by_class = {
        event_class: mean_or_nan(
            [
                float(metric["amplitude_underestimate"])
                for metric in event_metrics
                if metric["class"] == event_class
            ]
        )
        for event_class in EVENT_CLASSES
    }
    peak_error_by_class = {
        event_class: mean_or_nan(
            [
                float(metric["peak_error"])
                for metric in event_metrics
                if metric["class"] == event_class
            ]
        )
        for event_class in EVENT_CLASSES
    }
    peak_error_by_group = {
        group_name: mean_or_nan(
            [
                float(metric["peak_error"])
                for metric in event_metrics
                if metric["class"] in group_classes
            ]
        )
        for group_name, group_classes in EVENT_GROUPS.items()
    }

    return {
        "amplitude_by_class": amplitude_by_class,
        "peak_error_by_class": peak_error_by_class,
        "peak_error_by_group": peak_error_by_group,
        "mean_amplitude_underestimate": mean_or_nan(amplitude_values),
        "mean_peak_error": mean_or_nan(peak_errors),
    }


def analyze_pickle(path: Path) -> tuple[dict, dict]:
    """Analyze one pickle file across all configured leads."""
    pred, real = load_pickle_arrays(path)

    event_metrics: list[dict] = []
    composite_samples = empty_composite_samples()
    for lead in LEADS:
        col = lead - 1
        lead_metrics, lead_composites = event_metrics_for_lead(pred[:, col], real[:, col])
        event_metrics.extend(lead_metrics)
        merge_composite_samples(composite_samples, lead_composites)

    return summarize_year_metrics(event_metrics), composite_samples


def collect_all_results(data_sources: list[dict]) -> tuple[list[dict], dict, dict]:
    """Analyze every configured data source and collect all plot inputs."""
    all_results: list[dict] = []
    composites_by_dataset = {
        source["id"]: empty_composite_samples()
        for source in data_sources
    }
    observed_composites = empty_composite_samples()

    for source in data_sources:
        dataset_id = source["id"]
        label = source["label"]
        files_by_year = list_pickle_files_by_year(source["pickle_dir"])
        years = sorted(files_by_year)

        print("=" * 72)
        print(f"{label}: found {len(years)} pickle files")
        print(f"Years: {years[0]}-{years[-1]}")
        print(f"Directory: {source['pickle_dir']}")
        print("=" * 72)

        for year in years:
            result, composite_samples = analyze_pickle(files_by_year[year])
            result["dataset"] = dataset_id
            result["year"] = year
            all_results.append(result)

            merge_composite_samples(composites_by_dataset[dataset_id], composite_samples)
            for event_class in EVENT_CLASSES:
                observed_composites[event_class]["real"].extend(
                    composite_samples[event_class]["real"]
                )

            print(
                f"{label:18s} year={year} "
                f"mean_amp_under={result['mean_amplitude_underestimate']:.4f} "
                f"mean_peak_error={result['mean_peak_error']:.4f}"
            )

    return all_results, composites_by_dataset, observed_composites


# =============================================================================
# Plotting
# =============================================================================

def lead_token() -> str:
    """Return the compact lead token used in output filenames."""
    return "-".join(str(lead) for lead in LEADS)


def years_and_values(results: list[dict], dataset_id: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted years and values for one dataset and result field."""
    rows = [row for row in results if row["dataset"] == dataset_id]
    rows = sorted(rows, key=lambda row: row["year"])
    return (
        np.array([row["year"] for row in rows]),
        np.array([row[field] for row in rows], dtype=float),
    )


def class_years_and_values(
    results: list[dict],
    dataset_id: str,
    event_class: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted years and class-specific metric values."""
    rows = [row for row in results if row["dataset"] == dataset_id]
    rows = sorted(rows, key=lambda row: row["year"])
    return (
        np.array([row["year"] for row in rows]),
        np.array([row[metric_key][event_class] for row in rows], dtype=float),
    )


def group_years_and_values(
    results: list[dict],
    dataset_id: str,
    event_group: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted years and grouped peak-error values."""
    rows = [row for row in results if row["dataset"] == dataset_id]
    rows = sorted(rows, key=lambda row: row["year"])
    return (
        np.array([row["year"] for row in rows]),
        np.array([row["peak_error_by_group"][event_group] for row in rows], dtype=float),
    )


def dataset_color(dataset_id: str) -> str:
    """Return a stable color for one data source."""
    return shared_dataset_color(dataset_id)


def mean_composite(windows: list[np.ndarray]) -> np.ndarray | None:
    """Return the mean composite time series, or None if no windows are available."""
    if not windows:
        return None
    return np.nanmean(np.vstack(windows), axis=0)


def plot_amplitude_by_class(
    results: list[dict],
    data_sources: list[dict],
    output_path: Path,
) -> None:
    """Plot event-mean amplitude underestimation by ENSO class."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 8.0),
        dpi=FIGURE_DPI,
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    for ax, event_class in zip(axes, EVENT_CLASSES):
        for source in data_sources:
            dataset_id = source["id"]
            label = source["label"]
            years, values = class_years_and_values(
                results,
                dataset_id,
                event_class,
                "amplitude_by_class",
            )
            ax.plot(
                years,
                values,
                color=dataset_color(dataset_id),
                linewidth=1.9,
                label=label,
            )

        ax.axhline(0, color="#555555", linewidth=0.8, linestyle=(0, (5, 4)))
        ax.set_title(CLASS_TITLES[event_class], fontsize=TITLE_SIZE, fontweight="bold")
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, linestyle=":")
        ax.grid(True, axis="x", color="#eeeeee", linewidth=0.5, linestyle=":")
        ax.tick_params(axis="both", direction="in", labelsize=8)
        style_open_axes(ax)

    for ax in axes[2:]:
        ax.set_xlabel("Test-set start year", fontsize=AXIS_LABEL_SIZE)
    for ax in axes[::2]:
        ax.set_ylabel("Amplitude underestimation", fontsize=AXIS_LABEL_SIZE)

    legend_handles = [
        Line2D([0], [0], color=dataset_color(source["id"]), linewidth=2.0, label=source["label"])
        for source in data_sources
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(len(legend_handles), 5),
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        fontsize=LEGEND_SIZE,
    )
    fig.suptitle(
        f"ENSO event amplitude underestimation by class (lead {lead_token()}M)",
        fontsize=PANEL_LABEL_SIZE,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_peak_month_error(
    results: list[dict],
    data_sources: list[dict],
    output_path: Path,
) -> None:
    """Plot signed mean peak-month error for El Niño and La Niña groups."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 4.8),
        dpi=FIGURE_DPI,
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    for ax, event_group in zip(axes, EVENT_GROUPS):
        for source in data_sources:
            dataset_id = source["id"]
            label = source["label"]
            years, values = group_years_and_values(results, dataset_id, event_group)
            ax.plot(
                years,
                values,
                color=dataset_color(dataset_id),
                linewidth=1.9,
                label=label,
            )

        ax.axhline(0, color="#555555", linewidth=0.8, linestyle=(0, (5, 4)))
        ax.set_title(event_group, fontsize=TITLE_SIZE, fontweight="bold")
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, linestyle=":")
        ax.grid(True, axis="x", color="#eeeeee", linewidth=0.5, linestyle=":")
        ax.tick_params(axis="both", direction="in", labelsize=8)
        style_open_axes(ax)

    for ax in axes:
        ax.set_xlabel("Test-set start year", fontsize=AXIS_LABEL_SIZE)
    axes[0].set_ylabel("Peak-month error (months)", fontsize=AXIS_LABEL_SIZE)

    legend_handles = [
        Line2D([0], [0], color=dataset_color(source["id"]), linewidth=2.0, label=source["label"])
        for source in data_sources
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(len(legend_handles), 5),
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        fontsize=LEGEND_SIZE,
    )
    fig.suptitle(
        f"ENSO event peak-month error by ENSO phase (lead {lead_token()}M)",
        fontsize=PANEL_LABEL_SIZE,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_figures(
    results: list[dict],
    composites_by_dataset: dict,
    observed_composites: dict,
    data_sources: list[dict],
) -> list[Path]:
    """Create all output figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    token = lead_token()

    amplitude_path = OUTPUT_DIR / (
        f"{FIGURE_ID}_{FIGURE_NAME}_amplitude_underestimate_by_class_lead{token}.png"
    )
    peak_error_path = OUTPUT_DIR / (
        f"{FIGURE_ID}_{FIGURE_NAME}_peak_month_error_by_phase_lead{token}.png"
    )

    plot_amplitude_by_class(results, data_sources, amplitude_path)
    plot_peak_month_error(results, data_sources, peak_error_path)

    return [amplitude_path, peak_error_path]


# =============================================================================
# Self-tests and main workflow
# =============================================================================

def run_self_tests() -> None:
    """Run lightweight checks for event classification and peak alignment."""
    threshold_cases = {
        -1.5: "Strong La Niña",
        -0.6: "Weak La Niña",
        0.6: "Weak El Niño",
        1.5: "Strong El Niño",
    }
    for value, expected in threshold_cases.items():
        actual = classify_peak(value)
        if actual != expected:
            raise AssertionError(
                f"classify_peak({value}) returned {actual}, expected {expected}."
            )

    global COMPOSITE_MONTHS_BEFORE, COMPOSITE_MONTHS_AFTER
    original_before = COMPOSITE_MONTHS_BEFORE
    original_after = COMPOSITE_MONTHS_AFTER
    COMPOSITE_MONTHS_BEFORE = 1
    COMPOSITE_MONTHS_AFTER = 1
    try:
        real = np.array([0.0, 0.7, 1.4, 1.6, 1.2, 0.2, -0.6, -1.7, -1.1, 0.0])
        pred = np.array([0.0, 0.5, 1.0, 1.1, 1.7, 0.1, -0.4, -1.0, -1.8, 0.0])
        metrics, composites = event_metrics_for_lead(pred, real)

        if [metric["class"] for metric in metrics] != ["Strong El Niño", "Strong La Niña"]:
            raise AssertionError("Event classes should be based on observed event peaks.")
        if [metric["peak_error"] for metric in metrics] != [1, 1]:
            raise AssertionError("Both synthetic peak errors should be +1 month.")

        warm_real = composites["Strong El Niño"]["real"][0]
        if not np.allclose(warm_real, np.array([1.4, 1.6, 1.2])):
            raise AssertionError("Composite window should be centered on observed peak month.")
    finally:
        COMPOSITE_MONTHS_BEFORE = original_before
        COMPOSITE_MONTHS_AFTER = original_after


def main() -> None:
    configure_publication_style()
    validate_data_sources(DATA_SOURCES)
    if RUN_SELF_TEST:
        run_self_tests()

    results, composites_by_dataset, observed_composites = collect_all_results(DATA_SOURCES)
    output_paths = write_figures(
        results,
        composites_by_dataset,
        observed_composites,
        DATA_SOURCES,
    )

    print("=" * 72)
    for output_path in output_paths:
        print(output_path)
    print("=" * 72)


if __name__ == "__main__":
    main()
