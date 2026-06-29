"""
Confusion-matrix and class-metric figures for ENSO event forecasts.

This script compares multiple input-data sources. It computes each metric from
the full sample only; no leave-one-out event exclusion is used.

Edit the configuration block below to change data sources, leads,
classification type, or metric choices.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from A_basic_sources import FIGURE_ROOT, get_dl_sources, load_source_forecast_table
from plot_style import (
    AXIS_LABEL_SIZE,
    EVENT_COLORS,
    PANEL_LABEL_SIZE,
    TITLE_SIZE,
    configure_publication_style,
    style_boxed_axes,
    style_open_axes,
    validate_data_sources,
)


# =============================================================================
# User configuration
# =============================================================================

BASE_YEAR = 1871
FIGURE_ID = "G"
FIGURE_NAME = "enso_classification_metrics"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
FIGURE_DPI = 600
OUTPUT_FORMATS = ("png", "pdf")

# Double-column publication figure size.
PUB_FIG_WIDTH_MM = 183
PUB_FIG_HEIGHT_MM = 245
PANEL_TITLE_FONT_SIZE = TITLE_SIZE

# Set to None to use all leads available in the loaded pickle files.
LEADS = [6]

# Event classification type: 3, 5, or 7.
N_TYPE = 5

# Per-class metric: "recall", "precision", or "f1".
CLASS_METRIC = "f1"

DATA_SOURCES = get_dl_sources()
    
# Colorblind-friendly categories, ordered from strong El Niño to strong La Niña.


# =============================================================================
# Data loading
# =============================================================================

def load_all_predictions(folder: Path) -> pd.DataFrame:
    """
    Load all pickle files in one folder into a long-format prediction table.

    Each row corresponds to one absolute target month and one leading month.
    """
    return load_source_forecast_table({"pickle_dir": Path(folder)}, base_year=BASE_YEAR)


# =============================================================================
# ENSO event classification
# =============================================================================

def classify_Niño_event_3type(Niño_value: float) -> str:
    """Classify Niño3.4 value into El Niño, Neutral, or La Niña."""
    if Niño_value >= 0.5:
        return "El_Niño"
    if Niño_value > -0.5:
        return "Neutral"
    return "La_Niña"


def classify_Niño_event_5type(Niño_value: float) -> str:
    """Classify Niño3.4 value into five ENSO intensity categories."""
    if Niño_value >= 1.5:
        return "Strong_El_Niño"
    if Niño_value >= 0.5:
        return "Weak_El_Niño"
    if Niño_value >= -0.5:
        return "Neutral"
    if Niño_value >= -1.5:
        return "Weak_La_Niña"
    return "Strong_La_Niña"


def classify_Niño_event_7type(Niño_value: float) -> str:
    """Classify Niño3.4 value into seven ENSO intensity categories."""
    if Niño_value >= 2:
        return "Very_Strong_El_Niño"
    if Niño_value >= 1.5:
        return "Strong_El_Niño"
    if Niño_value >= 0.5:
        return "Weak_El_Niño"
    if Niño_value >= -0.5:
        return "Neutral"
    if Niño_value >= -1.5:
        return "Weak_La_Niña"
    if Niño_value >= -2:
        return "Strong_La_Niña"
    return "Very_Strong_La_Niña"


def event_settings(n_type: int) -> tuple[list[str], callable]:
    """Return event order and classifier for the selected category count."""
    if n_type == 7:
        return (
            [
                "Very_Strong_El_Niño",
                "Strong_El_Niño",
                "Weak_El_Niño",
                "Neutral",
                "Weak_La_Niña",
                "Strong_La_Niña",
                "Very_Strong_La_Niña",
            ],
            classify_Niño_event_7type,
        )
    if n_type == 5:
        return (
            [
                "Strong_El_Niño",
                "Weak_El_Niño",
                "Neutral",
                "Weak_La_Niña",
                "Strong_La_Niña",
            ],
            classify_Niño_event_5type,
        )
    if n_type == 3:
        return (["El_Niño", "Neutral", "La_Niña"], classify_Niño_event_3type)

    raise ValueError("N_TYPE must be 3, 5, or 7.")


# =============================================================================
# Metric calculation
# =============================================================================

def normalize_requested_leads(df: pd.DataFrame, requested_leads: list[int] | None) -> list[int]:
    """Return requested leads after checking that each exists in the data."""
    available_leads = sorted(int(lead) for lead in df["leading"].unique())
    if requested_leads is None:
        return available_leads

    missing = sorted(set(requested_leads) - set(available_leads))
    if missing:
        raise ValueError(
            f"Requested leads {missing} are not available. "
            f"Available leads: {available_leads}"
        )
    return list(requested_leads)


def calculate_confusion_metrics(
    df: pd.DataFrame,
    n_type: int,
    class_metric: str = "recall",
    leads: list[int] | None = None,
) -> tuple[list[str], dict[int, dict[str, object]]]:
    """Compute full-sample confusion matrices and class metrics for each lead."""
    if class_metric not in {"recall", "precision", "f1"}:
        raise ValueError('CLASS_METRIC must be "recall", "precision", or "f1".')

    event_order, classify_fn = event_settings(n_type)
    selected_leads = normalize_requested_leads(df, leads)
    results = {}
    for leading in selected_leads:
        monthly = (
            df.loc[df["leading"] == leading]
            .groupby("abs_month", as_index=False)[["pred", "real"]]
            .mean()
            .dropna(subset=["pred", "real"])
        )
        monthly["real_class"] = monthly["real"].apply(classify_fn)
        monthly["pred_class"] = monthly["pred"].apply(classify_fn)

        raw_matrix = confusion_matrix(
            monthly["real_class"], monthly["pred_class"], labels=event_order
        )
        normalized_matrix = confusion_matrix(
            monthly["real_class"], monthly["pred_class"], labels=event_order, normalize="true"
        )

        class_scores = []
        for index in range(len(event_order)):
            true_positive = raw_matrix[index, index]
            actual_count = raw_matrix[index, :].sum()
            predicted_count = raw_matrix[:, index].sum()
            precision = true_positive / predicted_count if predicted_count else 0.0
            recall = true_positive / actual_count if actual_count else np.nan

            if class_metric == "precision":
                score = precision if actual_count else np.nan
            elif class_metric == "recall":
                score = recall
            elif actual_count == 0:
                score = np.nan
            elif precision + recall == 0:
                score = 0.0
            else:
                score = 2 * precision * recall / (precision + recall)
            class_scores.append(float(score) if not np.isnan(score) else np.nan)

        results[leading] = {
            "confusion": np.nan_to_num(normalized_matrix, nan=0.0),
            "scores": class_scores,
            "n_samples": len(monthly),
        }

    return event_order, results


# =============================================================================
# Publication plotting and main workflow
# =============================================================================

def plot_all_datasets_figure(
    dataset_results: list[dict],
    lead: int,
    event_order: list[str],
    output_base: Path,
) -> list[Path]:
    """Create one compact 5-by-2 figure for all configured data sources."""
    figure = plt.figure(
        figsize=(PUB_FIG_WIDTH_MM / 25.4, PUB_FIG_HEIGHT_MM / 25.4),
        facecolor="white",
    )
    grid = figure.add_gridspec(
        len(dataset_results), 2,
        width_ratios=[1.0, 1.08],
        left=0.155, right=0.94, bottom=0.075, top=0.97,
        wspace=0.12, hspace=0.28,
    )
    x_tick_labels = [event.replace("_", "\n") for event in event_order]
    y_tick_labels = [event.replace("_", " ") for event in event_order]
    metric_name = {"recall": "Recall", "precision": "Precision", "f1": "F1-score"}[CLASS_METRIC]
    x_positions = np.arange(len(event_order)) + 0.5
    image = None
    figure.text(
        0.155,
        0.988,
        f"Lead {lead}M",
        ha="left",
        va="top",
        fontsize=PANEL_TITLE_FONT_SIZE,
    )

    for row_index, result in enumerate(dataset_results):
        dataset_label = result["label"]
        metrics = result["metrics_by_lead"][lead]
        confusion = metrics["confusion"]
        class_scores = metrics["scores"]
        panel_label = f"({chr(ord('a') + row_index)})"
        show_shared_x_labels = row_index == len(dataset_results) - 1
        panel_title = f"{panel_label} {dataset_label}"

        confusion_ax = figure.add_subplot(grid[row_index, 0])
        score_ax = figure.add_subplot(grid[row_index, 1])
        image = confusion_ax.imshow(
            confusion, cmap="Blues", vmin=0.0, vmax=1.0,
            aspect="auto", interpolation="nearest",
        )
        confusion_ax.set_xticks(range(len(event_order)))
        confusion_ax.set_xticklabels(x_tick_labels if show_shared_x_labels else [], fontsize=7.5)
        confusion_ax.set_yticks(range(len(event_order)), y_tick_labels, fontsize=7.5)
        confusion_ax.tick_params(length=2.2, width=0.6, pad=1.5)
        if show_shared_x_labels:
            confusion_ax.set_xlabel("Predicted", fontsize=AXIS_LABEL_SIZE, labelpad=3)
        confusion_ax.set_ylabel("Real", fontsize=AXIS_LABEL_SIZE, labelpad=3)
        confusion_ax.set_title(
            panel_title,
            fontsize=PANEL_TITLE_FONT_SIZE,
            pad=4,
        )
        for matrix_row in range(confusion.shape[0]):
            for column_index in range(confusion.shape[1]):
                value = confusion[matrix_row, column_index]
                confusion_ax.text(
                    column_index, matrix_row, f"{value:.2f}",
                    ha="center", va="center", fontsize=8, fontweight="semibold",
                    color="white" if value >= 0.55 else "#1A1A1A",
                )
        style_boxed_axes(confusion_ax)

        bars = score_ax.bar(
            x_positions, class_scores, width=0.74,
            color=EVENT_COLORS[: len(event_order)], edgecolor="#333333", linewidth=0.7,
        )
        for bar, value in zip(bars, class_scores):
            label = "NA" if np.isnan(value) else f"{value:.3f}"
            label_y = 0.025 if np.isnan(value) or value == 0 else min(value + 0.035, 1.045)
            score_ax.text(
                bar.get_x() + bar.get_width() / 2, label_y, label,
                ha="center", va="bottom", fontsize=8, fontweight="semibold",
                color="#666666" if np.isnan(value) else "#1A1A1A",
            )
        score_ax.set_xlim(0, len(event_order))
        score_ax.set_ylim(0, 1.08)
        score_ax.set_xticks(x_positions)
        score_ax.set_xticklabels(x_tick_labels if show_shared_x_labels else [], fontsize=7.5)
        score_ax.set_yticks(np.arange(0, 1.01, 0.2))
        score_ax.tick_params(axis="y", labelsize=8, length=2.2, width=0.6, pad=1.5)
        score_ax.tick_params(axis="x", length=2.2, width=0.6, pad=1.5)
        score_ax.set_title(
            f"{metric_name}", fontsize=PANEL_TITLE_FONT_SIZE, pad=4
        )
        score_ax.grid(axis="y", color="#C9C9C9", linewidth=0.55, linestyle="--", alpha=0.65)
        score_ax.set_axisbelow(True)
        style_open_axes(score_ax)

    colorbar_ax = figure.add_axes([0.42, 0.025, 0.20, 0.012])
    colorbar = figure.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.ax.tick_params(labelsize=6.5, length=1.8, width=0.55, pad=1.2)
    style_boxed_axes(colorbar.ax)

    saved_paths = []
    for file_format in OUTPUT_FORMATS:
        path = output_base.with_suffix(f".{file_format}")
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if file_format == "png":
            save_kwargs["dpi"] = FIGURE_DPI
        figure.savefig(path, **save_kwargs)
        saved_paths.append(path)
    plt.close(figure)
    return saved_paths


def main() -> None:
    configure_publication_style()
    validate_data_sources(DATA_SOURCES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_results = []
    for source in DATA_SOURCES:
        print("=" * 72)
        print(f"Loading {source['label']}")
        df = load_all_predictions(source["pickle_dir"])
        event_order, metrics_by_lead = calculate_confusion_metrics(
            df, N_TYPE, CLASS_METRIC, LEADS
        )

        pickle_years = sorted(int(year) for year in df["pickle_year"].unique())
        print(f"Pickle files: {len(pickle_years)}; pickle years: {pickle_years[0]}-{pickle_years[-1]}")
        print(f"Rows: {len(df)}; plotted leads: {list(metrics_by_lead)}")

        dataset_results.append(
            {
                "id": source["id"],
                "label": source["label"],
                "metrics_by_lead": metrics_by_lead,
            }
        )

    common_leads = list(dataset_results[0]["metrics_by_lead"])
    for result in dataset_results[1:]:
        if list(result["metrics_by_lead"]) != common_leads:
            raise ValueError("All data sources must contain the same leads for combined plotting.")

    saved_paths = []
    for lead in common_leads:
        output_base = OUTPUT_DIR / (
            f"{FIGURE_ID}_{FIGURE_NAME}_type{N_TYPE}_{CLASS_METRIC}_all_sources_lead{lead}"
        )
        saved_paths.extend(
            plot_all_datasets_figure(dataset_results, lead, event_order, output_base)
        )

    print("=" * 72)
    for path in saved_paths:
        print(f"Saved figure: {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
