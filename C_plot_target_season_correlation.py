"""
Target-month and target-season Pearson correlation heatmaps.

Compares multiple ENSO forecast input-data sources.
Edit the configuration block below.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from A_basic_sources import FIGURE_ROOT, get_dl_sources, load_source_forecast_table
from plot_style import (
    AXIS_LABEL_SIZE,
    COLORBAR_TICK_SIZE,
    COMPACT_TICK_LABEL_SIZE,
    TITLE_SIZE,
    VALUE_LABEL_SIZE,
    add_shared_axis_labels,
    configure_publication_style,
    disable_axis_grid,
    panel_title,
    save_publication_figure,
    source_panel_grid_5x2,
    style_boxed_axes,
    style_colorbar,
    style_source_panel_axes_5x2,
    validate_data_sources,
)


# =============================================================================
# User configuration
# =============================================================================

BASE_YEAR = 1871
FIGURE_ID = "C"
FIGURE_NAME = "target_season_correlation"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
FIGURE_DPI = 600

MAX_LEAD_CALC = 18
MIN_SAMPLES = 3
ANNOTATE_CELLS = False

# Y-axis mode for the lead heatmap:
#   "target_month"       -> target calendar month
#   "target_season"      -> target season
#   "first_pred_month"   -> first prediction calendar month
#   "first_pred_season"  -> first prediction season
Y_MODE = "first_pred_month"

# Use None to keep the full available period. Otherwise use (year, month).
TIME_START = None
TIME_END = None

PUB_FIG_WIDTH_MM = 183
PUB_FIG_HEIGHT_MM = 180
REFERENCE_DATASET_ID = "source_1"
REFERENCE_CMAP = "viridis"
DELTA_CMAP = "PuOr"
REFERENCE_VMIN = 0.2
REFERENCE_VMAX = 1.0

DATA_SOURCES = get_dl_sources()

_MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_SEASON = ["DJF", "MAM", "JJA", "SON"]
Y_MODE_SETTINGS = {
    "target_month":       ("month",           list(range(1, 13)), _MONTH),
    "target_season":      ("target_season",    _SEASON,            _SEASON),
    "first_pred_month":   ("first_pred_month", list(range(1, 13)), _MONTH),
    "first_pred_season":  ("first_pred_season", _SEASON,           _SEASON),
}
_LABEL_MAP = {
    "target_month": "Target month",
    "target_season": "Target season",
    "first_pred_month": "First prediction month",
    "first_pred_season": "First prediction season",
}


# =============================================================================
# Data loading and preparation
# =============================================================================

def load_all_predictions(folder: Path) -> pd.DataFrame:
    df = load_source_forecast_table({"pickle_dir": Path(folder)}, base_year=BASE_YEAR)
    if TIME_START is not None or TIME_END is not None:
        ym = df["year"] * 12 + df["month"]
        if TIME_START is not None:
            df = df[ym >= TIME_START[0] * 12 + TIME_START[1]]
        if TIME_END is not None:
            df = df[ym <= TIME_END[0] * 12 + TIME_END[1]]

    return df


def compute_corr_table(df_avg: pd.DataFrame, max_lead: int) -> pd.DataFrame:
    if Y_MODE not in Y_MODE_SETTINGS:
        raise ValueError("Y_MODE must be target_month, target_season, first_pred_month, or first_pred_season.")
    y_col, y_values, y_labels = Y_MODE_SETTINGS[Y_MODE]
    corr_matrix = np.full((len(y_values), max_lead), np.nan)
    for row, y_value in enumerate(y_values):
        sub_y = df_avg[df_avg[y_col] == y_value]
        for lead in range(1, max_lead + 1):
            sub = sub_y[sub_y["leading"] == lead].dropna(subset=["pred", "real"])
            if len(sub) >= MIN_SAMPLES and sub["pred"].std() > 0 and sub["real"].std() > 0:
                corr_matrix[row, lead - 1] = np.corrcoef(sub["pred"], sub["real"])[0, 1]
    return pd.DataFrame(corr_matrix, index=y_labels, columns=[str(i) for i in range(1, max_lead + 1)])


def prepare_dataset(source: dict) -> dict:
    """Load one data source and compute both correlation matrices."""
    dataset_id = source["id"]
    label = source["label"]
    pickle_dir = source["pickle_dir"]

    print("=" * 72)
    print(f"Loading {label}")
    print(f"Pickle directory: {pickle_dir}")

    df = load_all_predictions(pickle_dir)

    available_leads = sorted(int(lead) for lead in df["leading"].unique())
    max_lead = min(MAX_LEAD_CALC, max(available_leads))
    df = df[df["leading"] <= max_lead].copy()

    season_map = {
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    }
    df["input_end_abs"] = df["abs_month"] - df["leading"]
    df["first_pred_abs"] = df["input_end_abs"] - df["input_len"] + 1
    df["first_pred_month"] = df["first_pred_abs"] % 12 + 1
    df["target_season"] = df["month"].map(season_map)
    df["first_pred_season"] = df["first_pred_month"].map(season_map)

    df_avg = (
        df.groupby(["abs_month", "leading"], as_index=False)
        .agg(
            pred=("pred", "mean"),
            real=("real", "mean"),
            month=("month", "first"),
            target_season=("target_season", "first"),
            first_pred_month=("first_pred_month", "first"),
            first_pred_season=("first_pred_season", "first"),
        )
    )

    pickle_years = sorted(int(year) for year in df["pickle_year"].unique())
    print(
        f"Pickle files: {len(pickle_years)}; "
        f"pickle years: {pickle_years[0]}-{pickle_years[-1]}"
    )
    print(f"Rows: {len(df)}; available leads: {available_leads}; plotted leads: 1-{max_lead}")

    return {
        "id": dataset_id,
        "label": label,
        "pickle_dir": pickle_dir,
        "max_lead": max_lead,
        "lead_corr": compute_corr_table(df_avg, max_lead),
    }


# =============================================================================
# Plotting and main workflow
# =============================================================================

def draw_heatmap(ax, matrix: pd.DataFrame, panel: str, title: str,
                cmap: str, norm: mpl.colors.Normalize) -> mpl.image.AxesImage:
    image = ax.imshow(matrix.values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title(panel_title(panel, title), loc="left", fontsize=TITLE_SIZE, pad=4)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, fontsize=COMPACT_TICK_LABEL_SIZE)
    ax.set_yticklabels(matrix.index, fontsize=COMPACT_TICK_LABEL_SIZE)
    ax.tick_params(length=2, width=0.55, pad=1.8)
    disable_axis_grid(ax)
    if ANNOTATE_CELLS:
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                v = matrix.iat[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=VALUE_LABEL_SIZE)
    style_boxed_axes(ax)
    return image


def plot_reference_delta_figure(results: list[dict], matrix_key: str, output_base: Path) -> list[Path]:
    """Plot SST_NOAA as reference and all other sources as deltas."""
    result_by_id = {result["id"]: result for result in results}
    labels_by_id = {result["id"]: result["label"] for result in results}
    reference = result_by_id[REFERENCE_DATASET_ID][matrix_key]
    comparison_ids = [source["id"] for source in DATA_SOURCES if source["id"] != REFERENCE_DATASET_ID]
    delta_matrices = [result_by_id[dataset_id][matrix_key] - reference for dataset_id in comparison_ids]

    max_abs_delta = max(float(np.nanmax(np.abs(delta.values))) for delta in delta_matrices)
    delta_limit = max(0.05, min(0.4, np.ceil(max_abs_delta * 20) / 20))

    configure_publication_style()

    fig = plt.figure(figsize=(PUB_FIG_WIDTH_MM / 25.4, PUB_FIG_HEIGHT_MM / 25.4))
    axes_list = source_panel_grid_5x2(
        fig,
        left=0.085,
        right=0.98,
        bottom=0.095,
        top=0.965,
        wspace=0.12,
        hspace=0.18,
    )
    panel_labels = [chr(ord("a") + index) for index in range(len(axes_list))]

    reference_norm = mpl.colors.Normalize(vmin=REFERENCE_VMIN, vmax=REFERENCE_VMAX)
    delta_norm = mpl.colors.TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)

    reference_image = draw_heatmap(
        axes_list[0],
        reference,
        "a",
        labels_by_id[REFERENCE_DATASET_ID],
        REFERENCE_CMAP,
        reference_norm,
    )

    delta_image = None
    for panel_label_value, ax, dataset_id, delta in zip(
        panel_labels[1:],
        axes_list[1:],
        comparison_ids,
        delta_matrices,
    ):
        delta_image = draw_heatmap(
            ax,
            delta,
            panel_label_value,
            f"{labels_by_id[dataset_id]} - {labels_by_id[REFERENCE_DATASET_ID]}",
            DELTA_CMAP,
            delta_norm,
        )

    if matrix_key == "lead_corr":
        x_label = "Lead time (months)"
        y_label = _LABEL_MAP[Y_MODE]

    style_source_panel_axes_5x2(axes_list, n_visible=1 + len(comparison_ids))
    add_shared_axis_labels(
        fig,
        xlabel=x_label,
        ylabel=y_label,
        xlabel_y=0.048,
        ylabel_x=0.018,
        fontsize=AXIS_LABEL_SIZE,
    )

    reference_position = axes_list[0].get_position()
    cax_reference = fig.add_axes(
        [reference_position.x1 + 0.012, reference_position.y0, 0.015, reference_position.height]
    )
    cbar_reference = fig.colorbar(reference_image, cax=cax_reference)
    style_colorbar(cbar_reference, label="Pearson r")

    cax_delta = fig.add_axes(
        [axes_list[0].get_position().x0, 0.016,
         axes_list[-1].get_position().x1 - axes_list[0].get_position().x0, 0.018],
    )
    cbar_delta = fig.colorbar(delta_image, cax=cax_delta, orientation="horizontal")
    style_colorbar(
        cbar_delta,
        label=f"Delta Pearson r vs {labels_by_id[REFERENCE_DATASET_ID]}",
        tick_labelsize=COLORBAR_TICK_SIZE,
    )

    saved_paths = save_publication_figure(
        fig,
        [output_base.with_suffix(f".{suffix}") for suffix in ["png", "pdf"]],
        dpi=FIGURE_DPI,
        pad_inches=0.02,
    )

    plt.close(fig)
    return saved_paths


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validate_data_sources(DATA_SOURCES)

    results = [prepare_dataset(source) for source in DATA_SOURCES]
    reference_label = next(source["label"] for source in DATA_SOURCES if source["id"] == REFERENCE_DATASET_ID)

    saved_paths = plot_reference_delta_figure(
        results,
        "lead_corr",
        OUTPUT_DIR / f"{FIGURE_ID}_{FIGURE_NAME}_lead_correlation_delta_vs_{reference_label}",
    )

    for path in saved_paths:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
