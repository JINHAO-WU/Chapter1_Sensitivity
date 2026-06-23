"""
Target-month and target-season Pearson correlation figures.

This script compares multiple ENSO forecast input-data sources. It loads all
pickle files from each configured folder, averages duplicate forecasts for the
same absolute target month and lead, then computes Pearson correlation by lead
and by target/prediction-start calendar grouping.

Edit the configuration block below to change data sources, leads, time ranges,
or plotting modes. The script is intended to run directly as a compact research
plotting script rather than a command-line tool.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import AXIS_LABEL_SIZE, TITLE_SIZE, configure_publication_style, style_boxed_axes, validate_data_sources


# =============================================================================
# User configuration
# =============================================================================

BASE_YEAR = 1871
OUTPUT_DIR = Path(r"target_season_figures")
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

DATA_SOURCES = [
    {
        "id": "source_1",
        "label": "SST_NOAA",
        "pickle_dir": Path(
            r"E:/OneDrive - University of Leeds/A-Research/Study_timeseies/TL_CMIP/File/"
            r"pickle_HamCNN_input6_var1_sst_NOAA"
        ),
    },
    {
        "id": "source_2",
        "label": "SST_HadI",
        "pickle_dir": Path(
            r"E:/OneDrive - University of Leeds/A-Research/Study_timeseies/TL_CMIP/File/"
            r"pickle_HamCNN_input6_var1_sst_HadI"
        ),
    },
    {
        "id": "source_3",
        "label": "SST_NOAA_PO",
        "pickle_dir": Path(
            r"E:/OneDrive - University of Leeds/A-Research/Study_timeseies/TL_CMIP/File/"
            r"pickle_HamCNN_input6_var1_sst_NOAA_PO"
        ),
    },
    {
        "id": "source_4",
        "label": "SST_OHC300_NOAA",
        "pickle_dir": Path(
            r"E:/OneDrive - University of Leeds/A-Research/Study_timeseies/TL_CMIP/File/"
            r"pickle_HamCNN_input6_var2_sst_ohc300_NOAA"
        ),
    },
    {
        "id": "source_5",
        "label": "SST_NOAA_5MIROC6",
        "pickle_dir": Path(
            r"E:/OneDrive - University of Leeds/A-Research/Study_timeseies/TL_CMIP/File/"
            r"pickle_HamCNN_input6_var1_sst_NOAA_5MIROC6"
        ),
    },
]

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_LABELS = ["DJF", "MAM", "JJA", "SON"]
TARGET_MONTH_ORDER = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
TARGET_MONTH_LABELS = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]


# =============================================================================
# Data loading and preparation
# =============================================================================

def load_all_predictions(folder: Path) -> pd.DataFrame:
    """
    Load all pickle files in one folder into a long-format prediction table.

    Each row represents one forecast value for one absolute target month and
    one leading time.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Pickle directory does not exist: {folder}")

    pickle_files = sorted(folder.glob("*.pickle"))
    if not pickle_files:
        raise FileNotFoundError(f"No pickle files found in: {folder}")

    records = []
    for path in pickle_files:
        year_match = re.search(r"_(\d{4})_", path.name)
        if not year_match:
            continue

        start_year = int(year_match.group(1))
        input_match = re.search(r"input(\d+)", path.name)
        input_len = int(input_match.group(1)) if input_match else 6
        with path.open("rb") as fh:
            data = pickle.load(fh)

        pred = np.asarray(data["predict_value"])
        real = np.asarray(data["real_value"])

        if pred.shape != real.shape:
            raise ValueError(
                f"{path.name}: predict_value and real_value have different shapes "
                f"{pred.shape} vs {real.shape}."
            )
        if pred.ndim != 2:
            raise ValueError(f"{path.name}: expected a 2D array, got {pred.ndim}D.")

        n_samples, n_lead = pred.shape
        base_offset = (start_year - BASE_YEAR) * 12

        for sample_index in range(n_samples):
            for lead in range(1, n_lead + 1):
                abs_month = base_offset + sample_index + input_len + (lead - 1)
                records.append(
                    {
                        "abs_month": abs_month,
                        "leading": lead,
                        "input_len": input_len,
                        "pickle_year": start_year,
                        "pred": pred[sample_index, lead - 1],
                        "real": real[sample_index, lead - 1],
                    }
                )

    if not records:
        raise ValueError(f"No usable pickle files with parseable years in: {folder}")

    df = pd.DataFrame(records)
    df["year"] = BASE_YEAR + df["abs_month"] // 12
    df["month"] = df["abs_month"] % 12 + 1
    if TIME_START is not None or TIME_END is not None:
        ym = df["year"] * 12 + df["month"]
        if TIME_START is not None:
            df = df[ym >= TIME_START[0] * 12 + TIME_START[1]]
        if TIME_END is not None:
            df = df[ym <= TIME_END[0] * 12 + TIME_END[1]]

    return df


def compute_corr_table(df_avg: pd.DataFrame, mode: str, max_lead: int) -> pd.DataFrame:
    """Compute one Pearson-correlation matrix."""
    if mode == "lead":
        settings = {
            "target_month": ("month", list(range(1, 13)), MONTH_LABELS),
            "target_season": ("target_season", SEASON_LABELS, SEASON_LABELS),
            "first_pred_month": ("first_pred_month", list(range(1, 13)), MONTH_LABELS),
            "first_pred_season": ("first_pred_season", SEASON_LABELS, SEASON_LABELS),
        }
        if Y_MODE not in settings:
            raise ValueError("Y_MODE must be target_month, target_season, first_pred_month, or first_pred_season.")

        y_col, y_values, y_labels = settings[Y_MODE]
        corr_matrix = np.full((len(y_values), max_lead), np.nan)
        for row, y_value in enumerate(y_values):
            sub_y = df_avg[df_avg[y_col] == y_value]
            for lead in range(1, max_lead + 1):
                sub = sub_y[sub_y["leading"] == lead].dropna(subset=["pred", "real"])
                if len(sub) >= MIN_SAMPLES and sub["pred"].std() > 0 and sub["real"].std() > 0:
                    corr_matrix[row, lead - 1] = np.corrcoef(sub["pred"], sub["real"])[0, 1]
        return pd.DataFrame(corr_matrix, index=y_labels, columns=[str(i) for i in range(1, max_lead + 1)])

    if mode == "target_vs_first":
        corr_matrix = np.full((12, 12), np.nan)
        for row, first_month in enumerate(range(1, 13)):
            for col, target_month in enumerate(TARGET_MONTH_ORDER):
                sub = df_avg[
                    (df_avg["first_pred_month"] == first_month)
                    & (df_avg["month"] == target_month)
                ].dropna(subset=["pred", "real"])
                if len(sub) >= MIN_SAMPLES and sub["pred"].std() > 0 and sub["real"].std() > 0:
                    corr_matrix[row, col] = np.corrcoef(sub["pred"], sub["real"])[0, 1]
        return pd.DataFrame(corr_matrix, index=MONTH_LABELS, columns=TARGET_MONTH_LABELS)

    raise ValueError("mode must be lead or target_vs_first.")


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
        "lead_corr": compute_corr_table(df_avg, "lead", max_lead),
        "target_vs_first_corr": compute_corr_table(df_avg, "target_vs_first", max_lead),
    }


# =============================================================================
# Plotting and main workflow
# =============================================================================

def draw_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    panel_label: str,
    title: str,
    cmap: str,
    norm: mpl.colors.Normalize,
) -> mpl.image.AxesImage:
    """Draw one compact heatmap panel."""
    image = ax.imshow(matrix.values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title(f"({panel_label}) {title}", loc="left", fontsize=TITLE_SIZE, pad=4)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, fontsize=7)
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.tick_params(length=2, width=0.55, pad=1.8)

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    if ANNOTATE_CELLS:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix.iat[row, col]
                if not np.isnan(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=4.5)

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
    grid = fig.add_gridspec(
        3,
        2,
        left=0.06,
        right=0.98,
        bottom=0.105,
        top=0.95,
        wspace=0.08,
        hspace=0.28,
    )
    top_row = grid[0, :].get_position(fig)
    panel_width = top_row.width * 0.54
    panel_height = top_row.height * 0.92
    panel_left = top_row.x0 + (top_row.width - panel_width) / 2
    panel_bottom = top_row.y0 + (top_row.height - panel_height) / 2
    axes = {
        "a": fig.add_axes([panel_left, panel_bottom, panel_width, panel_height]),
        "b": fig.add_subplot(grid[1, 0]),
        "c": fig.add_subplot(grid[1, 1]),
        "d": fig.add_subplot(grid[2, 0]),
        "e": fig.add_subplot(grid[2, 1]),
    }

    reference_norm = mpl.colors.Normalize(vmin=REFERENCE_VMIN, vmax=REFERENCE_VMAX)
    delta_norm = mpl.colors.TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)

    reference_image = draw_heatmap(
        axes["a"],
        reference,
        "a",
        labels_by_id[REFERENCE_DATASET_ID],
        REFERENCE_CMAP,
        reference_norm,
    )

    delta_image = None
    for panel, dataset_id, delta in zip(["b", "c", "d", "e"], comparison_ids, delta_matrices):
        delta_image = draw_heatmap(
            axes[panel],
            delta,
            panel,
            f"{labels_by_id[dataset_id]} - {labels_by_id[REFERENCE_DATASET_ID]}",
            DELTA_CMAP,
            delta_norm,
        )

    if matrix_key == "lead_corr":
        x_label = "Lead time (months)"
        y_label = {
            "target_month": "Target month",
            "target_season": "Target season",
            "first_pred_month": "First prediction month",
            "first_pred_season": "First prediction season",
        }[Y_MODE]
    else:
        x_label = "Target month"
        y_label = "First prediction month"

    for panel, ax in axes.items():
        if panel == "a":
            ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, labelpad=4)
        elif panel in {"b", "d"}:
            ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, labelpad=4)
        else:
            ax.set_yticklabels([])
        if panel in {"d", "e"}:
            ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE, labelpad=4)

    cax_reference = fig.add_axes([panel_left + panel_width + 0.012, panel_bottom, 0.015, panel_height])
    cbar_reference = fig.colorbar(reference_image, cax=cax_reference)
    cbar_reference.set_label("Pearson r", fontsize=7, labelpad=3)
    cbar_reference.ax.tick_params(labelsize=6.5, length=2, width=0.55, pad=1.8)
    style_boxed_axes(cbar_reference.ax)

    cax_delta = fig.add_axes([0.23, 0.038, 0.54, 0.015])
    cbar_delta = fig.colorbar(delta_image, cax=cax_delta, orientation="horizontal")
    cbar_delta.set_label(f"Delta Pearson r vs {labels_by_id[REFERENCE_DATASET_ID]}", fontsize=7, labelpad=3)
    cbar_delta.ax.tick_params(labelsize=6.5, length=2, width=0.55, pad=1.8)
    style_boxed_axes(cbar_delta.ax)

    saved_paths = []
    for suffix in ["png", "pdf"]:
        path = output_base.with_suffix(f".{suffix}")
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if suffix == "png":
            save_kwargs["dpi"] = FIGURE_DPI
        fig.savefig(path, **save_kwargs)
        saved_paths.append(path)

    plt.close(fig)
    return saved_paths


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validate_data_sources(DATA_SOURCES)

    results = [prepare_dataset(source) for source in DATA_SOURCES]
    reference_label = next(source["label"] for source in DATA_SOURCES if source["id"] == REFERENCE_DATASET_ID)

    saved_paths = []
    saved_paths.extend(
        plot_reference_delta_figure(
            results,
            "lead_corr",
            OUTPUT_DIR / f"paper_lead_correlation_delta_vs_{reference_label}",
        )
    )
    saved_paths.extend(
        plot_reference_delta_figure(
            results,
            "target_vs_first_corr",
            OUTPUT_DIR / f"paper_target_vs_first_delta_vs_{reference_label}",
        )
    )

    print("=" * 72)
    for path in saved_paths:
        print(f"Saved figure: {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
