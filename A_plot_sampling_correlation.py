"""
Sampling-based Pearson correlation for ENSO forecast skill across input-data sources.

Figure 1: overview line plot at one lead
Figure 2: small-multiple time series with 95% CI per source
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from A_basic_sources import (
    FIGURE_ROOT,
    get_dl_sources,
    list_pickle_files_by_year,
    load_prediction_arrays,
    parse_start_year,
)
from plot_style import (
    A_SAMPLING_STYLE,
    DEFAULT_FIGURE_DPI,
    add_compact_figure_legend,
    add_shared_axis_labels,
    configure_publication_style,
    dataset_color,
    figure_output_paths,
    mm_to_inches,
    panel_title,
    panel_title_only,
    save_publication_figure,
    source_panel_grid_5x2,
    style_light_grid,
    style_open_axes,
    style_source_panel_axes_5x2,
    validate_data_sources,
)


# =============================================================================
# User configuration
# =============================================================================

FIGURE_ID = "A"
FIGURE_NAME = "sampling_correlation"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
FIGURE_DPI = DEFAULT_FIGURE_DPI
PUB_FIG_WIDTH_MM = 183
PUB_FIG_HEIGHT_MM = 240
COMPARISON_FIG_HEIGHT_MM = 180

LEADS = [6]
COMPARISON_LEAD = 6
N_BOOTSTRAP = 5000
ALPHA = 0.05
RANDOM_SEED = 42

# "bootstrap_mean" or "observed"
FIGURE1_VALUE = "bootstrap_mean"
FIGURE1_REFERENCE_R = 0.5
REFERENCE_DATASET_ID = "source_1"

PLOT_STYLE = A_SAMPLING_STYLE

VALUE_OPTIONS = {
    "bootstrap_mean": ("r_mean", "bootstrap_mean"),
    "observed": ("observed_r", "observed"),
}

DATA_SOURCES = get_dl_sources(sample_size=60)


# =============================================================================
# Statistical helpers
# =============================================================================

def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation using NumPy."""
    return float(np.corrcoef(x, y)[0, 1])


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Return root mean squared error."""
    return float(np.sqrt(np.mean((x - y) ** 2)))


def bootstrap_like_correlation(
    x: np.ndarray,
    y: np.ndarray,
    sample_size: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw paired samples without replacement and compute Pearson r and RMSE.

    Each bootstrap-like iteration samples the same indices from x and y.
    This keeps forecast/observation pairs intact while reducing overlap effects.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if sample_size > len(x):
        raise ValueError(
            f"sample_size ({sample_size}) exceeds population size ({len(x)})."
        )

    rng = np.random.default_rng(seed)
    n_pop = len(x)
    boot_r = np.empty(n_bootstrap)
    boot_rmse = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.choice(n_pop, size=sample_size, replace=False)
        boot_r[i] = pearson_correlation(x[idx], y[idx])
        boot_rmse[i] = rmse(x[idx], y[idx])

    return boot_r, boot_rmse


def confidence_interval(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Return a percentile confidence interval."""
    lo = np.percentile(values, 100 * alpha / 2)
    hi = np.percentile(values, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# =============================================================================
# Data loading and calculation
# =============================================================================

def load_lead_arrays(path: Path, lead: int) -> tuple[np.ndarray, np.ndarray]:
    """Load prediction and observation arrays for one lead from one pickle file."""
    pred_output, real_output = load_prediction_arrays(path)
    if lead < 1 or lead > pred_output.shape[1]:
        raise ValueError(
            f"{path.name}: requested lead {lead}, but only "
            f"{pred_output.shape[1]} leads are available."
        )

    return pred_output[:, lead - 1], real_output[:, lead - 1]


def compute_file_result(path: Path, lead: int, sample_size: int, seed: int) -> dict:
    """Compute all metrics for one year, one lead, and one data source."""
    pred, real = load_lead_arrays(path, lead)
    observed_r = pearson_correlation(pred, real)
    boot_r, boot_rmse = bootstrap_like_correlation(
        pred,
        real,
        sample_size=sample_size,
        n_bootstrap=N_BOOTSTRAP,
        seed=seed,
    )
    ci_lo, ci_hi = confidence_interval(boot_r, ALPHA)

    return {
        "lead": lead,
        "year": parse_start_year(path),
        "sample_size": sample_size,
        "r_mean": float(boot_r.mean()),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "observed_r": observed_r,
        "rmse_mean": float(boot_rmse.mean()),
        "rmse_full": rmse(pred, real),
    }


def collect_all_results(data_sources: list[dict]) -> list[dict]:
    """Compute metrics for every configured data source, lead, and year."""
    all_results: list[dict] = []

    for source_index, source in enumerate(data_sources):
        dataset_id = source["id"]
        label = source["label"]
        sample_size = int(source["sample_size"])
        files_by_year = list_pickle_files_by_year(source["pickle_dir"])
        years = sorted(files_by_year)

        print("=" * 72)
        print(f"{label}: found {len(years)} pickle files")
        print(f"Years: {years[0]}-{years[-1]}")
        print(f"Sample size: {sample_size}")
        print("=" * 72)

        for year in years:
            path = files_by_year[year]
            for lead in LEADS:
                seed = RANDOM_SEED + source_index * 100000 + lead * 1000 + year
                result = compute_file_result(path, lead, sample_size, seed)
                result["dataset"] = dataset_id
                result["pickle_dir"] = str(source["pickle_dir"])
                all_results.append(result)

                print(
                    f"{label:22s} year={year} lead={lead:2d} "
                    f"r_mean={result['r_mean']:.4f} "
                    f"observed={result['observed_r']:.4f} "
                    f"95% CI=[{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]"
                )

    return all_results


def values_for(results: list[dict], dataset_id: str, lead: int, field: str) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted years and values for one dataset, one lead, and one field."""
    rows = [r for r in results if r["dataset"] == dataset_id and r["lead"] == lead]
    rows = sorted(rows, key=lambda r: r["year"])
    return np.array([r["year"] for r in rows]), np.array([r[field] for r in rows])


# =============================================================================
# Plotting
# =============================================================================


def plot_overview_lines(
    results: list[dict],
    data_sources: list[dict],
    field: str,
    output_path: Path,
) -> None:
    """Create Figure 1: grouped data-source comparisons at one lead."""
    years = np.array(sorted({result["year"] for result in results}))
    reference_source = next(source for source in data_sources if source["id"] == REFERENCE_DATASET_ID)
    comparison_sources = [source for source in data_sources if source["id"] != REFERENCE_DATASET_ID]
    source_groups = [
        comparison_sources[:5],       # (a) source_2–6   (with source_1 added below)
        comparison_sources[5:],       # (b) source_7–10  (with source_1 added below)
    ]

    values = np.array(
        [result[field] for result in results if result["lead"] == COMPARISON_LEAD],
        dtype=float,
    )
    reference_ci_values = np.array(
        [
            result[ci_field]
            for result in results
            if result["dataset"] == REFERENCE_DATASET_ID
            and result["lead"] == COMPARISON_LEAD
            for ci_field in ("ci_lo", "ci_hi")
        ],
        dtype=float,
    )
    y_values = np.concatenate([values, reference_ci_values])
    data_min = float(np.nanmin(y_values))
    data_max = float(np.nanmax(y_values))
    y_lower = min(FIGURE1_REFERENCE_R, data_min)
    y_upper = max(FIGURE1_REFERENCE_R, data_max)
    padding = max(0.1, (y_upper - y_lower) * 0.15)
    y_lower = max(-1.0, y_lower - padding)
    y_upper = min(1.05, y_upper + padding)

    fig_width = mm_to_inches(PUB_FIG_WIDTH_MM)
    fig_height = mm_to_inches(COMPARISON_FIG_HEIGHT_MM)
    fig, axes = plt.subplots(
        len(source_groups),
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.09,
        top=0.90,
        hspace=0.14,
    )
    axes = np.atleast_1d(axes)

    for panel_index, (ax, source_group) in enumerate(zip(axes, source_groups)):
        for source in [reference_source, *source_group]:
            dataset_id = source["id"]
            label = source["label"]
            lead_years, values = values_for(results, dataset_id, COMPARISON_LEAD, field)

            if dataset_id == "source_1":
                _, ci_lo = values_for(results, dataset_id, COMPARISON_LEAD, "ci_lo")
                _, ci_hi = values_for(results, dataset_id, COMPARISON_LEAD, "ci_hi")
                ax.fill_between(lead_years, ci_lo, ci_hi,
                    color=dataset_color(dataset_id), alpha=0.30,
                    linewidth=0, zorder=1)

            ax.plot(lead_years, values,
                color=dataset_color(dataset_id), linestyle="-",
                linewidth=PLOT_STYLE["comparison_line_width"], label=label)

        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(
            panel_title_only(panel_letter),
            loc="left", fontsize=PLOT_STYLE["panel_label_size"],
            fontweight="bold", pad=6,
        )
        ax.axhline(FIGURE1_REFERENCE_R, color="#5f5f5f",
            linewidth=PLOT_STYLE["reference_line_width"], linestyle="--", zorder=0,
        )
        ax.set_ylim(y_lower, y_upper)
        style_light_grid(ax, axis="y", linewidth=0.45)
        ax.tick_params(axis="both", direction="in", labelsize=PLOT_STYLE["tick_label_size"])
        style_open_axes(ax)

    tick_years = np.arange(years[0], years[-1] + 1, 10)
    axes[-1].set_xticks(tick_years)
    axes[-1].set_xlim(years[0], years[-1])
    add_shared_axis_labels(
        fig,
        xlabel="Year (start of period)",
        ylabel="Pearson r",
        xlabel_y=0.025,
        ylabel_x=0.022,
        fontsize=PLOT_STYLE["axis_label_size"],
    )

    legend_handles = []
    legend_labels = []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)
    
    ci_label = f"95% CI ({reference_source['label']})"
    legend_handles.append(
        Patch(facecolor="#999999", alpha=0.30, edgecolor="none", label=ci_label)
    )
    legend_labels.append(ci_label)

    add_compact_figure_legend(
        fig,
        handles=legend_handles,
        labels=legend_labels,
        ncol=4,
        bbox_to_anchor=(0.5, 0.992),
        fontsize=PLOT_STYLE["legend_size"],
        columnspacing=0.35,
        handlelength=1.00,
        labelspacing=0.18,
    )

    save_publication_figure(
        fig,
        figure_output_paths(output_path),
        dpi=FIGURE_DPI,
        pad_inches=0.02,
    )
    plt.close(fig)


def plot_small_multiples(
    results: list[dict],
    data_sources: list[dict],
    output_path: Path,
) -> None:
    """Create Figure 2: one time-series panel per data source."""
    source_ids = [source["id"] for source in data_sources]
    labels_by_id = {source["id"]: source["label"] for source in data_sources}
    years = np.array(sorted({result["year"] for result in results}))
    lead_colors = PLOT_STYLE["lead_colors"]
    min_ci = min(r["ci_lo"] for r in results)
    y_lower = max(-1.0, min(0.2, math.floor(min_ci * 10) / 10))
    y_upper = 1.05

    fig_width = mm_to_inches(PUB_FIG_WIDTH_MM)
    fig_height = max(mm_to_inches(PUB_FIG_HEIGHT_MM), 11.0)
    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = source_panel_grid_5x2(
        fig,
        left=0.08,
        right=0.99,
        bottom=0.11,
        top=0.94,
        wspace=0.12,
        hspace=0.28,
    )

    for panel_index, (ax, dataset_id) in enumerate(zip(axes, source_ids)):
        label = labels_by_id[dataset_id]
        for lead_index, lead in enumerate(LEADS):
            lead_color = lead_colors[lead_index % len(lead_colors)]
            lead_label = f"Leading {lead}M"
            lead_years, r_mean = values_for(results, dataset_id, lead, "r_mean")
            _, ci_lo = values_for(results, dataset_id, lead, "ci_lo")
            _, ci_hi = values_for(results, dataset_id, lead, "ci_hi")

            ax.fill_between(
                lead_years,
                ci_lo,
                ci_hi,
                color=lead_color,
                alpha=0.30,
                linewidth=0,
            )
            ax.plot(
                lead_years,
                r_mean,
                color=lead_color,
                linewidth=PLOT_STYLE["line_width"],
                label=lead_label,
            )

        ax.axhline(
            0.5,
            color="#5f5f5f",
            linewidth=PLOT_STYLE["reference_line_width"],
            linestyle="--",
            zorder=0,
        )
        ax.set_ylim(y_lower, y_upper)
        ax.set_xlim(years[0], years[-1])
        ax.set_title(
            panel_title(chr(ord("a") + panel_index), label),
            loc="left",
            fontsize=PLOT_STYLE["panel_label_size"],
            fontweight="bold",
            pad=4,
        )
        style_light_grid(ax, axis="y", linewidth=0.45)
        ax.tick_params(axis="both", direction="in", labelsize=PLOT_STYLE["small_tick_label_size"])
        style_open_axes(ax)

    tick_start = int(math.ceil(years[0] / 20) * 20)
    tick_years = np.arange(tick_start, years[-1] + 1, 20)
    for ax in axes:
        ax.set_xticks(tick_years)
    style_source_panel_axes_5x2(axes, n_visible=len(source_ids))
    add_shared_axis_labels(
        fig,
        xlabel="Year (start of period)",
        ylabel="Pearson r",
        xlabel_y=0.045,
        ylabel_x=0.014,
        fontsize=PLOT_STYLE["axis_label_size"],
    )

    legend_handles = []
    for lead_index, lead in enumerate(LEADS):
        legend_handles.append(
            Line2D([0], [0], color=lead_colors[lead_index % len(lead_colors)],
                   linewidth=PLOT_STYLE["line_width"], label=f"Leading {lead}M"))
    reference_source = next(source for source in data_sources if source["id"] == REFERENCE_DATASET_ID)
    ci_label = f"95% CI ({reference_source['label']})"
    legend_handles.append(
        Patch(facecolor="#999999", alpha=0.30, edgecolor="none", label=ci_label))

    if len(LEADS) > 1:
        add_compact_figure_legend(
            fig,
            handles=legend_handles,
            ncol=min(len(legend_handles), 3),
            bbox_to_anchor=(0.5, 1.005),
            columnspacing=0.45,
            handlelength=1.10,
            labelspacing=0.20,
            fontsize=PLOT_STYLE["legend_size"] - 0.5,
        )

    save_publication_figure(
        fig,
        figure_output_paths(output_path),
        dpi=FIGURE_DPI,
        pad_inches=0.02,
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configure_publication_style()
    validate_data_sources(DATA_SOURCES)

    if FIGURE1_VALUE not in VALUE_OPTIONS:
        raise ValueError('FIGURE1_VALUE must be "bootstrap_mean" or "observed".')
    figure1_field, figure1_token = VALUE_OPTIONS[FIGURE1_VALUE]

    if COMPARISON_LEAD not in LEADS:
        raise ValueError("COMPARISON_LEAD must be included in LEADS.")

    results = collect_all_results(DATA_SOURCES)
    lead_token = "-".join(str(lead) for lead in LEADS)

    figure1_path = OUTPUT_DIR / (
        f"{FIGURE_ID}_{FIGURE_NAME}_figure1_lines_{figure1_token}_lead{COMPARISON_LEAD}"
    )
    small_multiples_path = OUTPUT_DIR / (
        f"{FIGURE_ID}_{FIGURE_NAME}_figure2_small_multiples_lead{lead_token}"
    )

    plot_overview_lines(results, DATA_SOURCES, figure1_field, figure1_path)
    plot_small_multiples(results, DATA_SOURCES, small_multiples_path)

    print(f"Figure 1 → {figure1_path}.png  +  .pdf")
    print(f"Figure 2 → {small_multiples_path}.png  +  .pdf")


if __name__ == "__main__":
    main()
