"""Annual covariance contributions --- redesigned Figure I.

Layout (single integrated figure):

  (a) Top panel: NOAA and HadISST obs variance contribution background lines.
  (b) Middle panel: 10 source x verification year heatmap of annual
      cov(ensmean, obs) contribution.
  (c) Bottom panel: 10 source x verification year heatmap of annual
      cov(ensmean, obs) - var(obs) contribution.
      source_2 uses HadISST obs var; all others use NOAA obs var.

X-axes of (a), (b) and (c) are strictly aligned.

Run:  python I_1_plot_yearly_covariance_contribution.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import A_basic_sources
from A_basic_sources import (
    FIGURE_ROOT,
    get_dl_sources,
    list_pickle_files,
    load_prediction_arrays,
    parse_input_months,
    parse_start_year,
)
from plot_style import (
    DATASET_COLORS,
    DEFAULT_FIGURE_DPI,
    I_COVARIANCE_STYLE,
    configure_publication_style,
    figure_output_paths,
    mm_to_inches,
    save_publication_figure,
    style_open_axes,
    validate_data_sources,
)

# =============================================================================
# User configuration
# =============================================================================

BASE_YEAR: int = 1871
INPUT_WINDOW_MONTHS: int = 6
LEAD: int = 6
MIN_MONTHS: int = 3
OBSERVATION_TOLERANCE: float = 1e-10

FIGURE_ID: str = "I"
FIGURE_NAME: str = "yearly_covariance_contribution"
OUTPUT_DIR: Path = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
OUTPUT_BASENAME: str = f"I_1_{FIGURE_NAME}_lead6"
FIGURE_DPI: int = DEFAULT_FIGURE_DPI

DATA_SOURCES: list[dict] = get_dl_sources()

import os as _os
_VM_PICKLE_BASE: str = "/sessions/funny-inspiring-ramanujan/mnt/File"
if _os.path.isdir(_VM_PICKLE_BASE):
    A_basic_sources.PICKLE_BASE_DIR = Path(_VM_PICKLE_BASE)
    DATA_SOURCES = get_dl_sources()
    print(f"Override pickle base: {_VM_PICKLE_BASE}")
else:
    print(f"Using pickle base: {A_basic_sources.PICKLE_BASE_DIR}")

N_SOURCES: int = len(DATA_SOURCES)

# ---------------------------------------------------------------------------
# Layout geometry (figure fractions)
# ---------------------------------------------------------------------------
LEFT_MARGIN: float = 0.235
RIGHT_MARGIN: float = 0.875
BOTTOM_MARGIN: float = 0.090
TOP_MARGIN: float = 0.955
HSPACE_GS: float = 0.10

# Three panels: obs lines + cov heatmap + cov-minus-obs heatmap
HEIGHT_RATIOS: tuple[float, ...] = (0.9, 2.5, 2.5)

HEATMAP_CMAP_B = plt.cm.PuOr
HEATMAP_CMAP_C = plt.cm.RdBu_r
MISSING_GRAY_ALPHA: float = 0.35


def load_monthly_ensemble_mean(source: dict) -> pd.DataFrame:
    pickle_dir = Path(source["pickle_dir"])
    pickle_files = list_pickle_files(pickle_dir)

    tables: list[pd.DataFrame] = []
    for pickle_path in pickle_files:
        start_year = parse_start_year(pickle_path)
        input_months = parse_input_months(pickle_path, default=INPUT_WINDOW_MONTHS)
        prediction, observation = load_prediction_arrays(pickle_path)
        if LEAD < 1 or LEAD > prediction.shape[1]:
            raise ValueError(f"{pickle_path.name}: lead {LEAD} is unavailable")

        target_abs_month = (
            (start_year - BASE_YEAR) * 12
            + np.arange(prediction.shape[0])
            + input_months
            + LEAD
            - 1
        )
        tables.append(
            pd.DataFrame({
                "target_abs_month": target_abs_month,
                "prediction": prediction[:, LEAD - 1],
                "observation": observation[:, LEAD - 1],
            })
        )

    values = pd.concat(tables, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    values = values.dropna(subset=["prediction", "observation"])
    monthly = (
        values.groupby("target_abs_month", as_index=False)
        .agg(
            ensmean_prediction=("prediction", "mean"),
            observation=("observation", "mean"),
            observation_min=("observation", "min"),
            observation_max=("observation", "max"),
            n_pickle_forecasts=("prediction", "size"),
        )
        .sort_values("target_abs_month")
        .reset_index(drop=True)
    )
    observation_range = monthly["observation_max"] - monthly["observation_min"]
    if (observation_range > OBSERVATION_TOLERANCE).any():
        bad_months = monthly.loc[
            observation_range > OBSERVATION_TOLERANCE, "target_abs_month"
        ].tolist()
        raise ValueError(
            f"{source['label']}: inconsistent duplicate observations at {bad_months[:5]}"
        )

    monthly["year"] = BASE_YEAR + monthly["target_abs_month"] // 12
    monthly.attrs["n_pickle_files"] = len(pickle_files)
    return monthly


def annual_contributions(monthly: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    if len(monthly) < MIN_MONTHS:
        raise ValueError(f"Only {len(monthly)} valid target months are available")

    n_months = len(monthly)
    obs_anom = monthly["observation"] - monthly["observation"].mean()
    pred_anom = monthly["ensmean_prediction"] - monthly["ensmean_prediction"].mean()
    monthly = monthly.assign(
        observation_variance_contribution=obs_anom ** 2 / n_months,
        prediction_observation_covariance_contribution=pred_anom * obs_anom / n_months,
    )
    annual = (
        monthly.groupby("year", as_index=False)
        .agg(
            observation_variance_contribution=("observation_variance_contribution", "sum"),
            prediction_observation_covariance_contribution=("prediction_observation_covariance_contribution", "sum"),
            n_target_months=("target_abs_month", "size"),
            mean_available_pickles=("n_pickle_forecasts", "mean"),
        )
        .sort_values("year")
    )
    observation_variance = float(np.mean(obs_anom ** 2))
    prediction_observation_covariance = float(np.mean(pred_anom * obs_anom))
    return annual, observation_variance, prediction_observation_covariance


def _make_heatmap(ax, data_matrix, x_edges, y_edges, c_lim, cmap):
    im = ax.pcolormesh(
        x_edges, y_edges, data_matrix,
        cmap=cmap, vmin=-c_lim, vmax=c_lim,
        shading="flat", edgecolors="none", linewidth=0,
    )
    mask_nan = np.isnan(data_matrix)
    if mask_nan.any():
        gray_overlay = np.zeros_like(data_matrix)
        gray_overlay[mask_nan] = 0.5
        gray_overlay[~mask_nan] = np.nan
        ax.pcolormesh(
            x_edges, y_edges, gray_overlay,
            cmap=plt.cm.gray, vmin=0, vmax=1,
            alpha=MISSING_GRAY_ALPHA,
            shading="flat", edgecolors="none", linewidth=0,
        )
    return im


def _wrap_source_label(label: str, max_length: int = 25) -> str:
    if len(label) <= max_length or "+" not in label:
        return label
    parts = label.split("+")
    wrapped = parts[0]
    for part in parts[1:]:
        current_line = wrapped.split("\n")[-1]
        separator = "+\n" if len(current_line) + len(part) + 1 > max_length else "+"
        wrapped += separator + part
    return wrapped


def _style_heatmap_axes(ax, source_labels, year_ticks, year_min, year_max, xlabel=True):
    y_tick_positions = np.arange(N_SOURCES - 0.5, -0.5, -1, dtype=float)
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(source_labels, fontsize=I_COVARIANCE_STYLE["tick_label_size"])

    ax.set_xticks(year_ticks)
    if xlabel:
        ax.set_xticklabels(
            [str(int(t)) for t in year_ticks],
            fontsize=I_COVARIANCE_STYLE["tick_label_size"],
        )
    else:
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_xticklabels([])

    ax.set_xlim(year_min, year_max + 1)
    ax.set_ylim(N_SOURCES, 0)
    ax.tick_params(
        axis="both", direction="out", length=2.2, width=0.55, pad=1.8,
        labelsize=I_COVARIANCE_STYLE["tick_label_size"],
    )
    style_open_axes(ax)
    ax.grid(False)


def main() -> None:
    configure_publication_style()
    validate_data_sources(DATA_SOURCES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Compute all results
    results: list[dict] = []
    for source in DATA_SOURCES:
        monthly_values = load_monthly_ensemble_mean(source)
        annual_values, obs_var, pred_obs_cov = annual_contributions(monthly_values)
        results.append({
            **source,
            "annual": annual_values,
            "n_pickle_files": monthly_values.attrs["n_pickle_files"],
            "observation_variance": obs_var,
            "prediction_observation_covariance": pred_obs_cov,
        })
        print(
            f"{source['label']}: {monthly_values.attrs['n_pickle_files']} pickle files, "
            f"{len(monthly_values)} target months, "
            f"{monthly_values['year'].min()}-{monthly_values['year'].max()}, "
            f"var(obs)={obs_var:.4f}, cov(ensmean, obs)={pred_obs_cov:.4f}"
        )

    # 2. Determine global year range
    all_years = np.concatenate([r["annual"]["year"].to_numpy() for r in results])
    year_min: int = int(all_years.min())
    year_max: int = int(all_years.max())
    all_years_arr = np.arange(year_min, year_max + 1)
    n_years: int = len(all_years_arr)

    # 3. Extract obs variance annual series
    def _obs_annual_series(r: dict) -> pd.Series:
        return r["annual"].set_index("year")["observation_variance_contribution"]

    noaa_obs = _obs_annual_series(results[0])
    hadisst_obs = _obs_annual_series(results[1])

    # 4a. cov(ensmean, obs) contribution
    cov_matrix = np.full((N_SOURCES, n_years), np.nan, dtype=float)
    for i, result in enumerate(results):
        annual = result["annual"]
        year_col = annual["year"].to_numpy().astype(int)
        cov_col = annual["prediction_observation_covariance_contribution"].to_numpy()
        idx = year_col - year_min
        valid = (idx >= 0) & (idx < n_years)
        cov_matrix[i, idx[valid]] = cov_col[valid]

    # 4b. cov - var(obs)
    obs_var_matrix = np.full((N_SOURCES, n_years), np.nan, dtype=float)
    for year_val in all_years_arr:
        col = year_val - year_min
        noaa_val = noaa_obs.get(year_val, np.nan)
        hadisst_val = hadisst_obs.get(year_val, np.nan)
        for i in range(N_SOURCES):
            obs_var_matrix[i, col] = hadisst_val if i == 1 else noaa_val

    cov_minus_obs_matrix = cov_matrix - obs_var_matrix

    # 5. Color limits
    c1_max = float(np.nanmax(np.abs(cov_matrix)))
    if c1_max < 1e-10:
        c1_max = 1e-3
    c1_lim = c1_max

    c2_max = float(np.nanmax(np.abs(cov_minus_obs_matrix)))
    if c2_max < 1e-10:
        c2_max = 1e-3
    c2_lim = c2_max

    print(f"\nCov color limits: vmin={-c1_lim:.5f}, vmax={c1_lim:.5f}")
    print(f"Cov-Obs color limits: vmin={-c2_lim:.5f}, vmax={c2_lim:.5f}")

    # 6. Create figure
    fig = plt.figure(figsize=(
        mm_to_inches(I_COVARIANCE_STYLE["heatmap_figure_width_mm"]),
        mm_to_inches(I_COVARIANCE_STYLE["heatmap_figure_height_mm"]),
    ))

    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=HEIGHT_RATIOS, hspace=HSPACE_GS,
        left=LEFT_MARGIN, right=RIGHT_MARGIN,
        bottom=BOTTOM_MARGIN, top=TOP_MARGIN,
    )

    source_labels = [_wrap_source_label(r["label"]) for r in results]
    x_edges = np.arange(year_min, year_max + 2)
    y_edges = np.arange(N_SOURCES, -1, -1)

    year_step = 20
    year_ticks = np.arange(
        ((year_min + year_step - 1) // year_step) * year_step,
        year_max + 1, year_step,
    )

    # ---- (a) Obs var lines ----
    ax_obs = fig.add_subplot(gs[0])
    ax_obs.grid(False)
    style_open_axes(ax_obs)
    ax_obs.plot(
        noaa_obs.index, noaa_obs.values,
        color=DATASET_COLORS["source_1"], linewidth=I_COVARIANCE_STYLE["line_width"],
        label="NOAA SST",
    )
    ax_obs.plot(
        hadisst_obs.index, hadisst_obs.values,
        color=DATASET_COLORS["source_2"], linewidth=I_COVARIANCE_STYLE["line_width"],
        linestyle="--", label="HadISST",
    )
    ax_obs.axhline(0, color="0.65", linewidth=I_COVARIANCE_STYLE["reference_line_width"], zorder=0, linestyle=":")
    ax_obs.set_xlim(year_min, year_max + 1)
    ax_obs.set_ylabel("Annual obs var.\ncontribution", fontsize=I_COVARIANCE_STYLE["axis_label_size"])
    ax_obs.tick_params(axis="x", labelbottom=False)
    ax_obs.tick_params(axis="y", labelsize=I_COVARIANCE_STYLE["tick_label_size"])
    ax_obs.set_xticks(year_ticks)
    ax_obs.legend(
        loc="upper left", frameon=False, fontsize=I_COVARIANCE_STYLE["legend_size"],
        handlelength=1.2, borderpad=0.1, borderaxespad=0.2,
    )

    # ---- (b) Cov heatmap ----
    ax_cov = fig.add_subplot(gs[1])
    im_cov = _make_heatmap(ax_cov, cov_matrix, x_edges, y_edges, c1_lim, HEATMAP_CMAP_B)
    _style_heatmap_axes(ax_cov, source_labels, year_ticks, year_min, year_max, xlabel=False)

    # ---- (c) Cov - Obs heatmap ----
    ax_diff = fig.add_subplot(gs[2])
    im_diff = _make_heatmap(ax_diff, cov_minus_obs_matrix, x_edges, y_edges, c2_lim, HEATMAP_CMAP_C)
    _style_heatmap_axes(ax_diff, source_labels, year_ticks, year_min, year_max, xlabel=True)
    ax_diff.set_xlabel("Verification year", fontsize=I_COVARIANCE_STYLE["axis_label_size"])

    # ---- Colorbars aligned with panels ----
    cbar_width = I_COVARIANCE_STYLE["colorbar_width"]
    pos_b = ax_cov.get_position()
    pos_c = ax_diff.get_position()

    cbar1_ax = fig.add_axes([RIGHT_MARGIN + 0.015, pos_b.y0, cbar_width, pos_b.height])
    cbar1 = fig.colorbar(im_cov, cax=cbar1_ax)
    cbar1.set_label("cov(ensmean, obs)", fontsize=I_COVARIANCE_STYLE["colorbar_label_size"], labelpad=3)
    cbar1.ax.tick_params(labelsize=I_COVARIANCE_STYLE["colorbar_tick_size"], length=2, width=0.55, pad=1.5)
    cbar1.ax.axhline(0, color="0.30", linewidth=0.4, linestyle=":")

    cbar2_ax = fig.add_axes([RIGHT_MARGIN + 0.015, pos_c.y0, cbar_width, pos_c.height])
    cbar2 = fig.colorbar(im_diff, cax=cbar2_ax)
    cbar2.set_label("cov " + chr(8722) + " var(obs)", fontsize=I_COVARIANCE_STYLE["colorbar_label_size"], labelpad=3)
    cbar2.ax.tick_params(labelsize=I_COVARIANCE_STYLE["colorbar_tick_size"], length=2, width=0.55, pad=1.5)
    cbar2.ax.axhline(0, color="0.30", linewidth=0.4, linestyle=":")

    # ---- Subfigure labels (bold label + gray description, tight spacing) ----
    LBL_SIZE = I_COVARIANCE_STYLE["panel_label_size"]
    LBL_WT = "bold"
    LBL_GREY = "0.25"
    pos_a = ax_obs.get_position()
    pos_b = ax_cov.get_position()
    pos_c = ax_diff.get_position()
    X0 = pos_a.x0

    def _add_label(y, letter, desc, color=LBL_GREY):
        """Place bold label then gray description with a tiny figure-coord gap."""
        # step: "letter desc" rendered with a single space so they sit close
        fig.text(X0, y, f"{letter}  {desc}",
                 fontsize=LBL_SIZE, fontweight=LBL_WT, ha="left", va="bottom")

    _add_label(pos_a.y1 + 0.005, "(a)", "Obs variance contribution")
    _add_label(pos_b.y1 + 0.005, "(b)", "cov(ensmean, obs)")
    _add_label(pos_c.y1 + 0.005, "(c)", "cov(ensmean, obs) " + chr(8722) + " var(obs)")

    # 7. Save
    output_base = OUTPUT_DIR / OUTPUT_BASENAME
    saved_paths = save_publication_figure(
        fig,
        figure_output_paths(output_base),
        dpi=FIGURE_DPI, bbox_inches=None, pad_inches=0.02,
    )
    for p in saved_paths:
        print(f"Saved figure: {p}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
