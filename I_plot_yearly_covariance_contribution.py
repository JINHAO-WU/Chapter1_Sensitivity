"""Annual covariance contributions for five Niño3.4 forecast data sources.

Duplicate forecasts for the same target month are averaged to form one
monthly ensemble-mean prediction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from A_basic_sources import (
    FIGURE_ROOT,
    get_dl_sources,
    list_pickle_files,
    load_prediction_arrays,
    parse_input_months,
    parse_start_year,
)
from plot_style import (
    DEFAULT_FIGURE_DPI,
    I_COVARIANCE_STYLE,
    NMME_COLORS,
    add_shared_axis_labels,
    configure_publication_style,
    figure_output_paths,
    mm_to_inches,
    panel_title,
    save_publication_figure,
    source_panel_grid_5x2,
    style_source_panel_axes_5x2,
    style_open_axes,
    validate_data_sources,
)


# =============================================================================
# User configuration
# =============================================================================

BASE_YEAR = 1871
INPUT_WINDOW_MONTHS = 6
LEAD = 6
MIN_MONTHS = 3
OBSERVATION_TOLERANCE = 1e-10

FIGURE_ID = "I"
FIGURE_NAME = "yearly_covariance_contribution"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
OUTPUT_BASENAME = f"{FIGURE_ID}_{FIGURE_NAME}_lead6"
FIGURE_DPI = DEFAULT_FIGURE_DPI

OBSERVATION_COLOR = "#000000"
PREDICTION_COVARIANCE_COLOR = NMME_COLORS[3]

DATA_SOURCES = get_dl_sources()


def wrap_long_panel_label(label: str, max_length: int = 27) -> str:
    """Wrap long source labels at plus signs so titles stay inside panels."""
    if len(label) <= max_length or "+" not in label:
        return label
    parts = label.split("+")
    wrapped = parts[0]
    for part in parts[1:]:
        separator = "+\n" if len(wrapped.split("\n")[-1]) + len(part) + 1 > max_length else "+"
        wrapped += separator + part
    return wrapped


def load_monthly_ensemble_mean(source: dict) -> pd.DataFrame:
    """Load one source and average duplicate forecasts for each target month."""
    pickle_dir = Path(source["pickle_dir"])
    pickle_files = list_pickle_files(pickle_dir)

    tables = []
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
            pd.DataFrame(
                {
                    "target_abs_month": target_abs_month,
                    "prediction": prediction[:, LEAD - 1],
                    "observation": observation[:, LEAD - 1],
                }
            )
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
        bad_months = monthly.loc[observation_range > OBSERVATION_TOLERANCE, "target_abs_month"].tolist()
        raise ValueError(f"{source['label']}: inconsistent duplicate observations at {bad_months[:5]}")

    monthly["year"] = BASE_YEAR + monthly["target_abs_month"] // 12
    monthly.attrs["n_pickle_files"] = len(pickle_files)
    return monthly


def annual_contributions(monthly: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Return annual contributions to var(obs) and cov(ensmean prediction, obs)."""
    if len(monthly) < MIN_MONTHS:
        raise ValueError(f"Only {len(monthly)} valid target months are available")

    n_months = len(monthly)
    observation_anomaly = monthly["observation"] - monthly["observation"].mean()
    prediction_anomaly = monthly["ensmean_prediction"] - monthly["ensmean_prediction"].mean()
    monthly = monthly.assign(
        observation_variance_contribution=observation_anomaly**2 / n_months,
        prediction_observation_covariance_contribution=(
            prediction_anomaly * observation_anomaly / n_months
        ),
    )
    annual = (
        monthly.groupby("year", as_index=False)
        .agg(
            observation_variance_contribution=("observation_variance_contribution", "sum"),
            prediction_observation_covariance_contribution=(
                "prediction_observation_covariance_contribution",
                "sum",
            ),
            n_target_months=("target_abs_month", "size"),
            mean_available_pickles=("n_pickle_forecasts", "mean"),
        )
        .sort_values("year")
    )
    observation_variance = float(np.mean(observation_anomaly**2))
    prediction_observation_covariance = float(np.mean(prediction_anomaly * observation_anomaly))
    return annual, observation_variance, prediction_observation_covariance


# =============================================================================
# Main workflow
# =============================================================================

configure_publication_style()
validate_data_sources(DATA_SOURCES)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = []
for source in DATA_SOURCES:
    monthly_values = load_monthly_ensemble_mean(source)
    annual_values, observation_variance, prediction_observation_covariance = annual_contributions(monthly_values)
    results.append(
        {
            **source,
            "annual": annual_values,
            "n_pickle_files": monthly_values.attrs["n_pickle_files"],
            "observation_variance": observation_variance,
            "prediction_observation_covariance": prediction_observation_covariance,
        }
    )
    print(
        f"{source['label']}: {monthly_values.attrs['n_pickle_files']} pickle files, "
        f"{len(monthly_values)} target months, "
        f"{monthly_values['year'].min()}-{monthly_values['year'].max()}, "
        f"var(obs)={observation_variance:.4f}, "
        f"cov(ensmean, obs)={prediction_observation_covariance:.4f}"
    )

all_years = np.concatenate([result["annual"]["year"].to_numpy() for result in results])
year_min, year_max = int(all_years.min()), int(all_years.max())
all_contributions = np.concatenate(
    [
        result["annual"][
            [
                "observation_variance_contribution",
                "prediction_observation_covariance_contribution",
            ]
        ].to_numpy().ravel()
        for result in results
    ]
)
y_min, y_max = float(np.nanmin(all_contributions)), float(np.nanmax(all_contributions))
y_padding = max(0.001, 0.05 * (y_max - y_min))
y_limits = (y_min - y_padding, y_max + y_padding)

figure = plt.figure(figsize=(
    mm_to_inches(I_COVARIANCE_STYLE["line_figure_width_mm"]),
    mm_to_inches(I_COVARIANCE_STYLE["line_figure_height_mm"]),
))
axes = source_panel_grid_5x2(
    figure,
    left=0.12,
    right=0.98,
    bottom=0.110,
    top=0.975,
    wspace=0.12,
    hspace=0.20,
)

for panel_index, (axis, result) in enumerate(zip(axes, results)):
    annual = result["annual"]
    axis.plot(
        annual["year"],
        annual["observation_variance_contribution"],
        color=OBSERVATION_COLOR,
        linewidth=I_COVARIANCE_STYLE["line_width"],
        label="Observation variance",
    )
    axis.plot(
        annual["year"],
        annual["prediction_observation_covariance_contribution"],
        color=PREDICTION_COVARIANCE_COLOR,
        linewidth=I_COVARIANCE_STYLE["line_width"],
        label="cov(ensmean, observation)",
    )
    axis.axhline(0, color="0.70", linewidth=I_COVARIANCE_STYLE["reference_line_width"], zorder=0)
    axis.set_title(
        panel_title(chr(ord("a") + panel_index), wrap_long_panel_label(result["label"])),
        loc="left",
        fontsize=I_COVARIANCE_STYLE["panel_label_size"],
        fontweight="bold",
        pad=4,
    )
    axis.text(
        0.99,
        0.96,
        f"var = {result['observation_variance']:.3f}\n"
        f"cov = {result['prediction_observation_covariance']:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=I_COVARIANCE_STYLE["annotation_size"],
    )
    axis.set_xlim(year_min, year_max)
    axis.set_ylim(*y_limits)
    style_open_axes(axis)

style_source_panel_axes_5x2(axes, n_visible=len(results))
add_shared_axis_labels(
    figure,
    xlabel="Verification year",
    ylabel="Annual covariance contribution",
    xlabel_y=0.058,
    ylabel_x=0.025,
    fontsize=I_COVARIANCE_STYLE["axis_label_size"],
)

figure.legend(
    handles=[
        Line2D([0], [0], color=OBSERVATION_COLOR, linewidth=1.35, label="var(obs)"),
        Line2D([0], [0], color=PREDICTION_COVARIANCE_COLOR,
               linewidth=1.35, label="cov(ensmean, obs)"),
    ],
    loc="upper center",
    frameon=False,
    fontsize=I_COVARIANCE_STYLE["legend_size"],
    ncol=2,
    bbox_to_anchor=(0.5, 1.008),
)
figure.text(
    0.5,
    0.035,
    f"Lead {LEAD} months; ensemble-mean duplicate forecasts",
    ha="center",
    va="center",
    fontsize=I_COVARIANCE_STYLE["legend_size"],
)
for axis in axes:
    axis.set_xticks(np.arange(((year_min + 19) // 20) * 20, year_max + 1, 20))
    axis.tick_params(axis="both", labelsize=I_COVARIANCE_STYLE["tick_label_size"])

output_base = OUTPUT_DIR / OUTPUT_BASENAME
saved_paths = save_publication_figure(
    figure,
    figure_output_paths(output_base),
    dpi=FIGURE_DPI,
    bbox_inches=None,
    pad_inches=0.02,
)
for output_path in saved_paths:
    print(f"Saved figure: {output_path}")
plt.close(figure)
