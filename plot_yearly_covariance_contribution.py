"""Plot annual covariance contributions for five Niño3.4 forecast data sources.

For each source, duplicate forecasts for the same target month are averaged to
form one monthly ensemble-mean prediction. The black line is the annual
contribution to var(observation); the purple line is the annual contribution
to cov(ensemble-mean prediction, observation). Neither quantity is divided by
standard deviations.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_style import (
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    NMME_COLORS,
    PANEL_LABEL_SIZE,
    configure_publication_style,
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

OUTPUT_DIR = Path("yearly_covariance_contribution_figures")
OUTPUT_BASENAME = "yearly_covariance_contribution_lead6"
OUTPUT_FORMATS = ("png", "pdf")
FIGURE_DPI = 600
FIGURE_WIDTH_INCH = 7.2
FIGURE_HEIGHT_INCH = 12.0

OBSERVATION_COLOR = "#000000"
PREDICTION_COVARIANCE_COLOR = NMME_COLORS[3]

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


def parse_pickle_metadata(pickle_path: Path) -> tuple[int, int]:
    """Return the test-start year and input-window length encoded in a filename."""
    year_match = re.search(r"_(\d{4})_", pickle_path.name)
    if year_match is None:
        raise ValueError(f"Could not parse a test-start year from {pickle_path.name}")
    input_match = re.search(r"input(\d+)", pickle_path.name)
    input_months = int(input_match.group(1)) if input_match else INPUT_WINDOW_MONTHS
    return int(year_match.group(1)), input_months


def load_monthly_ensemble_mean(source: dict) -> pd.DataFrame:
    """Load one source and average duplicate forecasts for each target month."""
    pickle_dir = Path(source["pickle_dir"])
    pickle_files = sorted(pickle_dir.glob("*.pickle"))
    if not pickle_files:
        raise FileNotFoundError(f"No pickle files found in: {pickle_dir}")

    tables = []
    for pickle_path in pickle_files:
        start_year, input_months = parse_pickle_metadata(pickle_path)
        with pickle_path.open("rb") as handle:
            dataset = pickle.load(handle)

        prediction = np.asarray(dataset["predict_value"], dtype=float)
        observation = np.asarray(dataset["real_value"], dtype=float)
        if prediction.shape != observation.shape or prediction.ndim != 2:
            raise ValueError(f"Unexpected prediction/observation shape in {pickle_path.name}")
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
    if not np.isclose(annual["observation_variance_contribution"].sum(), observation_variance):
        raise AssertionError("Annual observation contributions do not sum to var(observation)")
    if not np.isclose(
        annual["prediction_observation_covariance_contribution"].sum(),
        prediction_observation_covariance,
    ):
        raise AssertionError("Annual prediction-observation contributions do not sum to covariance")
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

figure, axes = plt.subplots(
    5,
    1,
    figsize=(FIGURE_WIDTH_INCH, FIGURE_HEIGHT_INCH),
    sharex=True,
    sharey=True,
)
figure.subplots_adjust(left=0.14, right=0.98, bottom=0.09, top=0.97, hspace=0.34)

for panel_index, (axis, result) in enumerate(zip(axes, results)):
    annual = result["annual"]
    axis.plot(
        annual["year"],
        annual["observation_variance_contribution"],
        color=OBSERVATION_COLOR,
        linewidth=1.35,
        label="Observation variance",
    )
    axis.plot(
        annual["year"],
        annual["prediction_observation_covariance_contribution"],
        color=PREDICTION_COVARIANCE_COLOR,
        linewidth=1.35,
        label="cov(ensmean, observation)",
    )
    axis.axhline(0, color="0.70", linewidth=0.65, zorder=0)
    axis.set_title(
        f"({chr(ord('a') + panel_index)}) {result['label']}",
        loc="left",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        pad=4,
    )
    axis.text(
        0.99,
        0.96,
        f"var(obs) = {result['observation_variance']:.3f}\n"
        f"cov(ensmean, obs) = {result['prediction_observation_covariance']:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=LEGEND_SIZE,
    )
    axis.set_xlim(year_min, year_max)
    axis.set_ylim(*y_limits)
    axis.set_ylabel("Annual covariance contribution", fontsize=AXIS_LABEL_SIZE)
    if panel_index == len(results) - 1:
        axis.set_xlabel("Verification year", fontsize=AXIS_LABEL_SIZE)
    style_open_axes(axis)

figure.legend(
    handles=[
        Line2D([0], [0], color=OBSERVATION_COLOR, linewidth=1.5, label="Observation variance"),
        Line2D(
            [0],
            [0],
            color=PREDICTION_COVARIANCE_COLOR,
            linewidth=1.5,
            label="cov(ensmean, observation)",
        ),
    ],
    loc="lower center",
    frameon=False,
    fontsize=LEGEND_SIZE,
    ncol=2,
    bbox_to_anchor=(0.5, 0.005),
)
figure.text(
    0.5,
    0.04,
    f"Lead {LEAD} months; duplicate target-month forecasts are ensemble averaged",
    ha="center",
    va="center",
    fontsize=LEGEND_SIZE,
)
for axis in axes:
    axis.set_xticks(np.arange(((year_min + 19) // 20) * 20, year_max + 1, 20))

output_base = OUTPUT_DIR / OUTPUT_BASENAME
for output_format in OUTPUT_FORMATS:
    output_path = output_base.with_suffix(f".{output_format}")
    save_kwargs = {}
    if output_format == "png":
        save_kwargs["dpi"] = FIGURE_DPI
    figure.savefig(output_path, **save_kwargs)
    print(f"Saved figure: {output_path}")
plt.close(figure)
