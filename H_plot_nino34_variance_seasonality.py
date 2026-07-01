"""Niño 3.4 variance seasonality by target month for every input data source.

Each panel compares observed Niño 3.4 variance with twelve forecast
trajectories starting in each calendar month at lead 1 through lead 12.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from A_basic_sources import (
    FIGURE_ROOT,
    get_dl_sources,
    list_pickle_files,
    load_prediction_arrays,
    parse_input_months,
    parse_start_year,
)
from plot_style import (
    AXIS_LABEL_SIZE,
    COLORBAR_TICK_SIZE,
    COMPACT_TICK_LABEL_SIZE,
    LEGEND_SIZE,
    TITLE_SIZE,
    add_shared_axis_labels,
    configure_publication_style,
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
LEADS = list(range(1, 13))
MIN_SAMPLES = 3

SPREAD_METRIC = "standard_deviation"

FIGURE_ID = "H"
FIGURE_NAME = "nino34_variance_seasonality"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
OUTPUT_BASENAME = f"{FIGURE_ID}_{FIGURE_NAME}_first_forecast_month_lead1-12"
FIGURE_DPI = 600
OUTPUT_FORMATS = ("png", "pdf")

FIGURE_WIDTH_INCH = 7.2
FIGURE_HEIGHT_INCH = 10.4
OBSERVATION_COLOR = "#1A1A1A"
FORECAST_LINE_WIDTH = 1.7
OBSERVATION_LINE_WIDTH = 1.8
FORECAST_MARKER_SIZE = 3.6

SEASON_COLORS = {
    "DJF": "#0072B2",
    "MAM": "#009E73",
    "JJA": "#E69F00",
    "SON": "#CC79A7",
}
MONTH_TO_SEASON = {
    1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA",
    7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON", 12: "DJF",
}
MONTH_POSITION_MARKERS = {1: "o", 2: "s", 3: "^"}

DATA_SOURCES = get_dl_sources()

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
EXTENDED_MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D",
                       "J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N"]


# =============================================================================
# Data preparation
# =============================================================================

def load_source_predictions(source: dict) -> pd.DataFrame:
    """Load one source into target-month, lead, prediction, and observation rows."""
    pickle_dir = Path(source["pickle_dir"])
    pickle_files = list_pickle_files(pickle_dir)

    tables = []
    for pickle_path in pickle_files:
        start_year = parse_start_year(pickle_path)
        input_months = parse_input_months(pickle_path, default=INPUT_WINDOW_MONTHS)
        if input_months != INPUT_WINDOW_MONTHS:
            raise ValueError(
                f"{pickle_path.name}: input window is {input_months} months, "
                f"but INPUT_WINDOW_MONTHS is {INPUT_WINDOW_MONTHS}."
            )

        prediction, observation = load_prediction_arrays(pickle_path)
        if prediction.shape[1] < max(LEADS):
            raise ValueError(
                f"{pickle_path.name}: only {prediction.shape[1]} leads are available; "
                f"lead {max(LEADS)} is required."
            )

        selected_prediction = prediction[:, np.array(LEADS) - 1]
        selected_observation = observation[:, np.array(LEADS) - 1]
        sample_index = np.arange(prediction.shape[0])[:, None]
        lead_values = np.asarray(LEADS)[None, :]
        target_abs_month = (
            (start_year - BASE_YEAR) * 12 + sample_index + input_months + lead_values - 1
        )

        tables.append(
            pd.DataFrame(
                {
                    "target_abs_month": target_abs_month.ravel(),
                    "lead": np.broadcast_to(lead_values, selected_prediction.shape).ravel(),
                    "prediction": selected_prediction.ravel(),
                    "observation": selected_observation.ravel(),
                }
            )
        )

    values = pd.concat(tables, ignore_index=True)
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.dropna(subset=["prediction", "observation"])
    values["target_month"] = values["target_abs_month"] % 12 + 1
    values["first_forecast_abs_month"] = values["target_abs_month"] - values["lead"] + 1
    values["first_forecast_month"] = values["first_forecast_abs_month"] % 12 + 1
    return values


def average_duplicate_forecasts(values: pd.DataFrame) -> pd.DataFrame:
    """Ensemble-average duplicate forecasts for each target month and lead."""
    return (
        values.groupby(["target_abs_month", "lead"], as_index=False)
        .agg(
            prediction=("prediction", "mean"),
            observation=("observation", "mean"),
            target_month=("target_month", "first"),
            first_forecast_abs_month=("first_forecast_abs_month", "first"),
            first_forecast_month=("first_forecast_month", "first"),
        )
        .sort_values(["target_abs_month", "lead"])
        .reset_index(drop=True)
    )


def common_target_months(source_values: dict[str, pd.DataFrame]) -> set[int]:
    """Return target months available in every configured source."""
    month_sets = [set(table["target_abs_month"].unique()) for table in source_values.values()]
    common_months = set.intersection(*month_sets)
    if not common_months:
        raise ValueError("No common target months are available across all data sources.")
    return common_months


def sample_spread(values: pd.Series) -> float:
    """Return the configured sample spread when enough values are available."""
    finite_values = values.dropna()
    if len(finite_values) < MIN_SAMPLES:
        return np.nan
    variance = float(finite_values.var(ddof=1))
    if SPREAD_METRIC == "variance":
        return variance
    if SPREAD_METRIC == "standard_deviation":
        return float(np.sqrt(variance))
    raise ValueError('SPREAD_METRIC must be "standard_deviation" or "variance".')


def spread_axis_label() -> str:
    """Return the y-axis label that matches the configured spread metric."""
    if SPREAD_METRIC == "variance":
        return r"Niño 3.4 variance (°C$^2$)"
    if SPREAD_METRIC == "standard_deviation":
        return "Niño 3.4 standard deviation (°C)"
    raise ValueError('SPREAD_METRIC must be "standard_deviation" or "variance".')


def calculate_monthly_variances(values: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Observed variance by calendar month and forecast variance by start month/lead."""
    observation_once = (
        values.groupby("target_abs_month", as_index=False)
        .agg(observation=("observation", "mean"), target_month=("target_month", "first"))
    )
    observation_variance = observation_once.groupby("target_month")["observation"].apply(sample_spread)

    forecast_variance = (
        values.groupby(["first_forecast_month", "lead"])["prediction"]
        .apply(sample_spread)
        .unstack("lead")
    )
    sample_counts = values.groupby(["first_forecast_month", "lead"]).size().unstack("lead")

    return (
        observation_variance.reindex(range(1, 13)),
        forecast_variance.reindex(index=range(1, 13), columns=LEADS),
        sample_counts.reindex(index=range(1, 13), columns=LEADS),
    )


# =============================================================================
# Plotting
# =============================================================================

def draw_source_panel(
    axis: plt.Axes,
    source: dict,
    observation_variance: pd.Series,
    forecast_variance: pd.DataFrame,
    start_month_styles: dict[int, dict[str, str]],
    panel_letter: str,
) -> None:
    """Draw observed variance plus one 12-lead path per first forecast month."""
    extended_month_positions = np.arange(1, 24)
    plot_order = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 12]
    for first_forecast_month in plot_order:
        trajectory_positions = first_forecast_month + np.arange(len(LEADS))
        axis.plot(
            trajectory_positions,
            forecast_variance.loc[first_forecast_month, LEADS],
            color=start_month_styles[first_forecast_month]["color"],
            linewidth=FORECAST_LINE_WIDTH,
            marker=start_month_styles[first_forecast_month]["marker"],
            markersize=FORECAST_MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=start_month_styles[first_forecast_month]["color"],
            markeredgewidth=0.9,
            alpha=0.95,
            zorder=3 if MONTH_TO_SEASON[first_forecast_month] == "DJF" else 2,
        )
    repeated_observation_variance = np.concatenate(
        [observation_variance.to_numpy(), observation_variance.iloc[:11].to_numpy()]
    )
    axis.plot(
        extended_month_positions,
        repeated_observation_variance,
        color=OBSERVATION_COLOR,
        linewidth=OBSERVATION_LINE_WIDTH,
        zorder=3,
    )

    axis.set_title(
        f"({panel_letter}) {source['label']}",
        loc="left",
        color="black",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=5,
    )
    axis.set_xlim(1, 23)
    axis.set_xticks([1, 4, 7, 10, 13, 16, 19, 22])
    axis.set_xticklabels(["Jul", "Oct", "Jan", "Apr", "Jul", "Oct", "Jan", "Apr"],
                          fontsize=COMPACT_TICK_LABEL_SIZE)
    axis.tick_params(axis="x", labelrotation=0, pad=1.5)
    style_open_axes(axis)


def add_compact_top_legend(figure: plt.Figure) -> None:
    """Add the H figure legend above the 5-by-2 panels."""
    handles = [
        Line2D([0], [0], color=OBSERVATION_COLOR, linewidth=OBSERVATION_LINE_WIDTH, label="Observed"),
        *[
            Line2D([0], [0], color=colour, linewidth=FORECAST_LINE_WIDTH, label=season)
            for season, colour in SEASON_COLORS.items()
        ],
        *[
            Line2D(
                [0],
                [0],
                color="0.30",
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=0.9,
                linewidth=0,
                markersize=FORECAST_MARKER_SIZE,
                label=label,
            )
            for marker, label in [
                ("o", "Jan/Apr/Jul/Oct"),
                ("s", "Feb/May/Aug/Nov"),
                ("^", "Mar/Jun/Sep/Dec"),
            ]
        ],
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        frameon=False,
        fontsize=6.0,
        ncol=3,
        handlelength=1.5,
        columnspacing=0.45,
        handletextpad=0.4,
        labelspacing=0.3,
    )


def plot_variance_seasonality(results: list[dict]) -> list[Path]:
    """Create and save the 5-by-2 start-month variance figure."""
    configure_publication_style()
    figure = plt.figure(figsize=(FIGURE_WIDTH_INCH, FIGURE_HEIGHT_INCH))
    axes = source_panel_grid_5x2(
        figure,
        left=0.17,
        right=0.98,
        bottom=0.075,
        top=0.91,
        wspace=0.12,
        hspace=0.22,
    )

    start_month_styles = {
        first_forecast_month: {
            "color": SEASON_COLORS[MONTH_TO_SEASON[first_forecast_month]],
            "marker": MONTH_POSITION_MARKERS[((first_forecast_month - 1) % 3) + 1],
        }
        for first_forecast_month in range(1, 13)
    }

    for panel_index, (axis, result) in enumerate(zip(axes, results)):
        draw_source_panel(
            axis,
            result["source"],
            result["observation_variance"],
            result["forecast_variance"],
            start_month_styles,
            chr(ord("a") + panel_index),
        )

    add_compact_top_legend(figure)

    style_source_panel_axes_5x2(axes, n_visible=len(results))
    add_shared_axis_labels(
        figure,
        xlabel="Forecast verification month",
        ylabel=spread_axis_label(),
        xlabel_y=0.035,
        ylabel_x=0.04,
        fontsize=AXIS_LABEL_SIZE,
    )

    all_variances = [
        result["observation_variance"].to_numpy(dtype=float)
        for result in results
    ] + [
        result["forecast_variance"].to_numpy(dtype=float).ravel()
        for result in results
    ]
    maximum_variance = float(np.nanmax(np.concatenate(all_variances)))
    y_upper = max(0.1, np.ceil(maximum_variance * 5) / 5)
    for axis in axes:
        axis.set_ylim(0, y_upper)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_base = OUTPUT_DIR / OUTPUT_BASENAME
    saved_paths = save_publication_figure(
        figure,
        [output_base.with_suffix(f".{output_format}") for output_format in OUTPUT_FORMATS],
        dpi=FIGURE_DPI,
        pad_inches=0.03,
    )
    plt.close(figure)
    return saved_paths


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    """Load all sources, calculate common-period variances, and save the figure."""
    validate_data_sources(DATA_SOURCES)
    if SPREAD_METRIC not in {"standard_deviation", "variance"}:
        raise ValueError('SPREAD_METRIC must be "standard_deviation" or "variance".')
    if not LEADS or min(LEADS) < 1 or LEADS != list(range(min(LEADS), max(LEADS) + 1)):
        raise ValueError("LEADS must be one consecutive, positive sequence of forecast months.")

    source_values: dict[str, pd.DataFrame] = {}
    for source in DATA_SOURCES:
        values = average_duplicate_forecasts(load_source_predictions(source))
        source_values[source["id"]] = values
        print(f"{source['label']}: {len(values)} target-month/lead forecasts after duplicate averaging")

    shared_months = common_target_months(source_values)
    shared_start = BASE_YEAR + min(shared_months) // 12
    shared_end = BASE_YEAR + max(shared_months) // 12
    print(f"Common target-month period: {shared_start}-{shared_end} ({len(shared_months)} months)")

    results = []
    for source in DATA_SOURCES:
        values = source_values[source["id"]]
        values = values[values["target_abs_month"].isin(shared_months)].copy()
        observation_variance, forecast_variance, sample_counts = calculate_monthly_variances(values)
        minimum_count = int(sample_counts.min().min())
        print(f"{source['label']}: minimum samples per target-month/lead group = {minimum_count}")
        results.append(
            {
                "source": source,
                "observation_variance": observation_variance,
                "forecast_variance": forecast_variance,
            }
        )

    saved_paths = plot_variance_seasonality(results)
    for saved_path in saved_paths:
        print(f"Saved figure: {saved_path}")


if __name__ == "__main__":
    main()
