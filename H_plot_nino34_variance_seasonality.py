"""Nino 3.4 variance seasonality by target month for every input data source.

Each panel compares observed Nino 3.4 variance with twelve forecast
trajectories starting in each calendar month at lead 1 through lead 12.
"""

from __future__ import annotations

from pathlib import Path

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
    DEFAULT_FIGURE_DPI,
    H_VARIANCE_SEASONALITY_STYLE,
    add_shared_axis_labels,
    configure_publication_style,
    figure_output_paths,
    mm_to_inches,
    save_publication_figure,
    source_panel_grid_5x2,
    style_source_panel_axes_5x2,
    style_open_axes,
    validate_data_sources,
)

BASE_YEAR = 1871
INPUT_WINDOW_MONTHS = 6
LEADS = list(range(1, 13))
MIN_SAMPLES = 3

SPREAD_METRIC = "standard_deviation"

FIGURE_ID = "H"
FIGURE_NAME = "nino34_variance_seasonality"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
OUTPUT_BASENAME = f"{FIGURE_ID}_{FIGURE_NAME}_first_forecast_month_lead1-12"
FIGURE_DPI = DEFAULT_FIGURE_DPI

OBSERVATION_COLOR = "#1A1A1A"
FORECAST_LINE_WIDTH = H_VARIANCE_SEASONALITY_STYLE["forecast_line_width"]
OBSERVATION_LINE_WIDTH = H_VARIANCE_SEASONALITY_STYLE["observation_line_width"]
FORECAST_MARKER_SIZE = H_VARIANCE_SEASONALITY_STYLE["forecast_marker_size"]
FORECAST_ALPHA = H_VARIANCE_SEASONALITY_STYLE["forecast_alpha"]
Y_AXIS_PADDING_FRACTION = 0.06

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

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def load_source_predictions(source):
    pickle_dir = Path(source["pickle_dir"])
    pickle_files = list_pickle_files(pickle_dir)
    tables = []
    for pickle_path in pickle_files:
        start_year = parse_start_year(pickle_path)
        input_months = parse_input_months(pickle_path, default=INPUT_WINDOW_MONTHS)
        if input_months != INPUT_WINDOW_MONTHS:
            raise ValueError(f"{pickle_path.name}: input window mismatch")
        prediction, observation = load_prediction_arrays(pickle_path)
        if prediction.shape[1] < max(LEADS):
            raise ValueError(f"{pickle_path.name}: insufficient leads")
        selected_prediction = prediction[:, np.array(LEADS) - 1]
        selected_observation = observation[:, np.array(LEADS) - 1]
        sample_index = np.arange(prediction.shape[0])[:, None]
        lead_values = np.asarray(LEADS)[None, :]
        target_abs_month = (start_year - BASE_YEAR) * 12 + sample_index + input_months + lead_values - 1
        tables.append(pd.DataFrame({
            "target_abs_month": target_abs_month.ravel(),
            "lead": np.broadcast_to(lead_values, selected_prediction.shape).ravel(),
            "prediction": selected_prediction.ravel(),
            "observation": selected_observation.ravel(),
        }))
    values = pd.concat(tables, ignore_index=True)
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.dropna(subset=["prediction", "observation"])
    values["target_month"] = values["target_abs_month"] % 12 + 1
    values["first_forecast_abs_month"] = values["target_abs_month"] - values["lead"] + 1
    values["first_forecast_month"] = values["first_forecast_abs_month"] % 12 + 1
    return values


def average_duplicate_forecasts(values):
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


def common_target_months(source_values):
    month_sets = [set(table["target_abs_month"].unique()) for table in source_values.values()]
    common_months = set.intersection(*month_sets)
    if not common_months:
        raise ValueError("No common target months across all data sources.")
    return common_months


def sample_spread(values):
    finite_values = values.dropna()
    if len(finite_values) < MIN_SAMPLES:
        return np.nan
    variance = float(finite_values.var(ddof=1))
    if SPREAD_METRIC == "variance":
        return variance
    if SPREAD_METRIC == "standard_deviation":
        return float(np.sqrt(variance))
    raise ValueError('SPREAD_METRIC must be "standard_deviation" or "variance".')


def spread_axis_label():
    if SPREAD_METRIC == "variance":
        return r"Nino 3.4 variance (degC^2)"
    if SPREAD_METRIC == "standard_deviation":
        return "Nino 3.4 standard deviation (degC)"
    raise ValueError('SPREAD_METRIC must be "standard_deviation" or "variance".')


def calculate_monthly_variances(values):
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


def draw_source_panel(axis, source, observation_variance, forecast_variance, start_month_styles, panel_letter):
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
            markeredgewidth=H_VARIANCE_SEASONALITY_STYLE["marker_edge_width"],
            alpha=FORECAST_ALPHA,
            zorder=2,
        )
    repeated_obs = np.concatenate([observation_variance.to_numpy(), observation_variance.iloc[:11].to_numpy()])
    axis.plot(extended_month_positions, repeated_obs,
              color=OBSERVATION_COLOR, linewidth=OBSERVATION_LINE_WIDTH, zorder=5)
    axis.set_title(f"$\\mathbf{{({panel_letter})}}$ {source['label']}",
                   loc="left", color="black",
                   fontsize=H_VARIANCE_SEASONALITY_STYLE["panel_label_size"],
                   pad=3)
    axis.set_xlim(1, 23)
    axis.set_xticks([1, 4, 7, 10, 13, 16, 19, 22])
    axis.set_xticklabels(["Jul", "Oct", "Jan", "Apr", "Jul", "Oct", "Jan", "Apr"],
                          fontsize=H_VARIANCE_SEASONALITY_STYLE["tick_label_size"])
    axis.tick_params(
        axis="both",
        labelsize=H_VARIANCE_SEASONALITY_STYLE["tick_label_size"],
        length=H_VARIANCE_SEASONALITY_STYLE["tick_length"],
        width=H_VARIANCE_SEASONALITY_STYLE["tick_width"],
        pad=H_VARIANCE_SEASONALITY_STYLE["tick_pad"],
    )
    style_open_axes(axis)


def add_compact_top_legend(figure):
    obs_handle = Line2D([0], [0], color=OBSERVATION_COLOR, linewidth=OBSERVATION_LINE_WIDTH + 0.8, label="Observed")
    season_handles = [
        Line2D([0], [0], color=colour, linewidth=FORECAST_LINE_WIDTH + 0.4, label=f"{season} start")
        for season, colour in SEASON_COLORS.items()
    ]
    marker_handles = [
        Line2D([0], [0], color="0.35", marker=marker, markerfacecolor="white",
               markeredgewidth=H_VARIANCE_SEASONALITY_STYLE["marker_edge_width"],
               linewidth=0, markersize=FORECAST_MARKER_SIZE + 0.4, label=label)
        for marker, label in [("^", "1st month in season"), ("o", "2nd month in season"), ("s", "3rd month in season")]
    ]
    leg1 = figure.legend(handles=[obs_handle] + season_handles, loc="upper center",
                         bbox_to_anchor=(0.5, 0.994), frameon=False,
                         fontsize=H_VARIANCE_SEASONALITY_STYLE["legend_size"], ncol=5,
                         handlelength=1.8, columnspacing=0.65, handletextpad=0.5, labelspacing=0.35)
    figure.add_artist(leg1)
    figure.legend(handles=marker_handles, loc="upper center",
                  bbox_to_anchor=(0.5, 0.970), frameon=False,
                  fontsize=H_VARIANCE_SEASONALITY_STYLE["marker_legend_size"], ncol=3,
                  handlelength=1.6, columnspacing=0.70, handletextpad=0.5, labelspacing=0.30)


def calculate_shared_y_limits(results):
    all_vals = []
    for r in results:
        all_vals.append(r["observation_variance"].to_numpy(dtype=float))
        all_vals.append(r["forecast_variance"].to_numpy(dtype=float).ravel())
    combined = np.concatenate(all_vals)
    finite = combined[np.isfinite(combined)]
    if not len(finite):
        return 0.0, 0.1
    y_min = float(np.nanmin(finite))
    y_max = float(np.nanmax(finite))
    rng = y_max - y_min
    padding = max(0.03, Y_AXIS_PADDING_FRACTION * rng) if rng > 0 else 0.03
    return max(0.0, y_min - padding), y_max + padding


def plot_variance_seasonality(results):
    configure_publication_style()
    figure = plt.figure(figsize=(
        mm_to_inches(H_VARIANCE_SEASONALITY_STYLE["figure_width_mm"]),
        mm_to_inches(H_VARIANCE_SEASONALITY_STYLE["figure_height_mm"]),
    ))
    axes = source_panel_grid_5x2(figure, left=0.10, right=0.98, bottom=0.070, top=0.932,
                                 wspace=0.12, hspace=0.26)
    start_month_styles = {}
    for fm in range(1, 13):
        start_month_styles[fm] = {
            "color": SEASON_COLORS[MONTH_TO_SEASON[fm]],
            "marker": MONTH_POSITION_MARKERS[((fm - 1) % 3) + 1],
        }
    for idx, (axis, result) in enumerate(zip(axes, results)):
        draw_source_panel(axis, result["source"], result["observation_variance"],
                          result["forecast_variance"], start_month_styles, chr(ord("a") + idx))
    add_compact_top_legend(figure)
    style_source_panel_axes_5x2(axes, n_visible=len(results))
    add_shared_axis_labels(figure, xlabel="Forecast verification month", ylabel=spread_axis_label(),
                           xlabel_y=0.040, ylabel_x=0.014,
                           fontsize=H_VARIANCE_SEASONALITY_STYLE["axis_label_size"])
    y_lower, y_upper = calculate_shared_y_limits(results)
    for axis in axes:
        axis.set_ylim(y_lower, y_upper)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_base = OUTPUT_DIR / OUTPUT_BASENAME
    saved_paths = save_publication_figure(
        figure,
        figure_output_paths(output_base),
        dpi=FIGURE_DPI,
        pad_inches=0.03,
    )
    plt.close(figure)
    return saved_paths


def main():
    validate_data_sources(DATA_SOURCES)
    if SPREAD_METRIC not in {"standard_deviation", "variance"}:
        raise ValueError('SPREAD_METRIC must be "standard_deviation" or "variance".')
    source_values = {}
    for source in DATA_SOURCES:
        values = average_duplicate_forecasts(load_source_predictions(source))
        source_values[source["id"]] = values
        print(f"{source['label']}: {len(values)} forecasts after averaging")
    shared_months = common_target_months(source_values)
    shared_start = BASE_YEAR + min(shared_months) // 12
    shared_end = BASE_YEAR + max(shared_months) // 12
    print(f"Common period: {shared_start}-{shared_end} ({len(shared_months)} months)")
    results = []
    for source in DATA_SOURCES:
        values = source_values[source["id"]]
        values = values[values["target_abs_month"].isin(shared_months)].copy()
        ov, fv, sc = calculate_monthly_variances(values)
        print(f"{source['label']}: min samples per group = {int(sc.min().min())}")
        results.append({"source": source, "observation_variance": ov, "forecast_variance": fv})
    saved = plot_variance_seasonality(results)
    for p in saved:
        print(f"Saved figure: {p}")


if __name__ == "__main__":
    main()
