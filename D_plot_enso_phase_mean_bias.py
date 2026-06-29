"""Plot lead-dependent mean Niño3.4 bias for El Niño and La Niña months.

For every data source, overlapping pickle test periods are first averaged by
target month and forecast lead.  Bias is then calculated as prediction minus
observation and grouped using the observed Nino3.4 index.
"""

from __future__ import annotations

import sys
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
    parse_start_year,
)
from plot_style import (
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    PANEL_LABEL_SIZE,
    configure_publication_style,
    dataset_color,
    style_open_axes,
    validate_data_sources,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# User configuration
# =============================================================================

INPUT_MONTHS = 6
FORECAST_LEADS = np.arange(1, 19)
BASE_YEAR = 1871
FIGURE_ID = "D"
FIGURE_NAME = "enso_phase_mean_bias"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
OUTPUT_FORMATS = ("png", "pdf")
FIGURE_DPI = 600
RUN_SELF_TEST = True

PHASE_STYLES = {
    "El Niño": {"linestyle": "-", "marker": "o"},
    "La Niña": {"linestyle": "--", "marker": "s"},
}

DATA_SOURCES = get_dl_sources()


def start_year_from_name(pickle_path: Path) -> int:
    """Return the test-set start year encoded in a pickle file name."""
    return parse_start_year(pickle_path)


def source_monthly_forecasts(source: dict) -> pd.DataFrame:
    """Load one source and average duplicate target-month forecasts."""
    pickle_dir = source["pickle_dir"]
    pickle_files = list_pickle_files(pickle_dir)

    expected_samples = 360 - INPUT_MONTHS - len(FORECAST_LEADS)
    tables = []
    for pickle_path in pickle_files:
        prediction, observation = load_prediction_arrays(pickle_path)
        if prediction.shape[0] != expected_samples:
            raise ValueError(
                f"{pickle_path.name}: expected {expected_samples} samples for "
                f"INPUT_MONTHS={INPUT_MONTHS}, found {prediction.shape[0]}."
            )
        if prediction.shape[1] < int(FORECAST_LEADS.max()):
            raise ValueError(
                f"{pickle_path.name}: only {prediction.shape[1]} leads available, but "
                f"lead {int(FORECAST_LEADS.max())} was requested."
            )

        start_year = start_year_from_name(pickle_path)
        sample_index = np.arange(prediction.shape[0])[:, None]
        lead = FORECAST_LEADS[None, :]
        target_month = (
            (start_year - BASE_YEAR) * 12 + sample_index + INPUT_MONTHS + lead - 1
        )
        tables.append(
            pd.DataFrame(
                {
                    "target_month": target_month.ravel(),
                    "lead": np.broadcast_to(lead, prediction[:, : len(FORECAST_LEADS)].shape).ravel(),
                    "prediction": prediction[:, : len(FORECAST_LEADS)].ravel(),
                    "observation": observation[:, : len(FORECAST_LEADS)].ravel(),
                }
            )
        )

    monthly = pd.concat(tables, ignore_index=True)
    monthly = monthly.groupby(["target_month", "lead"], as_index=False).mean()
    monthly = monthly[np.isfinite(monthly["prediction"]) & np.isfinite(monthly["observation"])]
    monthly["bias"] = monthly["prediction"] - monthly["observation"]
    return monthly


def run_self_tests() -> None:
    """Check thresholds, bias sign, duplicate averaging, and missing phases."""
    test_rows = pd.DataFrame(
        {
            "target_month": [1, 1, 2, 3],
            "lead": [1, 1, 1, 1],
            "prediction": [0.8, 1.2, -0.7, 0.0],
            "observation": [0.5, 0.5, -0.5, 0.0],
        }
    )
    test_rows = test_rows.groupby(["target_month", "lead"], as_index=False).mean()
    test_rows["bias"] = test_rows["prediction"] - test_rows["observation"]
    el_nino = test_rows[test_rows["observation"] >= 0.5]
    la_nina = test_rows[test_rows["observation"] <= -0.5]

    if len(el_nino) != 1 or not np.isclose(el_nino["bias"].iloc[0], 0.5):
        raise AssertionError("El Niño threshold, duplicate averaging, or bias sign is incorrect.")
    if len(la_nina) != 1 or not np.isclose(la_nina["bias"].iloc[0], -0.2):
        raise AssertionError("La Niña threshold or bias sign is incorrect.")
    if test_rows[test_rows["observation"] >= 2.0].empty is False:
        raise AssertionError("An absent ENSO phase should be represented by an empty selection.")


def main() -> None:
    configure_publication_style()
    validate_data_sources(DATA_SOURCES)
    if RUN_SELF_TEST:
        run_self_tests()

    phase_bias = {}
    for source in DATA_SOURCES:
        monthly = source_monthly_forecasts(source)
        phase_bias[source["id"]] = {}
        for phase_name, selection in (
            ("El Niño", monthly["observation"] >= 0.5),
            ("La Niña", monthly["observation"] <= -0.5),
        ):
            summary = (
                monthly.loc[selection]
                .groupby("lead", as_index=False)["bias"]
                .agg(mean_bias="mean", samples="size")
                .set_index("lead")
                .reindex(FORECAST_LEADS)
            )
            phase_bias[source["id"]][phase_name] = summary
            print(f"\n{source['label']} — {phase_name}")
            print(summary.to_string(float_format=lambda value: f"{value:.3f}"))

    all_bias_values = np.concatenate(
        [
            phase_bias[source["id"]][phase_name]["mean_bias"].dropna().to_numpy()
            for source in DATA_SOURCES
            for phase_name in ("El Niño", "La Niña")
        ]
    )
    if not len(all_bias_values):
        raise ValueError("No El Niño or La Niña forecast months were available.")
    y_padding = max(0.05, 0.08 * np.ptp(all_bias_values))
    y_limits = (all_bias_values.min() - y_padding, all_bias_values.max() + y_padding)

    figure, axis = plt.subplots(figsize=(7.2, 5.6), dpi=FIGURE_DPI)
    for phase_name in ("El Niño", "La Niña"):
        for source in DATA_SOURCES:
            summary = phase_bias[source["id"]][phase_name]
            phase_style = PHASE_STYLES[phase_name]
            axis.plot(
                FORECAST_LEADS,
                summary["mean_bias"],
                color=dataset_color(source["id"]),
                linestyle=phase_style["linestyle"],
                marker=phase_style["marker"],
                markersize=3.5,
                linewidth=1.8,
            )
    axis.axhline(0, color="0.45", linewidth=0.9, linestyle="--", zorder=0)
    axis.set_title(
        "Mean Niño3.4 forecast bias (pred − obs)",
        loc="left",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        pad=62,
    )
    axis.set_ylabel("Mean bias (°C)", fontsize=AXIS_LABEL_SIZE)
    axis.set_xlabel("Forecast lead (months)", fontsize=AXIS_LABEL_SIZE)
    axis.set_xlim(0.5, int(FORECAST_LEADS.max()) + 0.5)
    axis.set_ylim(*y_limits)
    axis.set_xticks(FORECAST_LEADS)
    axis.grid(True, axis="y", color="#d9d9d9", linewidth=0.6, linestyle=":")
    style_open_axes(axis)

    phase_handles = [
        Line2D(
            [0], [0], color="#444444", linewidth=1.8,
            linestyle=PHASE_STYLES[phase_name]["linestyle"],
            marker=PHASE_STYLES[phase_name]["marker"], markersize=4,
            label=phase_name,
        )
        for phase_name in PHASE_STYLES
    ]
    source_handles = [
        Line2D(
            [0], [0], color=dataset_color(source["id"]), linewidth=2.0,
            label=source["label"],
        )
        for source in DATA_SOURCES
    ]
    phase_legend = axis.legend(
        handles=phase_handles, loc="upper left", bbox_to_anchor=(0.0, 1.21),
        frameon=False, ncol=2, fontsize=LEGEND_SIZE, handlelength=2.4,
    )
    axis.add_artist(phase_legend)
    axis.legend(
        handles=source_handles, loc="upper right", bbox_to_anchor=(1.0, 1.21),
        frameon=False, ncol=3, fontsize=LEGEND_SIZE, handlelength=2.4,
        columnspacing=1.1,
    )
    figure.subplots_adjust(top=0.75, left=0.11, right=0.98, bottom=0.13)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_base = OUTPUT_DIR / f"{FIGURE_ID}_{FIGURE_NAME}_by_lead"
    for output_format in OUTPUT_FORMATS:
        output_path = output_base.with_suffix(f".{output_format}")
        save_kwargs = {"bbox_inches": "tight"}
        if output_format == "png":
            save_kwargs["dpi"] = FIGURE_DPI
        figure.savefig(output_path, **save_kwargs)
        print(f"\nSaved figure: {output_path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
