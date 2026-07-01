"""Plot lead-dependent mean Nino3.4 bias for El Nino and La Nina months.

For every data source, overlapping pickle test periods are first averaged by
target month and forecast lead. Bias is then calculated as prediction minus
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
    add_compact_figure_legend,
    configure_publication_style,
    dataset_color,
    save_publication_figure,
    style_light_grid,
    style_open_axes,
    validate_data_sources,
)

# Ensure Unicode output on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# User configuration
INPUT_MONTHS = 6
FORECAST_LEADS = np.arange(1, 19)
BASE_YEAR = 1871
FIGURE_ID = "D"
FIGURE_NAME = "enso_phase_mean_bias"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
OUTPUT_FORMATS = ("png", "pdf")
FIGURE_DPI = 600
RUN_SELF_TEST = True

DATA_SOURCES = get_dl_sources()


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

        start_year = parse_start_year(pickle_path)
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

    assert len(el_nino) == 1 and np.isclose(el_nino["bias"].iloc[0], 0.5), (
        "El Nino threshold, duplicate averaging, or bias sign is incorrect."
    )
    assert len(la_nina) == 1 and np.isclose(la_nina["bias"].iloc[0], -0.2), (
        "La Nina threshold or bias sign is incorrect."
    )
    assert test_rows[test_rows["observation"] >= 2.0].empty, (
        "An absent ENSO phase should be represented by an empty selection."
    )


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
            ("El Nino", monthly["observation"] >= 0.5),
            ("La Nina", monthly["observation"] <= -0.5),
        ):
            summary = (
                monthly.loc[selection]
                .groupby("lead", as_index=False)["bias"]
                .agg(mean_bias="mean", samples="size")
                .set_index("lead")
                .reindex(FORECAST_LEADS)
            )
            phase_bias[source["id"]][phase_name] = summary
            print(f"\n{source['label']} - {phase_name}")
            print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    PHASE_TITLES = {"El Nino": "El Niño", "La Nina": "La Niña"}

    all_bias_values = np.concatenate(
        [
            phase_bias[source["id"]][phase_name]["mean_bias"].dropna().to_numpy()
            for source in DATA_SOURCES
            for phase_name in ("El Nino", "La Nina")
        ]
    )
    if not len(all_bias_values):
        raise ValueError("No El Nino or La Nina forecast months were available.")
    y_padding = max(0.05, 0.08 * np.ptp(all_bias_values))
    y_limits = (all_bias_values.min() - y_padding, all_bias_values.max() + y_padding)

    fig, (ax_en, ax_ln) = plt.subplots(
        1, 2, figsize=(13.5, 5.0), dpi=FIGURE_DPI, sharey=True,
    )
    for phase_name, ax in zip(("El Nino", "La Nina"), (ax_en, ax_ln)):
        for source in DATA_SOURCES:
            summary = phase_bias[source["id"]][phase_name]
            ax.plot(
                FORECAST_LEADS, summary["mean_bias"],
                color=dataset_color(source["id"]),
                linewidth=1.8, marker="o" if phase_name == "El Nino" else "s",
                markersize=3.5,
                label=source["label"],
            )
        ax.axhline(0, color="0.45", linewidth=0.9, linestyle="--", zorder=0)
        ax.set_title(
            PHASE_TITLES[phase_name], loc="left",
            fontsize=PANEL_LABEL_SIZE, fontweight="bold", pad=6,
        )
        ax.set_ylim(*y_limits)
        ax.set_xlim(0.5, int(FORECAST_LEADS.max()) + 0.5)
        ax.set_xticks(FORECAST_LEADS)
        style_light_grid(ax, axis="y", linewidth=0.6)
        style_open_axes(ax)

    ax_en.set_ylabel("Mean bias (°C)", fontsize=AXIS_LABEL_SIZE)
    for ax in (ax_en, ax_ln):
        ax.set_xlabel("Forecast lead (months)", fontsize=AXIS_LABEL_SIZE)

    handles = [Line2D([0], [0], color=dataset_color(s["id"]), linewidth=2.0, label=s["label"]) for s in DATA_SOURCES]
    add_compact_figure_legend(fig, handles=handles, ncol=5, bbox_to_anchor=(0.5, 1.00),
                              fontsize=LEGEND_SIZE, handlelength=1.35,
                              columnspacing=0.40, labelspacing=0.25)
    fig.subplots_adjust(top=0.83, left=0.08, right=0.99, bottom=0.14, wspace=0.10)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_base = OUTPUT_DIR / f"{FIGURE_ID}_{FIGURE_NAME}_by_lead"
    saved_paths = save_publication_figure(
        fig,
        [output_base.with_suffix(f".{fmt}") for fmt in OUTPUT_FORMATS],
        dpi=FIGURE_DPI,
        pad_inches=0.02,
    )
    for p in saved_paths:
        print(f"\nSaved figure: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
