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
    DEFAULT_FIGURE_DPI,
    add_compact_figure_legend,
    configure_publication_style,
    dataset_color,
    figure_output_paths,
    mm_to_inches,
    save_publication_figure,
    style_light_grid,
    style_open_axes,
    validate_data_sources,
)


# =============================================================================
# User configuration
# =============================================================================

D_ENSO_PHASE_STYLE = {
    "axis_label_size": 8.8,
    "tick_label_size": 8.0,
    "panel_label_size": 8.8,
    "legend_size": 6.8,
    "line_width": 1.25,
    "legend_line_width": 1.45,
    "marker_size": 3.0,
    "zero_line_width": 0.65,
    "grid_line_width": 0.55,
    "tick_length": 2.4,
    "tick_width": 0.6,
    "figure_width_mm": 183,
    "figure_height_mm": 132,
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


INPUT_MONTHS = 6
FORECAST_LEADS = np.arange(1, 19)
BASE_YEAR = 1871
FIGURE_ID = "D"
FIGURE_NAME = "enso_phase_mean_bias"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
FIGURE_DPI = DEFAULT_FIGURE_DPI
RUN_SELF_TEST = True

DATA_SOURCES = get_dl_sources()


def source_monthly_forecasts(source):
    pickle_dir = source["pickle_dir"]
    pickle_files = list_pickle_files(pickle_dir)

    expected_samples = 360 - INPUT_MONTHS - len(FORECAST_LEADS)
    tables = []
    for pickle_path in pickle_files:
        prediction, observation = load_prediction_arrays(pickle_path)
        if prediction.shape[0] != expected_samples:
            raise ValueError(
                f"{pickle_path.name}: expected {expected_samples} samples, "
                f"found {prediction.shape[0]}."
            )
        if prediction.shape[1] < int(FORECAST_LEADS.max()):
            raise ValueError(
                f"{pickle_path.name}: only {prediction.shape[1]} leads "
                f"available, lead {int(FORECAST_LEADS.max())} requested."
            )

        start_year = parse_start_year(pickle_path)
        sample_index = np.arange(prediction.shape[0])[:, None]
        lead = FORECAST_LEADS[None, :]
        target_month = (
            (start_year - BASE_YEAR) * 12 + sample_index + INPUT_MONTHS + lead - 1
        )
        tables.append(
            pd.DataFrame({
                "target_month": target_month.ravel(),
                "lead": np.broadcast_to(
                    lead, prediction[:, :len(FORECAST_LEADS)].shape
                ).ravel(),
                "prediction": prediction[:, :len(FORECAST_LEADS)].ravel(),
                "observation": observation[:, :len(FORECAST_LEADS)].ravel(),
            })
        )

    monthly = pd.concat(tables, ignore_index=True)
    monthly = monthly.groupby(["target_month", "lead"], as_index=False).mean()
    monthly = monthly[
        np.isfinite(monthly["prediction"]) & np.isfinite(monthly["observation"])
    ]
    monthly["bias"] = monthly["prediction"] - monthly["observation"]
    return monthly


def run_self_tests():
    test_rows = pd.DataFrame({
        "target_month": [1, 1, 2, 3],
        "lead": [1, 1, 1, 1],
        "prediction": [0.8, 1.2, -0.7, 0.0],
        "observation": [0.5, 0.5, -0.5, 0.0],
    })
    test_rows = test_rows.groupby(["target_month", "lead"], as_index=False).mean()
    test_rows["bias"] = test_rows["prediction"] - test_rows["observation"]
    el_nino = test_rows[test_rows["observation"] >= 0.5]
    la_nina = test_rows[test_rows["observation"] <= -0.5]

    assert len(el_nino) == 1 and np.isclose(el_nino["bias"].iloc[0], 0.5), (
        "El Nino test failed."
    )
    assert len(la_nina) == 1 and np.isclose(la_nina["bias"].iloc[0], -0.2), (
        "La Nina test failed."
    )
    assert test_rows[test_rows["observation"] >= 2.0].empty, (
        "Empty phase test failed."
    )


def main():
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

    PHASE_TITLES = {"El Nino": r"El Ni$\tilde{n}$o", "La Nina": r"La Ni$\tilde{n}$a"}

    all_bias_values = np.concatenate([
        phase_bias[sid][pn]["mean_bias"].dropna().to_numpy()
        for sid in (s["id"] for s in DATA_SOURCES)
        for pn in ("El Nino", "La Nina")
    ])
    if not len(all_bias_values):
        raise ValueError("No El Nino or La Nina forecast months available.")
    y_padding = max(0.05, 0.08 * np.ptp(all_bias_values))
    y_limits = (all_bias_values.min() - y_padding, all_bias_values.max() + y_padding)

    fig, (ax_en, ax_ln) = plt.subplots(
        2,
        1,
        figsize=(
            mm_to_inches(D_ENSO_PHASE_STYLE["figure_width_mm"]),
            mm_to_inches(D_ENSO_PHASE_STYLE["figure_height_mm"]),
        ),
        dpi=FIGURE_DPI,
        sharex=True,
    )
    for phase_name, ax, panel_label in zip(
        ("El Nino", "La Nina"), (ax_en, ax_ln), ("a", "b")
    ):
        for source in DATA_SOURCES:
            summary = phase_bias[source["id"]][phase_name]
            ax.plot(
                FORECAST_LEADS, summary["mean_bias"],
                color=dataset_color(source["id"]),
                linewidth=D_ENSO_PHASE_STYLE["line_width"],
                marker="o" if phase_name == "El Nino" else "s",
                markersize=D_ENSO_PHASE_STYLE["marker_size"],
                label=source["label"],
            )
        ax.axhline(
            0,
            color="0.45",
            linewidth=D_ENSO_PHASE_STYLE["zero_line_width"],
            linestyle="--",
            zorder=0,
        )
        title_text = f"$\\mathbf{{({panel_label})}}$ {PHASE_TITLES[phase_name]}"
        ax.set_title(title_text, loc="left", fontsize=D_ENSO_PHASE_STYLE["panel_label_size"], pad=4)
        ax.set_ylim(*y_limits)
        style_light_grid(ax, axis="y", linewidth=D_ENSO_PHASE_STYLE["grid_line_width"])
        style_open_axes(ax)
        ax.tick_params(
            axis="both",
            labelsize=D_ENSO_PHASE_STYLE["tick_label_size"],
            length=D_ENSO_PHASE_STYLE["tick_length"],
            width=D_ENSO_PHASE_STYLE["tick_width"],
            pad=2.0,
        )

    # Shared y-label centred between the two panels
    fig.supylabel(
        "Mean bias (degC)",
        fontsize=D_ENSO_PHASE_STYLE["axis_label_size"],
        x=0.045,
    )
    # x-axis label only on bottom panel; hide top panel x tick labels
    ax_en.tick_params(axis="x", labelbottom=False)
    ax_ln.set_xlabel(
        "Forecast lead (months)",
        fontsize=D_ENSO_PHASE_STYLE["axis_label_size"],
    )
    for ax in (ax_en, ax_ln):
        ax.set_xlim(0.5, int(FORECAST_LEADS.max()) + 0.5)
        ax.set_xticks(FORECAST_LEADS)

    handles = [
        Line2D(
            [0],
            [0],
            color=dataset_color(s["id"]),
            linewidth=D_ENSO_PHASE_STYLE["legend_line_width"],
            label=s["label"],
        )
        for s in DATA_SOURCES
    ]
    add_compact_figure_legend(
        fig,
        handles=handles,
        ncol=5,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=D_ENSO_PHASE_STYLE["legend_size"],
        handlelength=1.20,
        columnspacing=0.35,
        labelspacing=0.20,
    )
    fig.subplots_adjust(top=0.895, left=0.105, right=0.985, bottom=0.105, hspace=0.18)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_base = OUTPUT_DIR / f"{FIGURE_ID}_{FIGURE_NAME}_by_lead"
    saved_paths = save_publication_figure(
        fig,
        figure_output_paths(output_base),
        dpi=FIGURE_DPI,
        pad_inches=0.02,
    )
    for p in saved_paths:
        print(f"\nSaved figure: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
