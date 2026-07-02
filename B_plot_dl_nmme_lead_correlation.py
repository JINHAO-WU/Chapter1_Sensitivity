"""
Compare DL Niño3.4 forecast correlations across lead times.

DL scores use the ``real_value`` stored in each pickle test dataset.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from A_basic_sources import (
    FIGURE_ROOT,
    get_dl_sources,
    list_pickle_files,
    load_source_forecast_table,
)
from plot_style import (
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    TICK_LABEL_SIZE,
    add_compact_figure_legend,
    configure_publication_style,
    dataset_color,
    save_publication_figure,
    style_open_axes,
    validate_data_sources,
)

# =============================================================================
# User configuration
# =============================================================================

BASE_YEAR = 1871
FIGURE_ID = "B"
FIGURE_NAME = "dl_nmme_lead_correlation"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
FIGURE_DPI = 600
OUTPUT_FORMATS = ("png", "pdf")
MIN_SAMPLES = 3

# Double-column publication layout (183 mm wide).
FIGURE_WIDTH_INCH = 7.2
FIGURE_HEIGHT_INCH = 3.8

DL_SOURCES = get_dl_sources()

# =============================================================================
# DL: load every pickle in one directory, then ensemble-average duplicate months
# =============================================================================

dl_metrics: dict[str, pd.DataFrame] = {}
validate_data_sources(DL_SOURCES)
dl_labels_by_id = {source["id"]: source["label"] for source in DL_SOURCES}

for source in DL_SOURCES:
    source_id = source["id"]
    dl_table = load_source_forecast_table(
        source, base_year=BASE_YEAR, value_names=("pred", "obs"),
    )
    dl_monthly = (
        dl_table.groupby(["target_month", "lead"], as_index=False)[["pred", "obs"]]
        .mean()
        .dropna()
    )

    rows = []
    for lead, group in dl_monthly.groupby("lead", sort=True):
        n_samples = len(group)
        if (
            n_samples < MIN_SAMPLES
            or group["pred"].std() == 0
            or group["obs"].std() == 0
        ):
            correlation = np.nan
        else:
            correlation = float(group["pred"].corr(group["obs"]))
        rows.append({"lead": int(lead), "r": correlation, "n_samples": n_samples})
    dl_metrics[source_id] = pd.DataFrame(rows)
    print(
        f"{source['label']}: {len(list_pickle_files(source['pickle_dir']))} pickle files; "
        f"{len(dl_monthly)} ensemble-mean monthly forecasts"
    )

print("\nDL Pearson r (pickle real_value):")
for source_id, metrics in dl_metrics.items():
    print(dl_labels_by_id[source_id])
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

# =============================================================================
# Figure
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
configure_publication_style()

figure, axis = plt.subplots(
    1, 1, figsize=(FIGURE_WIDTH_INCH, FIGURE_HEIGHT_INCH),
)

for source_id, metrics in dl_metrics.items():
    axis.plot(
        metrics["lead"], metrics["r"],
        color=dataset_color(source_id),
        marker="o", markersize=4, linewidth=1.8,
        label=dl_labels_by_id[source_id],
    )

axis.axhline(0, color="0.75", linewidth=0.7, zorder=0)
axis.axhline(0.5, color="0.45", linewidth=1.0, linestyle="--", zorder=0)
axis.set_xlim(0.5, 18.5)
axis.set_ylim(-0.15, 1.02)
axis.set_xticks(np.arange(1, 19))
axis.set_xlabel("Forecast lead (months)", fontsize=AXIS_LABEL_SIZE)
axis.set_ylabel("Pearson correlation coefficient", fontsize=AXIS_LABEL_SIZE)
axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
style_open_axes(axis)

handles, labels = axis.get_legend_handles_labels()
add_compact_figure_legend(
    figure,
    handles=handles,
    labels=labels,
    ncol=2,
    fontsize=LEGEND_SIZE,
    handlelength=1.2,
    columnspacing=0.5,
    labelspacing=0.3,
    bbox_to_anchor=(0.98, 0.98),
    loc="upper right",
)

figure.subplots_adjust(
    left=0.12, right=0.98, bottom=0.15, top=0.96,
)

output_base = OUTPUT_DIR / f"{FIGURE_ID}_{FIGURE_NAME}"
saved_paths = save_publication_figure(
    figure,
    [output_base.with_suffix(f".{fmt}") for fmt in OUTPUT_FORMATS],
    dpi=FIGURE_DPI,
    pad_inches=0.02,
)
for output_path in saved_paths:
    print(f"Saved figure: {output_path}")
plt.close(figure)
