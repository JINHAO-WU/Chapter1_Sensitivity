"""
Compare DL and NMME Niño3.4 forecast correlations across lead times.

DL scores use the ``real_value`` stored in each pickle test dataset. NMME
hindcasts use OIv2 and are restricted automatically to the OIv2 period.
The two groups therefore provide a visual comparison, not a same-sample
statistical ranking.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from plot_style import (
    AXIS_LABEL_SIZE as SHARED_AXIS_LABEL_SIZE,
    LEGEND_SIZE as SHARED_LEGEND_SIZE,
    PANEL_LABEL_SIZE,
    TICK_LABEL_SIZE as SHARED_TICK_LABEL_SIZE,
    configure_publication_style,
    dataset_color,
    nmme_color_mapping,
    style_open_axes,
    validate_data_sources,
)


# =============================================================================
# User configuration
# =============================================================================

BASE_YEAR = 1871
OUTPUT_DIR = Path("nmme_comparison_figures")
FIGURE_DPI = 600
OUTPUT_FORMATS = ("png", "pdf")
MIN_SAMPLES = 3

# Double-column publication layout (183 mm wide) with readable final-size text.
FIGURE_WIDTH_INCH = 7.2
FIGURE_HEIGHT_INCH = 7.8
AXIS_LABEL_SIZE = SHARED_AXIS_LABEL_SIZE
TICK_LABEL_SIZE = SHARED_TICK_LABEL_SIZE
LEGEND_SIZE = SHARED_LEGEND_SIZE

DL_SOURCES = [
    {"id": "source_1", 
     "label": "SST_NOAA", 
     "pickle_dir": Path(r"E:/OneDrive - University of Leeds/A-Research/Study_timeseies/TL_CMIP/File/pickle_HamCNN_input6_var1_sst_NOAA")
    },
    {"id": "source_2", 
     "label": "SST_HadI", 
     "pickle_dir": Path(r"E:/OneDrive - University of Leeds/A-Research/Study_timeseies/TL_CMIP/File/pickle_HamCNN_input6_var1_sst_HadI")
    },
    {"id": "source_3", 
     "label": "SST_NOAA_PO", 
     "pickle_dir": Path(r"E:/OneDrive - University of Leeds/A-Research/Study_timeseies/TL_CMIP/File/pickle_HamCNN_input6_var1_sst_NOAA_PO")
    },
    {"id": "source_4", 
     "label": "SST_OHC300_NOAA", 
     "pickle_dir": Path(r"E:/OneDrive - University of Leeds/A-Research/Study_timeseies/TL_CMIP/File/pickle_HamCNN_input6_var2_sst_ohc300_NOAA")
    },
    {"id": "source_5", 
     "label": "SST_NOAA_5MIROC6", 
     "pickle_dir": Path(r"E:/OneDrive - University of Leeds/A-Research/Study_timeseies/TL_CMIP/File/pickle_HamCNN_input6_var1_sst_NOAA_5MIROC6")
    },
]   

NMME_DIR = Path(r"E:/OneDrive - University of Leeds/A-Research/Study_timeseies/Data/NMME")
OIV2_FILE = NMME_DIR / "NMME_OIv2.nc"

# Delete models you do not want. Set to [] or "all" to use every Hindcast file.
NMME_MODELS: list[str] | str = [
    "CanCM4i",
    "CanSIPS-IC3",
    "CanSIPS-IC4",
    "CanSIPSv2",
    "CMC1-CanCM3",
    "CMC2-CanCM4",
    "NCEP-CFSv2",
]

MME_COLOR = "#000000"


def monthly_anomaly_correlations(pairs: pd.DataFrame) -> pd.DataFrame:
    """Calculate lead-wise r after removing paired-period monthly climatologies."""
    rows = []
    for lead, group in pairs.groupby("lead", sort=True):
        group = group.dropna(subset=["pred", "obs"]).copy()
        group["pred_anomaly"] = group["pred"] - group.groupby("target_month")["pred"].transform("mean")
        group["obs_anomaly"] = group["obs"] - group.groupby("target_month")["obs"].transform("mean")
        n_samples = len(group)
        if (
            n_samples < MIN_SAMPLES
            or group["pred_anomaly"].std() == 0
            or group["obs_anomaly"].std() == 0
        ):
            correlation = np.nan
        else:
            correlation = float(group["pred_anomaly"].corr(group["obs_anomaly"]))
        rows.append({"lead": int(lead), "r": correlation, "n_samples": n_samples})
    return pd.DataFrame(rows)


# =============================================================================
# DL: load every pickle in one directory, then ensemble-average duplicate months
# =============================================================================

dl_metrics: dict[str, pd.DataFrame] = {}
validate_data_sources(DL_SOURCES)
dl_labels_by_id = {source["id"]: source["label"] for source in DL_SOURCES}

for source in DL_SOURCES:
    source_id = source["id"]
    source_label = source["label"]
    pickle_dir = source["pickle_dir"]
    if not pickle_dir.exists():
        raise FileNotFoundError(f"DL pickle directory does not exist: {pickle_dir}")

    pickle_files = sorted(pickle_dir.glob("*.pickle"))
    if not pickle_files:
        raise FileNotFoundError(f"No pickle files found in: {pickle_dir}")

    tables = []
    for pickle_path in pickle_files:
        year_match = re.search(r"_(\d{4})_", pickle_path.name)
        input_match = re.search(r"input(\d+)", pickle_path.name)
        if year_match is None:
            continue

        start_year = int(year_match.group(1))
        input_months = int(input_match.group(1)) if input_match else 6
        with pickle_path.open("rb") as handle:
            data = pickle.load(handle)

        prediction = np.asarray(data["predict_value"], dtype=float)
        observation = np.asarray(data["real_value"], dtype=float)
        if prediction.shape != observation.shape or prediction.ndim != 2:
            raise ValueError(f"Unexpected prediction/observation shape in {pickle_path.name}")

        sample_index = np.arange(prediction.shape[0])[:, None]
        lead = np.arange(1, prediction.shape[1] + 1)[None, :]
        target_month = (start_year - BASE_YEAR) * 12 + sample_index + input_months + lead - 1
        tables.append(
            pd.DataFrame(
                {
                    "target_month": target_month.ravel(),
                    "lead": np.broadcast_to(lead, prediction.shape).ravel(),
                    "pred": prediction.ravel(),
                    "obs": observation.ravel(),
                }
            )
        )

    dl_monthly = pd.concat(tables, ignore_index=True)
    dl_monthly = (
        dl_monthly.groupby(["target_month", "lead"], as_index=False)[["pred", "obs"]]
        .mean()
        .dropna()
    )

    rows = []
    for lead, group in dl_monthly.groupby("lead", sort=True):
        n_samples = len(group)
        if n_samples < MIN_SAMPLES or group["pred"].std() == 0 or group["obs"].std() == 0:
            correlation = np.nan
        else:
            correlation = float(group["pred"].corr(group["obs"]))
        rows.append({"lead": int(lead), "r": correlation, "n_samples": n_samples})
    dl_metrics[source_id] = pd.DataFrame(rows)
    print(f"{source_label}: {len(pickle_files)} pickle files; {len(dl_monthly)} ensemble-mean monthly forecasts")


# =============================================================================
# NMME: select Hindcast models, pair valid_time = S + L with OIv2, then score
# =============================================================================

hindcast_files = sorted(NMME_DIR.glob("NMME_*_Hindcast_EA.nc"))
available_models = {
    path.name.removeprefix("NMME_").removesuffix("_Hindcast_EA.nc"): path
    for path in hindcast_files
}
print(f"Available NMME models: {', '.join(available_models)}")

if NMME_MODELS == "all" or NMME_MODELS == []:
    selected_models = list(available_models)
else:
    selected_models = list(NMME_MODELS)
    missing_models = sorted(set(selected_models) - set(available_models))
    if missing_models:
        raise ValueError(f"Selected NMME models not found: {missing_models}")

if not selected_models:
    raise ValueError("No NMME models selected.")
print(f"Selected NMME models: {', '.join(selected_models)}")

with xr.open_dataset(OIV2_FILE, decode_times=False) as oiv2_dataset:
    oiv2_time = np.asarray(oiv2_dataset["T"].values, dtype=float)
    oiv2_sst = np.asarray(oiv2_dataset["sst"].values, dtype=float)

oiv2 = pd.DataFrame({"valid_time": np.round(oiv2_time, 6), "obs": oiv2_sst})
oiv2 = oiv2[np.isfinite(oiv2["obs"]) & (oiv2["obs"] != -999.0)].copy()
print(f"OIv2 verification period: {oiv2['valid_time'].min():.1f} to {oiv2['valid_time'].max():.1f}")

nmme_metrics: dict[str, pd.DataFrame] = {}
nmme_pairs = []

for model_name in selected_models:
    with xr.open_dataset(available_models[model_name], decode_times=False) as dataset:
        start_time = np.asarray(dataset["S"].values, dtype=float)
        raw_lead = np.asarray(dataset["L"].values, dtype=float)
        prediction = np.asarray(dataset["sst"].values, dtype=float)
        prediction_units = str(dataset["sst"].attrs.get("units", "")).lower()

    # The selected files mix Kelvin and Celsius. Use one physical unit before
    # averaging models; a constant unit offset does not affect individual r.
    if "kelvin" in prediction_units or prediction_units == "k":
        prediction = prediction - 273.15
        print(f"{model_name}: converted forecast SST from Kelvin to Celsius")
    elif "celsius" not in prediction_units and prediction_units not in {"c", "degc", "degree_celsius"}:
        print(f"{model_name}: unrecognised SST units {prediction_units!r}; assuming Celsius")

    lead_month = np.rint(raw_lead - 0.5).astype(int)
    valid_time = start_time[:, None] + raw_lead[None, :]
    model_pairs = pd.DataFrame(
        {
            "valid_time": np.round(valid_time.ravel(), 6),
            "lead": np.broadcast_to(lead_month, prediction.shape).ravel(),
            "pred": prediction.ravel(),
        }
    )
    model_pairs = model_pairs.merge(oiv2, on="valid_time", how="inner")
    model_pairs = model_pairs[np.isfinite(model_pairs["pred"])].copy()
    model_pairs["target_month"] = np.floor(model_pairs["valid_time"]).astype(int) % 12 + 1

    if model_pairs.empty:
        print(f"{model_name}: no overlap with OIv2; skipped")
        continue

    # L=0.5 is NMME lead 0 (initial-condition month), which is not plotted.
    model_pairs = model_pairs[model_pairs["lead"] > 0].copy()
    nmme_metrics[model_name] = monthly_anomaly_correlations(model_pairs)
    model_pairs["model"] = model_name
    nmme_pairs.append(model_pairs)
    print(f"{model_name}: {len(model_pairs)} OIv2-matched forecast values")

if not nmme_pairs:
    raise ValueError("None of the selected NMME models overlap OIv2.")

# Average selected NMME models first, then calculate MME correlation with OIv2.
mme_pairs = pd.concat(nmme_pairs, ignore_index=True)
mme_pairs = (
    mme_pairs.groupby(["valid_time", "lead"], as_index=False)
    .agg(pred=("pred", "mean"), obs=("obs", "first"))
)
mme_pairs["target_month"] = np.floor(mme_pairs["valid_time"]).astype(int) % 12 + 1
mme_metrics = monthly_anomaly_correlations(mme_pairs)

print("\nDL Pearson r (pickle real_value):")
for source_id, metrics in dl_metrics.items():
    print(dl_labels_by_id[source_id])
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print("\nNMME Pearson r (monthly anomalies against OIv2):")
for label, metrics in nmme_metrics.items():
    print(label)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print(f"MME ({len(nmme_metrics)} selected models)")
print(mme_metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


# =============================================================================
# Figure: DL + MME above, selected NMME models + MME below
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
configure_publication_style()
figure, axes = plt.subplots(
    2,
    1,
    figsize=(FIGURE_WIDTH_INCH, FIGURE_HEIGHT_INCH),
    sharey=True,
    constrained_layout=True,
)

for source_id, metrics in dl_metrics.items():
    axes[0].plot(metrics["lead"], metrics["r"], color=dataset_color(source_id), marker="o", markersize=4, linewidth=1.8, label=dl_labels_by_id[source_id])
axes[0].plot(mme_metrics["lead"], mme_metrics["r"], color=MME_COLOR, marker="o", markersize=4.5, linewidth=2.7, label=f"NMME MME (n={len(nmme_metrics)})")
axes[0].set_title("(a) DL forecasts and NMME MME", loc="left", fontsize=PANEL_LABEL_SIZE, fontweight="bold")
# axes[0].text(0.99, 0.05, "DL: pickle real_value; NMME: OIv2", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=LEGEND_SIZE)

nmme_colors_by_label = nmme_color_mapping(list(nmme_metrics))
for label, metrics in nmme_metrics.items():
    axes[1].plot(metrics["lead"], metrics["r"], color=nmme_colors_by_label[label], marker="o", markersize=3.5, linewidth=1.5, alpha=0.9, label=label)
axes[1].plot(mme_metrics["lead"], mme_metrics["r"], color=MME_COLOR, marker="o", markersize=4.5, linewidth=2.7, label=f"MME (n={len(nmme_metrics)})")
axes[1].set_title("(b) Individual NMME models and MME", loc="left", fontsize=PANEL_LABEL_SIZE, fontweight="bold")

for axis in axes:
    axis.axhline(0, color="0.75", linewidth=0.7, zorder=0)
    axis.axhline(0.5, color="0.45", linewidth=1.0, linestyle="--", zorder=0)
    axis.set_xlim(0.5, 18.5)
    axis.set_ylim(-0.15, 1.02)
    axis.set_xticks(np.arange(1, 19))
    axis.set_ylabel("Pearson correlation coefficient", fontsize=AXIS_LABEL_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    style_open_axes(axis)
    axis.legend(frameon=False, ncol=3, fontsize=LEGEND_SIZE, handlelength=2.1, columnspacing=1.1)

axes[1].set_xlabel("Forecast lead (months)", fontsize=AXIS_LABEL_SIZE)
output_base = OUTPUT_DIR / "dl_nmme_lead_correlation"
for output_format in OUTPUT_FORMATS:
    output_path = output_base.with_suffix(f".{output_format}")
    save_kwargs = {"bbox_inches": "tight"}
    if output_format == "png":
        save_kwargs["dpi"] = FIGURE_DPI
    figure.savefig(output_path, **save_kwargs)
    print(f"Saved figure: {output_path}")
plt.close(figure)
