"""Shared data sources and pickle-loading helpers for Chapter 1 figures."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


PICKLE_BASE_DIR = Path(r"E:/OneDrive - University of Leeds/A-Research/Study_timeseies/TL_CMIP/File")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_ROOT = PROJECT_ROOT / "Figures_ALL"

_SOURCE_DEFINITIONS = [
    {"id": "source_1", "folder": "pickle_HamCNN_input6_var1_sst_NOAA", "label": "SST (NOAA)"},
    {"id": "source_2", "folder": "pickle_HamCNN_input6_var1_sst_HadI", "label": "SST (HadISST)"},
    {"id": "source_3", "folder": "pickle_HamCNN_input6_var1_sst_NOAA_PO", "label": "SST (NOAA$_{\\mathrm{Pac}}$)"},
    {"id": "source_4", "folder": "pickle_HamCNN_input6_var2_sst_ohc300_NOAA", "label": "SST (NOAA+OHC300)"},
    {"id": "source_5", "folder": "pickle_HamCNN_input6_var1_sst_NOAA_10MIROC6", "label": "SST (NOAA+10MIROC6)"},
    {"id": "source_6", "folder": "pickle_HamCNN_input6_var1_sst_NOAA_10MIROC6_CanESM5", "label": "SST (NOAA+10MIROC6+10CanESM5)"},
    {"id": "source_7", "folder": "pickle_HamCNN_input6_var1_sst_10CMIP6", "label": "SST (10CMIP6)"},
    {"id": "source_8", "folder": "pickle_HamCNN_input6_var1_sst_NOAA_10CMIP6", "label": "SST (NOAA+10CMIP6)"},
    {"id": "source_9", "folder": "pickle_HamCNN_input6_var1_sst_NOAA_10CMIP6_TL", "label": "SST (TL-10CMIP6+NOAA)"},
    {"id": "source_10", "folder": "pickle_HamCNN_input6_var1_sst_NOAA_20CMIP6", "label": "SST (NOAA+20CMIP6)"},
]

def get_dl_sources(sample_size=None):
    sources = []
    for d in _SOURCE_DEFINITIONS:
        s = {"id": d["id"], "label": d["label"], "pickle_dir": PICKLE_BASE_DIR / d["folder"]}
        if sample_size is not None:
            s["sample_size"] = int(sample_size)
        sources.append(s)
    return sources

def validate_pickle_source_dirs(sources):
    for s in sources:
        d = Path(s["pickle_dir"])
        if not d.exists():
            raise FileNotFoundError(f"Pickle directory does not exist: {d}")
        if not any(d.glob("*.pickle")):
            raise FileNotFoundError(f"No pickle files found in: {d}")

def parse_start_year(path):
    path = Path(path)
    m = re.search(r"_(\d{4})_", path.name)
    if m is None:
        raise ValueError(f"Cannot parse test start year from {path.name!r}.")
    return int(m.group(1))

def parse_input_months(path, default=6):
    path = Path(path)
    m = re.search(r"input(\d+)", path.name)
    return int(m.group(1)) if m is not None else int(default)

def list_pickle_files(pickle_dir):
    pickle_dir = Path(pickle_dir)
    if not pickle_dir.exists():
        raise FileNotFoundError(f"Pickle directory does not exist: {pickle_dir}")
    files = sorted(pickle_dir.glob("*.pickle"))
    if not files:
        raise FileNotFoundError(f"No pickle files found in: {pickle_dir}")
    return files

def list_pickle_files_by_year(pickle_dir):
    fm = {}
    for p in list_pickle_files(pickle_dir):
        yr = parse_start_year(p)
        fm.setdefault(yr, []).append(p)
    return {y: sorted(ps)[0] for y, ps in sorted(fm.items())}

def load_prediction_arrays(path):
    path = Path(path)
    with path.open("rb") as h:
        data = pickle.load(h)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected a dict, got {type(data).__name__}")
    if "predict_value" not in data or "real_value" not in data:
        raise KeyError(f"{path.name}: missing predict_value or real_value")
    pred = np.asarray(data["predict_value"], dtype=float)
    obs = np.asarray(data["real_value"], dtype=float)
    if pred.shape != obs.shape:
        raise ValueError(f"{path.name}: shape mismatch {pred.shape} vs {obs.shape}")
    if pred.ndim != 2:
        raise ValueError(f"{path.name}: expected 2D, got {pred.ndim}D")
    return pred, obs

def load_source_forecast_table(source, base_year, input_months=None, leads=None, value_names=("pred","real")):
    pn, on = value_names
    recs = []
    for pp in list_pickle_files(source["pickle_dir"]):
        try:
            sy = parse_start_year(pp)
        except ValueError:
            continue
        fim = parse_input_months(pp, default=input_months if input_months is not None else 6)
        if input_months is not None and fim != input_months:
            raise ValueError(f"{pp.name}: input window mismatch")
        pr, ob = load_prediction_arrays(pp)
        av = np.arange(1, pr.shape[1]+1)
        sl = av if leads is None else np.asarray(leads, dtype=int)
        if sl.size == 0:
            continue
        if sl.min() < 1 or sl.max() > pr.shape[1]:
            raise ValueError(f"{pp.name}: lead range exceeds available")
        li = sl - 1
        sp = pr[:, li]; so = ob[:, li]
        si = np.arange(pr.shape[0])[:, None]
        lv = sl[None, :]
        am = (sy - base_year)*12 + si + fim + lv - 1
        recs.append(pd.DataFrame({"abs_month": am.ravel(), "target_month": am.ravel(),
            "leading": np.broadcast_to(lv, sp.shape).ravel(), "lead": np.broadcast_to(lv, sp.shape).ravel(),
            "input_len": fim, "pickle_year": sy, pn: sp.ravel(), on: so.ravel()}))
    if not recs:
        raise ValueError(f"No usable pickle files in: {source['pickle_dir']}")
    tbl = pd.concat(recs, ignore_index=True)
    tbl["year"] = int(base_year) + tbl["abs_month"]//12
    tbl["month"] = tbl["abs_month"]%12 + 1
    return tbl
