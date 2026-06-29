"""Shared data sources and pickle-loading helpers for Chapter 1 figures."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


PICKLE_BASE_DIR = Path(r"E:/OneDrive - University of Leeds/A-Research/Study_timeseies/TL_CMIP/File")
FIGURE_ROOT = Path("Figures")

_SOURCE_DEFINITIONS = [
    {
        "id": "source_1",
        "folder": "pickle_HamCNN_input6_var1_sst_NOAA",
        "label": "SST (NOAA)",
    },
    {
        "id": "source_2",
        "folder": "pickle_HamCNN_input6_var1_sst_HadI",
        "label": "SST (HadI)",
    },
    {
        "id": "source_3",
        "folder": "pickle_HamCNN_input6_var1_sst_NOAA_PO",
        "label": "SST (NOAA, Pac)",
    },
    {
        "id": "source_4",
        "folder": "pickle_HamCNN_input6_var2_sst_ohc300_NOAA",
        "label": "SST (NOAA)+OHC300",
    },
    {
        "id": "source_5",
        "folder": "pickle_HamCNN_input6_var1_sst_NOAA_10CMIP6",
        "label": "SST (NOAA+10CMIP6)",
    },
]


def get_dl_sources(sample_size: int | None = None) -> list[dict]:
    """Return independent DL source dictionaries with the shared display labels."""
    sources = []
    for definition in _SOURCE_DEFINITIONS:
        source = {
            "id": definition["id"],
            "label": definition["label"],
            "pickle_dir": PICKLE_BASE_DIR / definition["folder"],
        }
        if sample_size is not None:
            source["sample_size"] = int(sample_size)
        sources.append(source)
    return sources


def validate_pickle_source_dirs(sources: list[dict]) -> None:
    """Validate that every configured source has an existing pickle directory."""
    for source in sources:
        pickle_dir = Path(source["pickle_dir"])
        if not pickle_dir.exists():
            raise FileNotFoundError(f"Pickle directory does not exist: {pickle_dir}")
        if not any(pickle_dir.glob("*.pickle")):
            raise FileNotFoundError(f"No pickle files found in: {pickle_dir}")


def parse_start_year(path: str | Path) -> int:
    """Extract the 4-digit test-start year from a pickle filename."""
    path = Path(path)
    match = re.search(r"_(\d{4})_", path.name)
    if match is None:
        raise ValueError(f"Cannot parse test start year from {path.name!r}.")
    return int(match.group(1))


def parse_input_months(path: str | Path, default: int = 6) -> int:
    """Extract input-window length from a pickle filename, or return default."""
    path = Path(path)
    match = re.search(r"input(\d+)", path.name)
    return int(match.group(1)) if match is not None else int(default)


def list_pickle_files(pickle_dir: str | Path) -> list[Path]:
    """Return sorted pickle files from one directory."""
    pickle_dir = Path(pickle_dir)
    if not pickle_dir.exists():
        raise FileNotFoundError(f"Pickle directory does not exist: {pickle_dir}")

    pickle_files = sorted(pickle_dir.glob("*.pickle"))
    if not pickle_files:
        raise FileNotFoundError(f"No pickle files found in: {pickle_dir}")
    return pickle_files


def list_pickle_files_by_year(pickle_dir: str | Path) -> dict[int, Path]:
    """Return one pickle path per test-start year."""
    file_map: dict[int, list[Path]] = {}
    for path in list_pickle_files(pickle_dir):
        year = parse_start_year(path)
        file_map.setdefault(year, []).append(path)
    return {year: sorted(paths)[0] for year, paths in sorted(file_map.items())}


def load_prediction_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate prediction and observation arrays from one pickle file."""
    path = Path(path)
    with path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected a dict, got {type(data).__name__}")
    if "predict_value" not in data or "real_value" not in data:
        raise KeyError(f"{path.name}: missing predict_value or real_value")

    prediction = np.asarray(data["predict_value"], dtype=float)
    observation = np.asarray(data["real_value"], dtype=float)
    if prediction.shape != observation.shape:
        raise ValueError(
            f"{path.name}: predict_value and real_value have different shapes "
            f"{prediction.shape} vs {observation.shape}."
        )
    if prediction.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2D array, got {prediction.ndim}D.")
    return prediction, observation


def load_source_forecast_table(
    source: dict,
    base_year: int,
    input_months: int | None = None,
    leads: list[int] | np.ndarray | None = None,
    value_names: tuple[str, str] = ("pred", "real"),
) -> pd.DataFrame:
    """
    Load one source into a long-format forecast table.

    The table contains one row per target month and lead, with both ``abs_month``
    and ``target_month`` set to the zero-based absolute target-month index.
    """
    prediction_name, observation_name = value_names
    records = []

    for pickle_path in list_pickle_files(source["pickle_dir"]):
        try:
            start_year = parse_start_year(pickle_path)
        except ValueError:
            continue
        file_input_months = parse_input_months(
            pickle_path,
            default=input_months if input_months is not None else 6,
        )
        if input_months is not None and file_input_months != input_months:
            raise ValueError(
                f"{pickle_path.name}: input window is {file_input_months} months, "
                f"but input_months is {input_months}."
            )

        prediction, observation = load_prediction_arrays(pickle_path)
        available_leads = np.arange(1, prediction.shape[1] + 1)
        selected_leads = available_leads if leads is None else np.asarray(leads, dtype=int)
        if selected_leads.size == 0:
            continue
        if selected_leads.min() < 1 or selected_leads.max() > prediction.shape[1]:
            raise ValueError(
                f"{pickle_path.name}: requested lead range {selected_leads.min()}-"
                f"{selected_leads.max()}, but only {prediction.shape[1]} leads are available."
            )

        lead_indices = selected_leads - 1
        selected_prediction = prediction[:, lead_indices]
        selected_observation = observation[:, lead_indices]
        sample_index = np.arange(prediction.shape[0])[:, None]
        lead_values = selected_leads[None, :]
        abs_month = (
            (start_year - base_year) * 12
            + sample_index
            + file_input_months
            + lead_values
            - 1
        )
        records.append(
            pd.DataFrame(
                {
                    "abs_month": abs_month.ravel(),
                    "target_month": abs_month.ravel(),
                    "leading": np.broadcast_to(lead_values, selected_prediction.shape).ravel(),
                    "lead": np.broadcast_to(lead_values, selected_prediction.shape).ravel(),
                    "input_len": file_input_months,
                    "pickle_year": start_year,
                    prediction_name: selected_prediction.ravel(),
                    observation_name: selected_observation.ravel(),
                }
            )
        )

    if not records:
        raise ValueError(f"No usable pickle files with parseable years in: {source['pickle_dir']}")

    table = pd.concat(records, ignore_index=True)
    table["year"] = int(base_year) + table["abs_month"] // 12
    table["month"] = table["abs_month"] % 12 + 1
    return table
