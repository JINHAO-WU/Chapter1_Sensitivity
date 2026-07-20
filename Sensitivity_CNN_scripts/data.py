"""ENSO observation/CMIP6 loading, period slicing and PyTorch dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import ConcatDataset, Dataset

import A_Parameter_set as par


class ENSOData_load(Dataset):
    """Create sliding input windows from one or more aligned predictor fields."""

    def __init__(self, variables, enso, data_time_len, input_length, leading_time):
        if not variables:
            raise ValueError("At least one input variable is required")
        self.variables = [np.asarray(variable) for variable in variables]
        self.enso = np.asarray(enso)
        self.data_time_len = min(data_time_len, len(self.enso), *(len(item) for item in self.variables))
        self.input_length, self.leading_time = input_length, leading_time

    def __len__(self):
        return max(0, self.data_time_len - self.input_length - self.leading_time)

    def __getitem__(self, index):
        stop = index + self.input_length
        inputs = np.concatenate([variable[index:stop] for variable in self.variables], axis=0)
        targets = self.enso[stop:stop + self.leading_time]
        return torch.as_tensor(inputs, dtype=torch.float32), torch.as_tensor(targets, dtype=torch.float32)


def load_observations(config=par):
    """Load selected variables and ENSO, aligned by calendar month and grid."""
    variables = {}
    for name in config.input_variables:
        if name not in config.OBSERVATION_VARIABLES:
            raise ValueError(f"Input variable '{name}' is not registered")
        spec = config.OBSERVATION_VARIABLES[name]
        field = xr.open_dataset(Path(spec["path"]))[spec["field"]].fillna(0)
        # Source products use different days within the same month; align them
        # on the monthly period rather than dropping otherwise matching months.
        field = field.assign_coords(time=pd.to_datetime(field.time.values).to_period("M").to_timestamp())
        variables[name] = field.loc[config.time_start:config.time_end, config.lat_start:config.lat_end, config.lon_start:config.lon_end]

    enso = xr.open_dataset(Path(config.ENSO_PATH))[config.ENSO_FIELD].fillna(0)
    enso = enso.assign_coords(time=pd.to_datetime(enso.time.values).to_period("M").to_timestamp())
    enso = enso.loc[config.time_start:config.time_end].squeeze()
    aligned = xr.align(*variables.values(), enso, join="inner")
    return dict(zip(variables, aligned[:-1])), aligned[-1]


def cut_data(variables, enso, dataset_label, start, end, split):
    """Select an inclusive period from every predictor and the target index."""
    selected = {name: field.loc[start:end] for name, field in variables.items()}
    selected_enso = enso.loc[start:end]
    print(f"  {split}: {dataset_label} (Time: {start} - {end})")
    print(f"  Variables: {', '.join(selected)}")
    return {"variables": selected, "enso": selected_enso, "data_time_len": selected_enso.shape[0]}


def load_cmip_datasets(config=par, model_names=None):
    """Return SST-only CMIP6 datasets selected by config.CMIP_MODELS.

    ``ALL`` scans SST files and uses only names with a matching ENSO file.
    Explicit model names remain supported for focused experiments.
    """
    if config.input_variables != ("sst",):
        raise ValueError("CMIP6 data is available only for input_variables = ('sst',)")

    selection = config.CMIP_MODELS if model_names is None else model_names
    if isinstance(selection, str):
        selection = (selection,)
    root = Path(config.CMIP_ROOT)
    sst_dir, enso_dir = root / "SST", root / "ENSO"
    if "ALL" in selection:
        names = sorted(
            path.stem.removeprefix("sst_")
            for path in sst_dir.glob("sst_*.nc")
            if (enso_dir / f"nino34_{path.stem.removeprefix('sst_')}.nc").is_file()
        )
    else:
        names = tuple(selection)

    if not names:
        raise FileNotFoundError(f"No paired CMIP6 SST/ENSO files found under {root}")
    datasets = []
    for model_name in names:
        sst_path = sst_dir / f"sst_{model_name}.nc"
        enso_path = enso_dir / f"nino34_{model_name}.nc"
        if not sst_path.is_file() or not enso_path.is_file():
            raise FileNotFoundError(f"Missing CMIP6 pair for '{model_name}': {sst_path}, {enso_path}")
        sst = xr.open_dataset(sst_path)["sst"].fillna(0)
        enso = xr.open_dataset(enso_path)["sst"].fillna(0).squeeze()
        sst = sst.loc[:, config.lat_start:config.lat_end, config.lon_start:config.lon_end]
        datasets.append(ENSOData_load([sst], enso, len(enso), config.input_length, config.leading_time))
    print(f"CMIP6 models ({len(names)}): {', '.join(names)}")
    return datasets


def load_cmip_pretraining_dataset(config=par):
    """Combine the configured CMIP6 SST datasets for pre-training."""
    return ConcatDataset(load_cmip_datasets(config))

