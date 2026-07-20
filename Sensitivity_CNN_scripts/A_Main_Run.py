"""Single and batch ENSO experiment entry points."""

from collections.abc import Iterable
import random

import numpy as np
import torch
import xarray as xr
from torch.utils.data import ConcatDataset, DataLoader

import A_Parameter_set as par
from data import ENSOData_load, cut_data, load_cmip_datasets, load_cmip_pretraining_dataset, load_observations
from train_function import pretrain_func, train_func
from Z_plt_cut import plot_cut

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def run_experiment(year: int, *, pretrain_name="no", pre_train=False, pre_pth=False,
                   combine_cmip6=False, only_cmip6=False, times="None"):
    """Run one 30-year hindcast experiment using the configured variables."""
    seed = par.seeds
    if seed not in (None, "None"):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        generator = torch.Generator().manual_seed(seed)

        def seed_worker(_):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        loader_seed_kwargs = {"generator": generator, "worker_init_fn": seed_worker}
    else:
        loader_seed_kwargs = {}

    variables, enso = load_observations(par)
    observation_label = par.variable_label
    cmip_allowed = par.input_variables == ("sst",)
    use_cmip = combine_cmip6 and cmip_allowed
    if combine_cmip6 and not cmip_allowed:
        print("CMIP6 disabled: only SST is available for the current CMIP6 data.")

    print("=" * 55)
    print(f"Test first year : {year}\nNetwork : {par.net_name}\nVariables : {observation_label}\nInput channels : {par.pa}")
    print(f"Train epochs : {par.epoch}\nLearning rate : {par.train_lr}\nBatch size : {par.train_batch_size}")
    print("=" * 55)

    split = par.Cut_Data_Parameter(year, par.input_length, par.leading_time, times)
    periods = [
        cut_data(variables, enso, observation_label, split["train_1_start"], split["train_1_end"], "Train"),
        cut_data(variables, enso, observation_label, split["train_2_start"], split["train_2_end"], "Train"),
    ]
    if 1871 <= year <= 1873 or 1994 <= year <= 1996:
        train_variables = {name: xr.concat([period["variables"][name] for period in periods], dim="time") for name in par.input_variables}
        train_enso = xr.concat([period["enso"] for period in periods], dim="time")
        train_dataset = ENSOData_load(list(train_variables.values()), train_enso, len(train_enso), par.input_length, par.leading_time)
    else:
        train_dataset = ConcatDataset(tuple(
            ENSOData_load(list(period["variables"].values()), period["enso"], period["data_time_len"], par.input_length, par.leading_time)
            for period in periods
        ))

    if use_cmip:
        print("-" * 55)
        cmip_datasets = load_cmip_datasets(par)
        if only_cmip6:
            print(f"Only CMIP6 data for training.")
            train_dataset = ConcatDataset(tuple(cmip_datasets))
            train_label = f"10CMIP6 SST"
        else:
            print(f"Combining CMIP6 data with observations for training.")
            train_dataset = ConcatDataset((*cmip_datasets, train_dataset))
            train_label = f"10CMIP6 SST + {observation_label}"
        print("-" * 55)
    else:
        train_label = observation_label

    test_period = cut_data(variables, enso, observation_label, split["test_start"], split["test_end"], "Test")
    test_dataset = ENSOData_load(list(test_period["variables"].values()), test_period["enso"], test_period["data_time_len"], par.input_length, par.leading_time)
    train_loader = DataLoader(train_dataset, batch_size=par.train_batch_size, shuffle=True, **loader_seed_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    test_time_index = test_period["enso"].time.values[par.input_length:-par.leading_time]

    network = par.create_network()
    transfer_label = "None"
    if pre_train:
        print("-" * 55)
        print(f"START CMIP6 PRE-TRAINING: {pretrain_name}")
        if not cmip_allowed:
            raise ValueError("CMIP6 pre-training requires SST-only input")
        pre_loader = DataLoader(load_cmip_pretraining_dataset(par), batch_size=par.train_batch_size, shuffle=False, **loader_seed_kwargs)
        print("-" * 55)
        pretrain_func(network, pre_loader, par.pre_epoch, torch.optim.Adam(network.parameters(), lr=par.train_lr), pretrain_name, test=False, log=False)
        print(f"END CMIP6 PRE-TRAINING: {pretrain_name}")
        transfer_label = "CMIP6 pre-training"
    elif pre_pth:
        transfer_label = "Existing CMIP6 weights"

    title_text = (
        f"Net={par.net_name}, epoch={par.epoch}, seeds={par.seeds}\n"
        f"input_length={par.input_length}, train_lr={par.train_lr}, train_batch_size={par.train_batch_size}\n"
        f"Variables={observation_label}, Region: lon({par.lon_start}, {par.lon_end}) lat({par.lat_start}, {par.lat_end})\n"
        f"Train_data={train_label}, Test_data={observation_label}, TL_data={transfer_label}\n"
        f"train_time={split['train_time']}, test_time={split['test_time']}"
    )
    print("START FINE-TUNING / TRAIN")
    results = train_func(network, train_loader, par.pretrain_dir / f"{pretrain_name}.pth", pre_pth,
                         par.epoch, torch.optim.Adam(network.parameters(), lr=par.train_lr), split["name_path"],
                         test_loader, test_time_index, criterion=torch.nn.MSELoss(), 
                         save_pth=False, test=True, log=True, )
    print("END FINE-TUNING / TRAIN")
    plot_cut(test_time_index, par.leading_time, split["name_path"], par.net_name, title_text)
    return results


def run_years(years: Iterable[int], **experiment_options):
    """Run independent experiments for all requested test start years."""
    return {year: run_experiment(year, **experiment_options, pretrain_name="10CMIP6_TL", pre_train=True, pre_pth=False,
                   combine_cmip6=False, only_cmip6=False, times="None") for year in years}


run_years([1981])
#   run_years(range(1871, 1997))
