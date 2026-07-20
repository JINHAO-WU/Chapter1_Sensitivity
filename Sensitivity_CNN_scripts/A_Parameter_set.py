"""Central configuration for ENSO training experiments."""

from pathlib import Path

# Experiment settings
method = "input6_var1_sst_NOAA_10CMIP6"
seeds = 0
pre_epoch = 10
epoch = 10
train_lr = 0.00038
train_batch_size = 1024
pre_process = "Remove linear trend and yearmonthly std 1871-2025 climatology"

# Select any registered variables in this order.  Add future variables only to
# OBSERVATION_VARIABLES; the loading and training code needs no new branch.
input_variables = ("sst")
# A single string such as ("ohc") is normalised to a one-item tuple.
if isinstance(input_variables, str):
    input_variables = (input_variables,)

# /home/users/jwu028/study/Data/Obs/SST/SST_NOAA_1871-2025.nc
OBSERVATION_VARIABLES = {
    "sst": {
        "path": "E:/OneDrive - University of Leeds/A-Research/Study_timeseies/Data/Obs/SST/SST_NOAA_1871-2025.nc",
        "field": "sst",
        "label": "NOAA SST",
    },
    "ohc": {
        "path": "E:/OneDrive - University of Leeds/A-Research/Study_timeseies/Data/Obs/OHC/ohc300_SODA_ORAS5_1871-2025.nc",
        "field": "ohc300",
        "label": "SODA/ORAS5 OHC300",
    },
}
ENSO_PATH = "E:/OneDrive - University of Leeds/A-Research/Study_timeseies/Data/Obs/ENSO/nino34_NOAA_1871-2025.nc"
ENSO_FIELD = "sst"

# These derived names remain available to older modules, but are never edited
# manually.  Each selected variable contributes input_length channels.
variable = "+".join(input_variables)
variable_tag = "_".join(input_variables)
variable_label = " + ".join(OBSERVATION_VARIABLES[name]["label"] for name in input_variables)
var = len(input_variables)
input_length = 6
leading_time = 18
pa = var * input_length

# Define HamCNN parameters once.  Both construction and displayed name use it.
net = "HamCNN"
network_kwargs = {"M_1": 32, "M_2": 16, "M_3": 64, "N_Num": 512}
net_name = f"{net}({network_kwargs['M_1']}, {network_kwargs['M_2']}, {network_kwargs['M_3']}, {network_kwargs['N_Num']})"
network_tag = f"{net}_{network_kwargs['M_1']}_{network_kwargs['M_2']}_{network_kwargs['M_3']}_{network_kwargs['N_Num']}"

# Shared output locations for pre-training weights, training results and plots.
SCRIPT_DIR = Path(__file__).resolve().parent
pretrain_dir = SCRIPT_DIR / "pre_pickle"
output_dir = SCRIPT_DIR / "pickle" / f"pickle_{net}_{method}"
graph_dir = SCRIPT_DIR / "graph" / f"graph_{net}_{method}"

# Spatial and temporal domain
lon_start, lon_end = 0, 357.5 # 160, 300 0, 357.5
lat_start, lat_end = -60, 60  # -20, 20 -60, 60
time_start, time_end = "1871", "2025"
cmip_time = "Yes"
# ``"ALL"`` discovers every SST/ENSO NetCDF pair under CMIP_ROOT.
# A string or tuple is accepted, for example ``"ALL"`` or ("CESM2_hist_r1",).
CMIP_ROOT = "E:/OneDrive - University of Leeds/A-Research/Study_timeseies/Data/CMIP6"
CMIP_MODELS = (
    "MIROC6_hist_r1",
    "MIROC6_hist_r2",
    "MIROC6_hist_r3",
    "MIROC6_hist_r4",
    "MIROC6_hist_r5",
    "MIROC6_hist_r6",
    "MIROC6_hist_r7",
    "MIROC6_hist_r8",
    "MIROC6_hist_r9",
    "MIROC6_hist_r10"
    )

def Cut_Data_Parameter(test_start_time: int, input_length: int, leading_time: int, times="None"):
    """Define non-overlapping train/test periods for a 30-year test window."""
    test_start, test_end = test_start_time, test_start_time + 29

    if input_length == 12:
        start_year, start_month = test_start + 1, 1
    elif input_length == 6:
        start_year, start_month = test_start, 7
    elif input_length == 3:
        start_year, start_month = test_start, 4
    else:
        raise ValueError("input_length must be one of 3, 6, or 12")

    if leading_time == 18:
        end_year, end_month = test_end - 1, 6
    elif leading_time == 12:
        end_year, end_month = test_end - 1, 12
    elif leading_time == 6:
        end_year, end_month = test_end, 6
    else:
        raise ValueError("leading_time must be one of 6, 12, or 18")

    if 1871 <= test_start <= 1873:
        train_1_start, train_1_end = test_start + 30, 1900
        train_2_start, train_2_end = 1901, 2025
    elif 1994 <= test_start <= 1996:
        train_1_start, train_1_end = 1871, 1900
        train_2_start, train_2_end = 1901, test_start - 1
    else:
        train_1_start, train_1_end = 1871, test_start - 1
        train_2_start, train_2_end = test_end + 1, 2025

    # Keep output names concise; `method` remains the user-controlled experiment label.
    run_name = f"{net}_{test_start}_{method}" if times == "None" else f"{net}_{times}"
    return {
        "train_1_start": str(train_1_start), "train_1_end": str(train_1_end),
        "train_2_start": str(train_2_start), "train_2_end": str(train_2_end),
        "test_start": str(test_start), "test_end": str(test_end),
        "time_list": [f"{start_year}-{start_month}-01", f"{end_year}-{end_month}-01"],
        "train_time": f"{train_1_start} - {train_1_end}_&_{train_2_start} - {train_2_end}",
        "test_time": f"{test_start} - {test_end}",
        "name_path": run_name,
    }


def create_network():
    """Create a fresh model; do not share weights between batch years."""
    if net != "HamCNN":
        raise ValueError(f"Unsupported network: {net}")
    from Model_HamCNN import HamCNN
    return HamCNN(input_channels=pa, output_steps=leading_time, **network_kwargs)
