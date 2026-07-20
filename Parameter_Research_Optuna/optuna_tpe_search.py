"""Optuna parameter search for the HamCNN ENSO forecast model."""

from pathlib import Path
import json
import time
import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from skorch import NeuralNet
from skorch.helper import SliceDataset
from torch.utils.data import ConcatDataset

import data_utils as du
import read_data
from models import HamCNN, count_trainable_parameters


# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------
# Update SEARCH_VERSION whenever the scoring method, search method, CMIP6 settings, or search space changes.
SEARCH_VERSION = "v1_tpe_obs_all_18_mse_10miroc6"
SEARCH_METHOD = "tpe"  # "tpe" or "grid"
# False: only train on observations. True: add selected CMIP6 members.
MAIN_SCORE_METHOD = "all_18_mse"  # "first_6_mse", "all_18_mse", "weighted_18_mse", "all_pearson_cor"
RANDOM_SEED = 0
# The first TPE_STARTUP_TRIALS trials use random exploration; TPE then samples from a model fitted to completed trials.
N_TRIALS = 10
TPE_STARTUP_TRIALS = 5
TPE_MULTIVARIATE = True

INCLUDE_CMIP6_TRAINING = True
RESUME_EXISTING_STUDY = False

TEST_START_YEAR = 1981
TEST_LENGTH_YEARS = 30
INPUT_LENGTH = 6
LEADING_TIME = 18
INPUT_VARIABLES = ["sst"]
EXPECTED_FIELD_SHAPE = (49, 144) # 17 57 49 144

LON_START = 0
LON_END = 357.5
LAT_START = -60
LAT_END = 60
TIME_START = "1871"
TIME_END = "2025"

# E:/OneDrive - University of Leeds/A-Research/Study_timeseies/Data
# /home/users/jwu028/study/Data
DATA_ROOT = Path("E:/OneDrive - University of Leeds/A-Research/Study_timeseies/Data") 
OBS_SST_PATH = DATA_ROOT / "Obs/SST/SST_NOAA_1871-2025.nc"
OBS_ENSO_PATH = DATA_ROOT / "Obs/ENSO/nino34_NOAA_1871-2025.nc"
OBS_OHC_PATH = DATA_ROOT / "Obs/OHC/ohc300_SODA_ORAS5_1871-2025.nc"
CMIP6_ROOT = DATA_ROOT / "CMIP6"
CMIP6_SST_DIR = CMIP6_ROOT / "SST"
CMIP6_ENSO_DIR = CMIP6_ROOT / "ENSO"
# "auto" uses every CMIP6 member for which both SST and ENSO files are available.
CMIP6_MEMBERS = (
    "MIROC6_hist_r1", "MIROC6_hist_r2",
    "MIROC6_hist_r3", "MIROC6_hist_r4",
    "MIROC6_hist_r5", "MIROC6_hist_r6",
    "MIROC6_hist_r7", "MIROC6_hist_r8",
    "MIROC6_hist_r9", "MIROC6_hist_r10",
                 )


# -----------------------------------------------------------------------------
# Search space and outputs
# -----------------------------------------------------------------------------
# Do not change the search space within the same SEARCH_VERSION, as this may cause conflicting Optuna distributions.
TPE_SEARCH_SPACE = {
    "lr": (1e-4, 2e-3),
    "batch_size": [128, 256, 512, 1024],
    "max_epochs": [10, 15, 20, 25, 30],
    "optimizer": ["Adam", "AdamW"],
    "M_1": [16, 32, 64],
    "M_2": [16, 32, 64],
    "M_3": [16, 32, 64],
    "N_Num": [16, 32, 64, 128, 256],
}
GRID_SEARCH_SPACE = {
    "lr": [0.0005, 0.001, 0.0015, 0.002, 0.0025],
    "batch_size": [128, 256, 512, 1024],
    "max_epochs": [10, 15, 20, 25, 30],
    "optimizer": ["Adam", "AdamW"],
    "M_1": [16, 32, 64],
    "M_2": [16, 32, 64],
    "M_3": [16, 32, 64],
    "N_Num": [16, 32, 64, 128, 256],
}
SEARCH_SPACE = GRID_SEARCH_SPACE if SEARCH_METHOD == "grid" else TPE_SEARCH_SPACE
OPTIMIZERS = {"Adam": optim.Adam, "AdamW": optim.AdamW}
# Lead weights for weighted_18_mse: the first six months receive greater weight.
LEAD_WEIGHTS = np.array([2.0] * 6 + [1.0] * 6 + [0.5] * 6)

RUN_DIR = Path("optuna_runs") / SEARCH_VERSION
# The Optuna database stores trial history, parameter distributions, and diagnostics.
STUDY_DB = RUN_DIR / "optuna_study.db"
BEST_PARAMS_JSON = RUN_DIR / "best_params.json"
TRIAL_SUMMARY_CSV = RUN_DIR / "trial_summary.csv"


def calculate_metrics(y_true, y_pred):
    """Return the main score plus lead-wise MSE and correlation."""
    mse_by_lead = ((y_true - y_pred) ** 2).mean(axis=0)
    corr_by_lead = []
    for lead in range(LEADING_TIME):
        true_lead = y_true[:, lead]
        pred_lead = y_pred[:, lead]
        if np.std(true_lead) == 0 or np.std(pred_lead) == 0:
            corr_by_lead.append(np.nan)
        else:
            corr_by_lead.append(float(np.corrcoef(true_lead, pred_lead)[0, 1]))

    true_all = y_true.ravel()
    pred_all = y_pred.ravel()
    if np.std(true_all) == 0 or np.std(pred_all) == 0:
        all_pearson_cor = np.nan
    else:
        all_pearson_cor = float(np.corrcoef(true_all, pred_all)[0, 1])

    scores = {
        # Optuna maximizes the objective; because lower MSE is better, use its negative as the score.
        "first_6_mse": -float(mean_squared_error(y_true[:, :6], y_pred[:, :6])),
        "all_18_mse": -float(mean_squared_error(y_true, y_pred)),
        "weighted_18_mse": -float(np.average(mse_by_lead, weights=LEAD_WEIGHTS)),
        "all_pearson_cor": all_pearson_cor,
    }
    return {
        "score": scores[MAIN_SCORE_METHOD],
        "all_pearson_cor": all_pearson_cor,
        "mse_by_lead": mse_by_lead,
        "corr_by_lead": np.array(corr_by_lead),
    }


def suggest_params(trial):
    """Suggest one parameter set from the active search space."""
    if SEARCH_METHOD == "grid":
        return {
            name: trial.suggest_categorical(name, choices)
            for name, choices in SEARCH_SPACE.items()
        }

    lr_low, lr_high = SEARCH_SPACE["lr"]
    params = {"lr": trial.suggest_float("lr", lr_low, lr_high, log=True)}
    for name, choices in SEARCH_SPACE.items():
        if name != "lr":
            params[name] = trial.suggest_categorical(name, choices)
    return params


def select_cmip6_members():
    """Select CMIP6 members that have both SST and ENSO files."""
    sst_members = {
        path.stem.removeprefix("sst_")
        for path in CMIP6_SST_DIR.glob("sst_*.nc")
    }
    enso_members = {
        path.stem.removeprefix("nino34_")
        for path in CMIP6_ENSO_DIR.glob("nino34_*.nc")
    }

    missing_enso = sorted(sst_members - enso_members)
    missing_sst = sorted(enso_members - sst_members)
    if missing_enso or missing_sst:
        raise FileNotFoundError(
            f"CMIP6 SST/ENSO files do not match. Missing ENSO: {missing_enso}; "
            f"missing SST: {missing_sst}"
        )

    detected_members = sorted(sst_members & enso_members)
    if CMIP6_MEMBERS == "auto":
        return detected_members

    selected_members = list(CMIP6_MEMBERS)
    missing_members = sorted(set(selected_members) - set(detected_members))
    if missing_members:
        raise ValueError(
            f"CMIP6 members not found or missing matched SST/ENSO files: {missing_members}. "
            f"Available members: {detected_members}"
        )
    return selected_members


def make_net(params):
    """Create the HamCNN skorch model for one trial."""
    return NeuralNet(
        module=HamCNN,
        module__input_channels=input_channels,
        module__leading_time=LEADING_TIME,
        module__M_1=params["M_1"],
        module__M_2=params["M_2"],
        module__M_3=params["M_3"],
        module__N_Num=params["N_Num"],
        criterion=torch.nn.MSELoss,
        optimizer=OPTIMIZERS[params["optimizer"]],
        max_epochs=params["max_epochs"],
        batch_size=params["batch_size"],
        lr=params["lr"],
        iterator_train__shuffle=True,
        train_split=None,
        device=device,
    )


def objective(trial):
    """Train and score one independent Optuna trial."""
    params = suggest_params(trial)
    start_time = time.perf_counter()
    du.set_random_seed(RANDOM_SEED)
    net = make_net(params)

    try:
        net.fit(train_x, train_y)
        metrics = calculate_metrics(valid_y, np.asarray(net.predict(valid_x)))
    finally:
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trial.set_user_attr("fit_and_score_time", time.perf_counter() - start_time)
    trial.set_user_attr("main_score_method", MAIN_SCORE_METHOD)
    trial.set_user_attr("search_method", SEARCH_METHOD)
    trial.set_user_attr("include_cmip6_training", INCLUDE_CMIP6_TRAINING)
    trial.set_user_attr("cmip6_members", ",".join(cmip6_members))
    trial.set_user_attr("train_samples", len(train_dataset))
    trial.set_user_attr("evaluation_samples", len(test_dataset))
    trial.set_user_attr("all_pearson_cor", None if np.isnan(metrics["all_pearson_cor"]) else float(metrics["all_pearson_cor"]))
    for lead in range(LEADING_TIME):
        trial.set_user_attr(f"mse_lead_{lead + 1:02d}", float(metrics["mse_by_lead"][lead]))
        corr = metrics["corr_by_lead"][lead]
        trial.set_user_attr(f"corr_lead_{lead + 1:02d}", None if np.isnan(corr) else float(corr))

    return metrics["score"]


def export_results(study):
    """Save the best parameters and per-trial diagnostics."""
    best = {
        "search_version": SEARCH_VERSION,
        "search_method": SEARCH_METHOD,
        "include_cmip6_training": INCLUDE_CMIP6_TRAINING,
        "cmip6_members": cmip6_members,
        "train_samples": len(train_dataset),
        "evaluation_samples": len(test_dataset),
        "main_score_method": MAIN_SCORE_METHOD,
        "best_trial": study.best_trial.number,
        "best_score": study.best_value,
        "best_params": study.best_params,
    }
    BEST_PARAMS_JSON.write_text(json.dumps(best, indent=2), encoding="utf-8")

    rows = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "score": trial.value,
            "state": trial.state.name,
            "search_method": trial.user_attrs.get("search_method"),
            "include_cmip6_training": trial.user_attrs.get("include_cmip6_training"),
            "cmip6_members": trial.user_attrs.get("cmip6_members"),
            "train_samples": trial.user_attrs.get("train_samples"),
            "evaluation_samples": trial.user_attrs.get("evaluation_samples"),
            "main_score_method": trial.user_attrs.get("main_score_method"),
            "fit_and_score_time": trial.user_attrs.get("fit_and_score_time"),
            "all_pearson_cor": trial.user_attrs.get("all_pearson_cor"),
        }
        row.update(trial.params)
        for lead in range(1, LEADING_TIME + 1):
            row[f"mse_lead_{lead:02d}"] = trial.user_attrs.get(f"mse_lead_{lead:02d}")
            row[f"corr_lead_{lead:02d}"] = trial.user_attrs.get(f"corr_lead_{lead:02d}")
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(by="score", ascending=False, na_position="last")
    summary.to_csv(TRIAL_SUMMARY_CSV, index=False, encoding="utf-8")


# -----------------------------------------------------------------------------
# Setup and data preparation
# -----------------------------------------------------------------------------
if SEARCH_METHOD not in {"tpe", "grid"}:
    raise ValueError('SEARCH_METHOD must be "tpe" or "grid".')
if MAIN_SCORE_METHOD not in {"first_6_mse", "all_18_mse", "weighted_18_mse", "all_pearson_cor"}:
    raise ValueError("MAIN_SCORE_METHOD is not valid.")
if len(LEAD_WEIGHTS) != LEADING_TIME:
    raise ValueError("LEAD_WEIGHTS must have one value per lead month.")
if INCLUDE_CMIP6_TRAINING and INPUT_VARIABLES != ["sst"]:
    raise ValueError('CMIP6 training currently supports INPUT_VARIABLES = ["sst"] only.')
if STUDY_DB.exists() and not RESUME_EXISTING_STUDY:
    # Resume an existing database only when both the search space and scoring method are unchanged.
    raise FileExistsError(
        f"{STUDY_DB} already exists. Change SEARCH_VERSION, delete the run directory, "
        "or set RESUME_EXISTING_STUDY = True only if the search space is unchanged."
    )

RUN_DIR.mkdir(parents=True, exist_ok=True)
du.set_random_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_channels = INPUT_LENGTH * len(INPUT_VARIABLES)

print("Device:", device)
print("Search version:", SEARCH_VERSION)
print("Search method:", SEARCH_METHOD)
print("Main score method:", MAIN_SCORE_METHOD)
print("Include CMIP6 training:", INCLUDE_CMIP6_TRAINING)
print("Run directory:", RUN_DIR)

obs_data = read_data.read_observation_data(
    obs_sst_path=OBS_SST_PATH,
    obs_enso_path=OBS_ENSO_PATH,
    obs_ohc_path=OBS_OHC_PATH,
    lat_range=(LAT_START, LAT_END),
    lon_range=(LON_START, LON_END),
    time_range=(TIME_START, TIME_END),
    include_sst="sst" in INPUT_VARIABLES,
    include_ohc="ohc" in INPUT_VARIABLES,
)

time_splits = du.build_time_splits(
    test_start_year=TEST_START_YEAR,
    input_length=INPUT_LENGTH,
    leading_time=LEADING_TIME,
    test_length_years=TEST_LENGTH_YEARS,
)
print("Train periods:", time_splits["train_periods"])
print("Test period:", time_splits["test_period"])

train_datasets = []
for train_start, train_end in time_splits["train_periods"]:
    train_cut = du.cut_train_data(
        obs_data,
        train_start,
        train_end,
        input_variables=INPUT_VARIABLES,
        dataset_name="NOAA_SODA_ORAS5",
    )
    train_datasets.append(du.make_dataset(train_cut, INPUT_VARIABLES, INPUT_LENGTH, LEADING_TIME))

if INCLUDE_CMIP6_TRAINING:
    cmip6_members = select_cmip6_members()
    print("CMIP6 members:", cmip6_members)
    cmip6_data = read_data.read_cmip6_dataset(
        member_names=cmip6_members,
        cmip6_sst_dir=CMIP6_SST_DIR,
        cmip6_enso_dir=CMIP6_ENSO_DIR,
        lat_range=(LAT_START, LAT_END),
        lon_range=(LON_START, LON_END),
    )
    for member_name, member_data in cmip6_data.items():
        train_datasets.append(
            du.make_dataset(
                member_data,
                input_variables=INPUT_VARIABLES,
                input_length=INPUT_LENGTH,
                leading_time=LEADING_TIME,
            )
        )
        print(f"Train: CMIP6 {member_name}")
else:
    cmip6_members = []

test_start, test_end = time_splits["test_period"]
test_cut = du.cut_test_data(
    obs_data,
    test_start,
    test_end,
    input_variables=INPUT_VARIABLES,
    dataset_name="NOAA_SODA_ORAS5",
)
test_dataset = du.make_dataset(test_cut, INPUT_VARIABLES, INPUT_LENGTH, LEADING_TIME)

train_dataset = ConcatDataset(train_datasets)
train_x = SliceDataset(train_dataset, idx=0)
train_y = SliceDataset(train_dataset, idx=1)
valid_x = SliceDataset(test_dataset, idx=0)
valid_y = np.stack([test_dataset[index][1].numpy() for index in range(len(test_dataset))])

sample_x, sample_y = train_dataset[0]
expected_x_shape = (input_channels, *EXPECTED_FIELD_SHAPE)
print("Train samples:", len(train_dataset))
print("Evaluation samples:", len(test_dataset))
print("X shape:", tuple(sample_x.shape))
print("y shape:", tuple(sample_y.shape))
assert tuple(sample_x.shape) == expected_x_shape
assert tuple(sample_y.shape) == (LEADING_TIME,)

preview_params = {name: values[0] for name, values in SEARCH_SPACE.items()}
model_preview = HamCNN(
    input_channels=input_channels,
    leading_time=LEADING_TIME,
    M_1=preview_params["M_1"],
    M_2=preview_params["M_2"],
    M_3=preview_params["M_3"],
    N_Num=preview_params["N_Num"],
)
print("Preview trainable parameters:", count_trainable_parameters(model_preview))


# -----------------------------------------------------------------------------
# Optuna search and exports
# -----------------------------------------------------------------------------
if SEARCH_METHOD == "grid":
    sampler = optuna.samplers.GridSampler(GRID_SEARCH_SPACE, seed=RANDOM_SEED)
    n_trials = 1
    for values in GRID_SEARCH_SPACE.values():
        n_trials *= len(values)
else:
    # Startup trials use random exploration; TPE then samples from a model fitted to completed trials.
    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_SEED,
        n_startup_trials=TPE_STARTUP_TRIALS,
        multivariate=TPE_MULTIVARIATE,
    )
    n_trials = N_TRIALS

study = optuna.create_study(
    study_name=f"hamcnn_{SEARCH_VERSION}",
    storage=f"sqlite:///{STUDY_DB.as_posix()}",
    load_if_exists=RESUME_EXISTING_STUDY,
    direction="maximize",
    sampler=sampler,
)
study.optimize(objective, n_trials=n_trials)
export_results(study)

print("Best score:", study.best_value)
print("Best parameters:", study.best_params)
print("Saved:", TRIAL_SUMMARY_CSV)
print("Saved:", BEST_PARAMS_JSON)
print("Study database:", STUDY_DB)
