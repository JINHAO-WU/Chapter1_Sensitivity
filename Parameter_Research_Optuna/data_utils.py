"""Dataset preparation and scoring utilities for ENSO parameter search."""

import random

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset


def set_random_seed(seed):
    """Set reproducible seeds when seed is an integer."""
    if seed is None:
        print("Random seed is disabled.")
        return

    print(f"Random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_time_splits(
    test_start_year,
    input_length,
    leading_time,
    test_length_years,
):
    """Build train/test periods for a 30-year independent test window."""
    test_start = int(test_start_year)
    test_end = test_start + test_length_years - 1

    start_year, start_month = _first_target_time(test_start, input_length)
    end_year, end_month = _last_target_time(test_end, leading_time)

    if 1871 <= test_start <= 1873:
        train_1_start, train_1_end = test_start + test_length_years, 1900
        train_2_start, train_2_end = 1901, 2025
    elif 1994 <= test_start <= 1996:
        train_1_start, train_1_end = 1871, 1900
        train_2_start, train_2_end = 1901, test_start - 1
    else:
        train_1_start, train_1_end = 1871, test_start - 1
        train_2_start, train_2_end = test_end + 1, 2025

    return {
        "train_periods": [
            (str(train_1_start), str(train_1_end)),
            (str(train_2_start), str(train_2_end)),
        ],
        "test_period": (str(test_start), str(test_end)),
        "target_time_range": [f"{start_year}-{start_month}-01", f"{end_year}-{end_month}-01"],
        "train_label": f"{train_1_start} - {train_1_end}_&_{train_2_start} - {train_2_end}",
        "test_label": f"{test_start} - {test_end}",
    }


def cut_train_data(data, train_start, train_end, input_variables, dataset_name):
    """Cut one training period from a data dictionary."""
    result = _cut_data(data, train_start, train_end, input_variables)
    print(f"Train: {dataset_name}  (Time: {train_start} - {train_end})")
    return result


def cut_test_data(data, test_start, test_end, input_variables, dataset_name):
    """Cut one test period from a data dictionary."""
    result = _cut_data(data, test_start, test_end, input_variables)
    print(f"Test: {dataset_name}  (Time: {test_start} - {test_end})")
    print("-" * 55)
    return result


class ENSODataset(Dataset):
    """Sliding-window ENSO dataset with one or more gridded input variables."""

    def __init__(self, variables, enso, input_length, leading_time):
        super().__init__()
        self.variables = [np.asarray(variable) for variable in variables]
        self.enso = np.asarray(enso)
        self.input_length = input_length
        self.leading_time = leading_time

    def __getitem__(self, index):
        data_x = np.concatenate(
            [variable[index : index + self.input_length] for variable in self.variables],
            axis=0,
        )
        data_y = self.enso[index + self.input_length : index + self.input_length + self.leading_time]
        return torch.tensor(data_x, dtype=torch.float32), torch.tensor(data_y, dtype=torch.float32)

    def __len__(self):
        return len(self.enso) - self.leading_time - self.input_length


def make_dataset(cut_data, input_variables, input_length, leading_time):
    """Create an ENSODataset from a cut data dictionary."""
    variables = [cut_data[name] for name in input_variables]
    return ENSODataset(
        variables=variables,
        enso=cut_data["enso"],
        input_length=input_length,
        leading_time=leading_time,
    )


def negative_mse_first_6_months(estimator, x, y):
    """Score predictions with negative MSE over the first six lead months."""
    y_pred = np.asarray(estimator.predict(x))
    y_true = np.asarray(y)
    return -mean_squared_error(y_true[:, :6], y_pred[:, :6])


def _cut_data(data, start, end, input_variables):
    cut = {"enso": data["enso"].loc[start:end]}
    for variable_name in input_variables:
        cut[variable_name] = data[variable_name].loc[start:end, :, :]
    return cut


def _first_target_time(test_start, input_length):
    start_month_by_input = {3: 4, 6: 7, 12: 1}
    if input_length == 12:
        return test_start + 1, start_month_by_input[input_length]
    return test_start, start_month_by_input.get(input_length, input_length + 1)


def _last_target_time(test_end, leading_time):
    if leading_time == 18:
        return test_end - 1, 6
    if leading_time == 12:
        return test_end - 1, 12
    if leading_time == 6:
        return test_end, 6
    end_offset_years = leading_time // 12
    end_month = 12 - leading_time % 12
    return test_end - end_offset_years, end_month
