"""Training, evaluation and result-serialisation helpers."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import scipy.stats
import torch
from torch import nn
from sklearn.metrics import r2_score

import A_Parameter_set as par


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CorrelationLoss(nn.Module):
    """One minus mean Pearson correlation across forecast lead columns."""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        predictions = predictions - predictions.mean(dim=0, keepdim=True)
        targets = targets - targets.mean(dim=0, keepdim=True)
        denominator = torch.sqrt(predictions.square().sum(0) * targets.square().sum(0)).clamp_min(self.epsilon)
        return 1 - ((predictions * targets).sum(0) / denominator).mean()


class EarlyStopping:
    """Stop when a monitored loss has not improved for ``patience`` epochs."""

    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, value):
        if self.best_score is None or value < self.best_score - self.delta:
            self.best_score, self.counter = value, 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience
        return self.early_stop


def remove_climatology(pred_data, real_data, time_index):
    """Remove the monthly climatology from prediction and observation series."""
    time_index = pd.to_datetime(time_index)

    def anomaly(values):
        frame = pd.DataFrame({"value": values, "month": time_index.month})
        return (frame["value"] - frame.groupby("month")["value"].transform("mean")).to_numpy(np.float32)

    return anomaly(pred_data), anomaly(real_data)


def test_func(network, testdata, test_time_index, leading_time, criterion=None, device=DEFAULT_DEVICE):
    """Evaluate ``network`` and calculate skill metrics for every lead month."""
    network.to(device).eval()
    predictions, targets = [], []
    with torch.no_grad():
        for inputs, real_outputs in testdata:
            predictions.append(network(inputs.to(device)).cpu().numpy())
            targets.append(real_outputs.cpu().numpy())

    pred = np.concatenate(predictions)
    real = np.concatenate(targets)
    correlations, r2_scores, pred_anomalies, real_anomalies, acc_values = [], [], [], [], []
    for lead in range(leading_time):
        correlation, _ = scipy.stats.pearsonr(pred[:, lead], real[:, lead])
        forecast_time = pd.to_datetime(test_time_index) + pd.DateOffset(months=lead)
        pred_anomaly, real_anomaly = remove_climatology(pred[:, lead], real[:, lead], forecast_time)
        denominator = np.linalg.norm(pred_anomaly) * np.linalg.norm(real_anomaly)
        acc = float(np.dot(pred_anomaly, real_anomaly) / denominator) if denominator else 0.0
        correlations.append(correlation)
        r2_scores.append(r2_score(real[:, lead], pred[:, lead]))
        pred_anomalies.append(pred_anomaly)
        real_anomalies.append(real_anomaly)
        acc_values.append(acc)
    return pred, real, correlations, r2_scores, pred_anomalies, real_anomalies, np.asarray(acc_values)


def _run_train_epoch(network, train_data, optimizer, criterion, device, epoch, stage="train"):
    network.train()
    losses = []
    for batch_index, (inputs, targets) in enumerate(train_data):
        optimizer.zero_grad()
        loss = criterion(network(inputs.to(device)), targets.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_index % 10 == 0 and epoch % 2 == 0:
            print(f"[{stage}] epoch={epoch}, batch={batch_index}, loss={loss.item():.6f}")
    return float(np.mean(losses))


def _save_results(pred, real, correlations, pred_anomaly, real_anomaly, acc, epoch, save_path):
    payload = {
        "predict_value": pred, "real_value": real, "Pearson": correlations,
        "pred_anomaly": pred_anomaly, "real_anomaly": real_anomaly,
        "Acc": acc, "epoch": epoch,
    }
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(payload, file)


def pretrain_func(network, trainData, epochs, optim, savename, testData=None, test_time_index=None,
                  criterion=None, device=DEFAULT_DEVICE, save_pth=True, test=True, log=True):
    """Train a model on pre-training data, optionally evaluate and save it."""
    criterion = criterion or torch.nn.MSELoss()
    network.to(device)
    for epoch in range(epochs):
        _run_train_epoch(network, trainData, optim, criterion, device, epoch, stage="pretrain")

    output_dir = par.pretrain_dir
    if save_pth:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(network.state_dict(), output_dir / f"{savename}.pth")
    if test:
        if testData is None or test_time_index is None:
            raise ValueError("testData and test_time_index are required when test=True")
        results = test_func(network, testData, test_time_index, network(torch.zeros(1, *next(iter(trainData))[0].shape[1:], device=device)).shape[-1], device=device)
        if log:
            _save_results(results[0], results[1], results[2], results[4], results[5], results[6], epochs - 1, output_dir / f"{savename}.pickle")


def train_func(Network, trainData, pretrain_path, pre_pth, epochs, optim, saveName, testData, test_time_index,
               criterion=None, device=DEFAULT_DEVICE, save_pth=True, test=True, log=True, earlystop=False):
    """Fine-tune a network while retaining the original public API."""
    criterion = criterion or torch.nn.MSELoss()
    if pre_pth:
        Network.load_state_dict(torch.load(pretrain_path, map_location=device))
    Network.to(device)
    final_epoch = 0

    # ``criterion`` controls both optimisation and optional validation.  The
    # default is MSE, but callers may pass ``CorrelationLoss()`` from this module to
    # train and early-stop directly on forecast correlation instead.
    if earlystop:
        early_stopping = EarlyStopping(patience=10, delta=0.1)

    for epoch in range(epochs):
        _run_train_epoch(Network, trainData, optim, criterion, device, epoch, stage="train")
        final_epoch = epoch
        if earlystop:
            # Evaluate the same chosen loss without gradients.  Lower loss is
            # always better, regardless of whether MSE or CorrelationLoss was
            # supplied by the caller.
            Network.eval()
            values = []
            with torch.no_grad():
                for inputs, targets in testData:
                    values.append(criterion(Network(inputs.to(device)), targets.to(device)).item())
            if early_stopping(float(np.mean(values))):
                print(f"Early stopping at epoch {epoch + 1}")
                break

    output_dir = par.output_dir
    if save_pth:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(Network.state_dict(), output_dir / f"{saveName}.pth")
    if not test:
        return None

    output_steps = next(iter(testData))[1].shape[-1]
    results = test_func(Network, testData, test_time_index, output_steps, device=device)
    pred, real, correlations, r2_scores, pred_anomaly, real_anomaly, acc = results
    # Convert NumPy scalar metrics to Python floats for clean console output.
    print("-" * 55)
    print("Pearson correlation:", [round(float(value), 4) for value in correlations])
    print("ACC correlation:", [round(float(value), 4) for value in acc])
    print("R2 score:", [round(float(value), 4) for value in r2_scores])
    if log:
        _save_results(pred, real, correlations, pred_anomaly, real_anomaly, acc, final_epoch, output_dir / f"{saveName}.pickle")
    return results
