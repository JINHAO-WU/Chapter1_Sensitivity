"""Plot saved ENSO forecast skill using title information from the current run."""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import xarray as xr

import A_Parameter_set as par

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_cut(test_time_index, leading_time, graph_path, network_name, title_text):
    """Create seasonal skill, lead-time skill and forecast comparison panels.

    ``network_name`` and ``title_text`` are supplied by A_Main_Run for
    this exact experiment, avoiding stale labels from module-level settings.
    """
    lead_labels = [str(month) for month in range(1, leading_time + 1)]
    # Use the actual prediction timestamps instead of an assumed 30-year window.
    time_start = pd.DatetimeIndex(test_time_index)
    results_path = par.output_dir / f"{graph_path}.pickle"
    with results_path.open("rb") as file:
        results = pickle.load(file)

    pred_value = results["predict_value"].T
    real_value = results["real_value"].T
    pearson = results["Pearson"]
    acc_values = results["Acc"]

    seasonal_correlation = np.full((12, leading_time), np.nan)
    time = time_start
    for lead in range(leading_time):
        pred = xr.DataArray(pred_value[lead], dims="time", coords={"time": time})
        real = xr.DataArray(real_value[lead], dims="time", coords={"time": time})
        for month in range(1, 13):
            pred_month = pred.where(pred.time.dt.month == month, drop=True)
            real_month = real.where(real.time.dt.month == month, drop=True)
            if pred_month.size > 1:
                seasonal_correlation[month - 1, lead] = scipy.stats.pearsonr(pred_month, real_month).statistic
        time = time + pd.DateOffset(months=1)

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(311)
    skills = pd.DataFrame(seasonal_correlation, index=MONTH_NAMES, columns=lead_labels)
    sns.heatmap(skills, annot=False, cmap=sns.color_palette("coolwarm", as_cmap=True),
                linewidths=0.5, cbar_kws={"label": "Pearson correlation"}, vmin=0, vmax=1, ax=ax1)
    low_skill = np.ma.masked_less(skills.values, 0.5)
    ax1.pcolor(lead_labels, MONTH_NAMES, low_skill, hatch="//", alpha=0)
    ax1.set_xlabel("Forecast lead (months)")
    ax1.set_ylabel("Month")
    ax1.set_title(title_text, loc="left", fontsize=10)

    ax2 = fig.add_subplot(312)
    leads = np.arange(1, leading_time + 1)
    ax2.plot(leads, acc_values, label="ACC", linewidth=1.5, color="#1f77b4")
    ax2.plot(leads, pearson, "-o", label=network_name, color="#d62728", markersize=3)
    ax2.axhline(0.5, linestyle="dotted", label="0.5")
    ax2.set(xlim=(0.5, leading_time + 0.5), ylim=(0, 1), xlabel="Forecast lead (months)", ylabel="Correlation skill")
    ax2.set_xticks(leads)
    ax2.legend(frameon=False)

    ax3 = fig.add_subplot(313)
    lead_indices = [0, min(2, leading_time - 1), min(5, leading_time - 1)]
    colors = ["#91d73f", "#2cb47d", "#984ca3"]
    labels = [f"lead {index + 1}" for index in lead_indices]
    for index, color, label in zip(lead_indices, colors, labels):
        lead_time_values = time_start + pd.DateOffset(months=index)
        ax3.plot(lead_time_values, pred_value[index], label=label, color=color)
    ax3.plot(time_start, real_value[0], label="Observed", color="black")
    ax3.axhline(0, linestyle="dotted", color="black")
    ax3.set(xlabel="Time", ylabel="Nino3.4 index")
    ax3.legend(loc="upper right")

    graph_dir = par.graph_dir
    graph_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(graph_dir / f"{graph_path}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Finish plot")




