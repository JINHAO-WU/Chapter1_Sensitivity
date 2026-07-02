"""
Confusion-matrix and class-metric figures for ENSO event forecasts.

All metrics use the full sample; no leave-one-out event exclusion.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from A_basic_sources import FIGURE_ROOT, get_dl_sources, load_source_forecast_table
from plot_style import (
    DEFAULT_FIGURE_DPI,
    EVENT_COLORS,
    G_CLASSIFICATION_STYLE,
    add_shared_axis_labels,
    configure_publication_style,
    disable_axis_grid,
    figure_output_paths,
    mm_to_inches,
    save_publication_figure,
    style_boxed_axes,
    style_colorbar,
    style_open_axes,
    validate_data_sources,
)


BASE_YEAR = 1871
FIGURE_ID = "G"
FIGURE_NAME = "enso_classification_metrics"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
FIGURE_DPI = DEFAULT_FIGURE_DPI

LEADS = [6]
N_TYPE = 5
CLASS_METRIC = "f1"

DATA_SOURCES = get_dl_sources()

ENSO_CLASSIFIERS = {
    3: (
        ["EN", "N", "LN"],
        [(0.5, "EN"), (-0.5, "N")],
        "LN",
    ),
    5: (
        ["Strong EN", "Weak EN", "N", "Weak LN", "Strong LN"],
        [(1.5, "Strong EN"), (0.5, "Weak EN"), (-0.5, "N"), (-1.5, "Weak LN")],
        "Strong LN",
    ),
    7: (
        ["VStr EN", "Strong EN", "Weak EN", "N", "Weak LN", "Strong LN", "VStr LN"],
        [(2.0, "VStr EN"), (1.5, "Strong EN"), (0.5, "Weak EN"),
         (-0.5, "N"), (-1.5, "Weak LN"), (-2.0, "Strong LN")],
        "VStr LN",
    ),
}


def classify_nino(value, n_type):
    _, thresholds, fallback = ENSO_CLASSIFIERS[n_type]
    for threshold, label in thresholds:
        if value >= threshold:
            return label
    return fallback


def event_order(n_type):
    return ENSO_CLASSIFIERS[n_type][0]


def wrap_long_label(label, max_len=27):
    if len(label) <= max_len or "+" not in label:
        return label
    parts = label.split("+")
    wrapped = parts[0]
    for part in parts[1:]:
        sep = "+\n" if len(wrapped.split("\n")[-1]) + len(part) + 1 > max_len else "+"
        wrapped += sep + part
    return wrapped


def _class_score(confusion, metric):
    scores = []
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]
        actual = confusion[i, :].sum()
        predicted = confusion[:, i].sum()
        prec = tp / predicted if predicted else 0.0
        rec = tp / actual if actual else np.nan
        if metric == "precision":
            score = prec if actual else np.nan
        elif metric == "recall":
            score = rec
        elif actual == 0:
            score = np.nan
        elif prec + rec == 0:
            score = 0.0
        else:
            score = 2 * prec * rec / (prec + rec)
        scores.append(float(score) if not np.isnan(score) else np.nan)
    return scores


def compute_metrics(df, n_type, metric, leads):
    order = event_order(n_type)
    available = sorted(int(l) for l in df["leading"].unique())
    selected = available if leads is None else list(leads)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Requested leads {missing} not available.")
    results = {}
    for lead in selected:
        monthly = (
            df.loc[df["leading"] == lead]
            .groupby("abs_month", as_index=False)[["pred", "real"]]
            .mean()
            .dropna(subset=["pred", "real"])
        )
        monthly["real_class"] = monthly["real"].apply(lambda v: classify_nino(v, n_type))
        monthly["pred_class"] = monthly["pred"].apply(lambda v: classify_nino(v, n_type))
        raw = confusion_matrix(monthly["real_class"], monthly["pred_class"], labels=order)
        norm = confusion_matrix(monthly["real_class"], monthly["pred_class"], labels=order, normalize="true")
        results[lead] = {
            "confusion": np.nan_to_num(norm, nan=0.0),
            "scores": _class_score(raw, metric),
            "n_samples": len(monthly),
        }
    return results


def plot_figure(dataset_results, lead, order, output_base):
    fig = plt.figure(
        figsize=(
            mm_to_inches(G_CLASSIFICATION_STYLE["figure_width_mm"]),
            mm_to_inches(G_CLASSIFICATION_STYLE["figure_height_mm"]),
        ),
        facecolor="white",
    )
    outer = fig.add_gridspec(5, 2, left=0.100, right=0.985, bottom=0.064, top=0.985,
                             wspace=0.09, hspace=0.13)
    panels = [outer[r, c] for r in range(5) for c in range(2)]

    x_labels = [e.replace(" ", "\n") for e in order]
    y_labels = [e.replace(" ", "\n") for e in order]
    metric_label = {"recall": "Recall", "precision": "Precision", "f1": "F1-score"}[CLASS_METRIC]
    x_pos = np.arange(len(order)) + 0.5
    image = None

    for i, (result, panel_spec) in enumerate(zip(dataset_results, panels)):
        metrics = result["metrics_by_lead"][lead]
        cm = metrics["confusion"]
        scores = metrics["scores"]
        bottom_row = i >= 8
        right_col = i % 2 == 1
        label = result["label"]
        title = f"$\\mathbf{{({chr(ord('a') + i)})}}$ {wrap_long_label(label)}"
        inner = panel_spec.subgridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.10)

        ax_cm = fig.add_subplot(inner[0, 0])
        ax_bar = fig.add_subplot(inner[0, 1])

        image = ax_cm.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
        ax_cm.set_xticks(range(len(order)))
        ax_cm.set_xticklabels(x_labels if bottom_row else [], fontsize=G_CLASSIFICATION_STYLE["tick_label_size"])
        ax_cm.set_yticks(range(len(order)))
        ax_cm.set_yticklabels([] if right_col else y_labels, fontsize=G_CLASSIFICATION_STYLE["tick_label_size"])
        ax_cm.tick_params(
            length=G_CLASSIFICATION_STYLE["tick_length"],
            width=G_CLASSIFICATION_STYLE["tick_width"],
            pad=G_CLASSIFICATION_STYLE["tick_pad"],
        )
        if bottom_row:
            ax_cm.set_xlabel("Predicted", fontsize=G_CLASSIFICATION_STYLE["axis_label_size"], labelpad=2)
        ax_cm.set_title(title, loc="left", fontsize=G_CLASSIFICATION_STYLE["panel_label_size"], pad=2)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                v = cm[r, c]
                ax_cm.text(c, r, f"{v:.2f}", ha="center", va="center",
                           fontsize=G_CLASSIFICATION_STYLE["cell_label_size"], fontweight="semibold",
                           color="white" if v >= 0.60 else "#1A1A1A")
        disable_axis_grid(ax_cm)
        style_boxed_axes(ax_cm)

        bars = ax_bar.bar(
            x_pos,
            scores,
            width=0.74,
            color=EVENT_COLORS[:len(order)],
            edgecolor="#333333",
            linewidth=G_CLASSIFICATION_STYLE["bar_edge_width"],
        )
        for bar, v in zip(bars, scores):
            txt = "NA" if np.isnan(v) else f"{v:.3f}"
            ly = 0.025 if np.isnan(v) or v == 0 else min(v + 0.035, 1.045)
            ax_bar.text(bar.get_x() + bar.get_width() / 2, ly, txt,
                        ha="center", va="bottom",
                        fontsize=G_CLASSIFICATION_STYLE["bar_label_size"],
                        fontweight="semibold",
                        color="#666666" if np.isnan(v) else "#1A1A1A")
        ax_bar.set_xlim(0, len(order))
        ax_bar.set_ylim(0, 1.08)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(x_labels if bottom_row else [], fontsize=G_CLASSIFICATION_STYLE["tick_label_size"])
        ax_bar.set_yticks(np.arange(0, 1.01, 0.2))
        ax_bar.tick_params(
            axis="y",
            labelsize=G_CLASSIFICATION_STYLE["tick_label_size"],
            length=G_CLASSIFICATION_STYLE["tick_length"],
            width=G_CLASSIFICATION_STYLE["tick_width"],
            pad=G_CLASSIFICATION_STYLE["tick_pad"],
            labelleft=not right_col,
        )
        ax_bar.tick_params(
            axis="x",
            length=G_CLASSIFICATION_STYLE["tick_length"],
            width=G_CLASSIFICATION_STYLE["tick_width"],
            pad=G_CLASSIFICATION_STYLE["tick_pad"],
        )
        # F1-score title on both top-row bar panels.
        if i < 2:
            ax_bar.set_title(metric_label, fontsize=G_CLASSIFICATION_STYLE["metric_title_size"], pad=2)
        disable_axis_grid(ax_bar)
        ax_bar.set_axisbelow(True)
        style_open_axes(ax_bar)

    add_shared_axis_labels(
        fig,
        ylabel="Real",
        ylabel_x=0.010,
        fontsize=G_CLASSIFICATION_STYLE["axis_label_size"],
    )

    fig.canvas.draw()
    x0 = outer[0, 0].get_position(fig).x0
    x1 = outer[0, 1].get_position(fig).x1
    cax = fig.add_axes([x0, 0.012, x1 - x0, G_CLASSIFICATION_STYLE["heatmap_colorbar_height"]])
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.5, 1.0])
    style_colorbar(
        cbar,
        tick_labelsize=G_CLASSIFICATION_STYLE["colorbar_tick_size"],
        tick_length=1.6,
        tick_pad=1.1,
    )

    saved = save_publication_figure(
        fig,
        figure_output_paths(output_base),
        dpi=FIGURE_DPI, pad_inches=0.02,
    )
    plt.close(fig)
    return saved


def main():
    configure_publication_style()
    validate_data_sources(DATA_SOURCES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    order = event_order(N_TYPE)
    dataset_results = []
    for source in DATA_SOURCES:
        print(f"{'='*72}\nLoading {source['label']}")
        df = load_source_forecast_table({"pickle_dir": source["pickle_dir"]}, base_year=BASE_YEAR)
        metrics = compute_metrics(df, N_TYPE, CLASS_METRIC, LEADS)
        years = sorted(int(y) for y in df["pickle_year"].unique())
        print(f"Pickle files: {len(years)}; years: {years[0]}-{years[-1]}")
        print(f"Rows: {len(df)}; leads: {list(metrics)}")
        dataset_results.append({"id": source["id"], "label": source["label"], "metrics_by_lead": metrics})

    common = list(dataset_results[0]["metrics_by_lead"])
    for r in dataset_results[1:]:
        if list(r["metrics_by_lead"]) != common:
            raise ValueError("All data sources must share the same leads.")

    for lead in common:
        out = OUTPUT_DIR / f"{FIGURE_ID}_{FIGURE_NAME}_type{N_TYPE}_{CLASS_METRIC}_all_sources_lead{lead}"
        for path in plot_figure(dataset_results, lead, order, out):
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
