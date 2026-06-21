"""Shared publication plotting style for the Chapter 1 sensitivity figures."""

from __future__ import annotations

from itertools import cycle

import matplotlib as mpl
from matplotlib.axes import Axes


# 优先使用 Arial；若系统未安装，则依次回退到 Helvetica、DejaVu Sans 和系统默认无衬线字体。
FONT_SANS_SERIF = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]

# 双栏（183 mm 宽）图件的统一字号层级，单位均为 point（pt）。
TICK_LABEL_SIZE = 10  # 坐标轴刻度标签和正文注释。
AXIS_LABEL_SIZE = 11  # x/y 轴标签。
LEGEND_SIZE = 8  # 图例；略小以避免多数据源图例拥挤。
PANEL_LABEL_SIZE = 10  # 子图编号，如 “(a)” 或 “a”。
TITLE_SIZE = 9  # 单个子图标题。

# 坐标轴和主刻度的统一线宽/长度；Matplotlib 使用 point（pt）作为显示单位。
AXES_LINEWIDTH = 0.65
TICK_LENGTH = 3

DATASET_COLORS = {
    "source_1": "#0072B2",
    "source_2": "#E69F00",
    "source_3": "#009E73",
    "source_4": "#D55E00",
    "source_5": "#CC79A7",
}

NMME_COLORS = ("#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#0072B2", "#999999")
EVENT_COLORS = ("#CC6677", "#DDCC77", "#44AA99", "#88CCEE", "#AA4499", "#332288", "#999933")


def configure_publication_style() -> None:
    """Apply the shared Matplotlib defaults before creating figures."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_SANS_SERIF,
            "font.size": TICK_LABEL_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.linewidth": AXES_LINEWIDTH,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": AXES_LINEWIDTH,
            "ytick.major.width": AXES_LINEWIDTH,
            "xtick.major.size": TICK_LENGTH,
            "ytick.major.size": TICK_LENGTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def validate_data_sources(data_sources: list[dict]) -> None:
    """Validate immutable IDs, display labels, and shared color assignments."""
    source_ids = [source.get("id") for source in data_sources]
    labels = [source.get("label") for source in data_sources]
    if any(not isinstance(source_id, str) or not source_id for source_id in source_ids):
        raise ValueError('Every data source must define a non-empty string "id".')
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Data-source IDs must be unique.")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError('Every data source must define a non-empty string "label".')
    missing_colors = [source_id for source_id in source_ids if source_id not in DATASET_COLORS]
    if missing_colors:
        raise ValueError(f"No shared color configured for data-source IDs: {missing_colors}")


def dataset_color(dataset_id: str) -> str:
    """Return the fixed display color for an immutable data-source ID."""
    try:
        return DATASET_COLORS[dataset_id]
    except KeyError as error:
        raise KeyError(f"No shared color configured for data-source ID {dataset_id!r}.") from error


def nmme_color_mapping(labels: list[str]) -> dict[str, str]:
    """Assign configured NMME labels a stable color in their displayed order."""
    return {label: color for label, color in zip(labels, cycle(NMME_COLORS))}


def style_open_axes(ax: Axes) -> None:
    """Apply the open-frame style used by ordinary Cartesian plots."""
    for spine in ax.spines.values():
        spine.set_linewidth(AXES_LINEWIDTH)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="in", length=TICK_LENGTH, width=AXES_LINEWIDTH)


def style_boxed_axes(ax: Axes) -> None:
    """Apply the full-frame style used by heatmaps and colorbars."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXES_LINEWIDTH)
    ax.tick_params(axis="both", direction="in", length=TICK_LENGTH, width=AXES_LINEWIDTH)
