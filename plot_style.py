"""Shared publication plotting style for the Chapter 1 sensitivity figures."""

from __future__ import annotations

from itertools import cycle
from math import ceil

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# Preferred sans-serif font fallback chain for publication figures.
FONT_SANS_SERIF = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]

# Shared font-size hierarchy in points.
TICK_LABEL_SIZE = 10  # Tick labels and compact body annotations.
AXIS_LABEL_SIZE = 11  # x/y axis labels.
LEGEND_SIZE = 8  # Legends; kept compact for multi-source figures.
PANEL_LABEL_SIZE = 10  # Attached panel identifiers, e.g. "(a) SST (...)".
TITLE_SIZE = 9  # Individual subplot titles.

# Shared axis and major-tick stroke settings in points.
AXES_LINEWIDTH = 0.65
TICK_LENGTH = 3

DATASET_COLORS = {
    "source_1": "#000000",
    "source_2": "#E69F00",
    "source_3": "#009E73",
    "source_4": "#D55E00",
    "source_5": "#CC79A7",
    "source_6": "#56B4E9",
    "source_7": "#F0E442",
    "source_8": "#0072B2",
    "source_9": "#999999",
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


def reorder_legend_handles_by_row(handles: list, labels: list[str], ncol: int) -> tuple[list, list[str]]:
    """
    Reorder legend entries so multi-column legends read left-to-right by row.

    Matplotlib lays many multi-column legends out column-first. Passing entries
    in this order makes the visual rows follow the original handle/label order.
    """
    if ncol <= 1 or len(handles) <= ncol:
        return handles, labels

    reordered_handles = []
    reordered_labels = []
    n_items = len(handles)
    nrows = ceil(n_items / ncol)
    for col in range(ncol):
        for row in range(nrows):
            index = row * ncol + col
            if index < n_items:
                reordered_handles.append(handles[index])
                reordered_labels.append(labels[index])
    return reordered_handles, reordered_labels


def add_compact_figure_legend(
    fig: Figure,
    *,
    handles: list,
    labels: list[str] | None = None,
    ncol: int = 3,
    bbox_to_anchor: tuple[float, float] = (0.5, 0.98),
    fontsize: float = LEGEND_SIZE,
    columnspacing: float = 0.65,
    handlelength: float = 1.6,
    labelspacing: float = 0.3,
    loc: str = "upper center",
):
    """Add a compact row-readable figure legend."""
    if labels is None:
        labels = [handle.get_label() for handle in handles]
    handles, labels = reorder_legend_handles_by_row(handles, labels, ncol)
    return fig.legend(
        handles=handles,
        labels=labels,
        loc=loc,
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=fontsize,
        columnspacing=columnspacing,
        handlelength=handlelength,
        labelspacing=labelspacing,
    )


def panel_title(panel_letter: str, label: str) -> str:
    """Return the shared attached panel-title format."""
    return f"({panel_letter}) {label}"


def source_panel_grid_1_plus_8(
    fig: Figure,
    *,
    left: float = 0.08,
    right: float = 0.98,
    bottom: float = 0.08,
    top: float = 0.95,
    wspace: float = 0.14,
    hspace: float = 0.34,
) -> list[Axes]:
    """
    Create a 9-panel source layout with source_1 centered above a 4-by-2 grid.

    The returned axes are ordered for source_1, then source_2 through source_9.
    """
    grid = fig.add_gridspec(
        5,
        2,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )
    lower_cell = grid[1, 0].get_position(fig)
    top_row = grid[0, :].get_position(fig)
    top_left = top_row.x0 + (top_row.width - lower_cell.width) / 2
    top_bottom = top_row.y0 + (top_row.height - lower_cell.height) / 2
    return [
        fig.add_axes([top_left, top_bottom, lower_cell.width, lower_cell.height]),
        *[fig.add_subplot(grid[row, column]) for row in range(1, 5) for column in range(2)],
    ]


def style_source_panel_axes(
    axes: list[Axes],
    *,
    has_reference_top: bool = True,
    bottom_row_start: int | None = None,
) -> None:
    """Hide repeated tick labels in source-panel layouts."""
    if bottom_row_start is None:
        bottom_row_start = 7 if has_reference_top else max(0, len(axes) - 2)

    for panel_index, ax in enumerate(axes):
        if has_reference_top and panel_index == 0:
            is_right_column = False
        else:
            lower_index = panel_index - 1 if has_reference_top else panel_index
            is_right_column = lower_index % 2 == 1
        if is_right_column:
            ax.tick_params(axis="y", labelleft=False)
        if panel_index < bottom_row_start:
            ax.tick_params(axis="x", labelbottom=False)


def add_shared_axis_labels(
    fig: Figure,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlabel_y: float = 0.035,
    ylabel_x: float = 0.02,
    fontsize: float = AXIS_LABEL_SIZE,
) -> None:
    """Add shared figure-level axis labels using Matplotlib's native helpers."""
    if xlabel:
        fig.supxlabel(xlabel, y=xlabel_y, fontsize=fontsize)
    if ylabel:
        fig.supylabel(ylabel, x=ylabel_x, fontsize=fontsize)


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
