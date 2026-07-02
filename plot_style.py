"""Shared publication plotting style for the Chapter 1 sensitivity figures."""

from __future__ import annotations

from itertools import cycle
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
import seaborn as sns

if TYPE_CHECKING:
    pass  # Axes, Colorbar, Figure imported above are used in signatures.


# Preferred sans-serif font fallback chain for publication figures.
FONT_SANS_SERIF = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]

# Shared font-size hierarchy in points.
TICK_LABEL_SIZE = 10
AXIS_LABEL_SIZE = 11
LEGEND_SIZE = 8
PANEL_LABEL_SIZE = 10
TITLE_SIZE = 9
COMPACT_TICK_LABEL_SIZE = 7.0
COLORBAR_TICK_SIZE = 6.5
VALUE_LABEL_SIZE = 5.8
ANNOTATION_SIZE = 7.5

# Shared axis and major-tick stroke settings in points.
AXES_LINEWIDTH = 0.65
TICK_LENGTH = 3
COMPACT_TICK_LENGTH = 2.2
COMPACT_TICK_WIDTH = 0.6
COMPACT_TICK_PAD = 1.5
LIGHT_GRID_COLOR = "#D9D9D9"
LIGHT_GRID_LINEWIDTH = 0.55
LIGHT_GRID_ALPHA = 0.8

# Shared publication export and dimension defaults.
MM_PER_INCH = 25.4
DEFAULT_FIGURE_DPI = 600
DEFAULT_OUTPUT_FORMATS = ("png", "pdf")

# Figure A uses a compact 183-mm publication layout. Keep this preset local to
# the sampling-correlation figures so later figures can choose their own scale.
A_SAMPLING_STYLE = {
    "axis_label_size": 9.0,
    "tick_label_size": 8.0,
    "panel_label_size": 8.5,
    "legend_size": 7.0,
    "small_tick_label_size": 7.0,
    "line_width": 1.0,
    "comparison_line_width": 1.45,
    "reference_line_width": 0.7,
    "lead_colors": (
        "#f28e8c",
        "#4c9be8",
        "#59a14f",
        "#af7aa1",
        "#edc948",
        "#76b7b2",
    ),
}

# Figure B is a single lead-correlation line plot. Keep markers visible but
# smaller than the current draft so the dense legend does not dominate.
B_LEAD_CORRELATION_STYLE = {
    "axis_label_size": 8.5,
    "tick_label_size": 8.0,
    "legend_size": 6.8,
    "line_width": 1.25,
    "marker_size": 3.0,
    "zero_line_width": 0.6,
    "reference_line_width": 0.8,
}

# Figure C is a dense 10-panel heatmap. Use compact typography so the panel
# titles and shared axis labels support the matrices without overpowering them.
C_TARGET_HEATMAP_STYLE = {
    "axis_label_size": 8.5,
    "title_size": 7.8,
    "tick_label_size": 6.6,
    "colorbar_label_size": 7.0,
    "colorbar_tick_size": 6.3,
    "cell_label_size": 5.4,
    "tick_length": 1.8,
    "tick_width": 0.5,
    "tick_pad": 1.4,
}

# Figure D is a two-panel lead-bias line figure. Keep the panel spacing compact
# and the markers secondary to the lead-dependent trajectories.
D_ENSO_PHASE_STYLE = {
    "axis_label_size": 8.8,
    "tick_label_size": 8.0,
    "panel_label_size": 8.8,
    "legend_size": 6.8,
    "line_width": 1.25,
    "legend_line_width": 1.45,
    "marker_size": 3.0,
    "zero_line_width": 0.65,
    "grid_line_width": 0.55,
    "tick_length": 2.4,
    "tick_width": 0.6,
    "figure_width_mm": 183,
    "figure_height_mm": 132,
}

# Figure E is a dense 10-panel scatter/time-series figure. Keep annotations and
# point outlines legible without letting them compete with the shared trends.
E_TRANSITION_STYLE = {
    "axis_label_size": 8.7,
    "tick_label_size": 7.6,
    "panel_label_size": 8.2,
    "annotation_size": 6.5,
    "colorbar_label_size": 6.8,
    "colorbar_tick_size": 6.3,
    "colorbar_width": 0.010,
    "colorbar_pad": 0.008,
    "point_size": 24,
    "highlight_point_size": 54,
    "point_edge_width": 0.28,
    "highlight_edge_width": 0.85,
    "fit_line_width": 0.85,
    "time_line_width": 0.85,
    "grid_line_width": 0.38,
    "tick_length": 2.2,
    "tick_width": 0.55,
    "figure_width_mm": 183,
    "figure_height_mm": 270,
}

# Figure G combines small confusion matrices with per-class F1 bars. The
# annotation hierarchy needs to stay compact so both halves of each panel read
# as one unit.
G_CLASSIFICATION_STYLE = {
    "axis_label_size": 8.5,
    "tick_label_size": 6.2,
    "panel_label_size": 7.5,
    "metric_title_size": 7.0,
    "cell_label_size": 5.2,
    "bar_label_size": 5.2,
    "colorbar_tick_size": 6.0,
    "tick_length": 1.8,
    "tick_width": 0.5,
    "tick_pad": 1.2,
    "bar_edge_width": 0.55,
    "heatmap_colorbar_height": 0.010,
    "figure_width_mm": 183,
    "figure_height_mm": 270,
}

# Figure H overlays many seasonal lead trajectories in each source panel. Keep
# the forecast lines light enough for the observed annual cycle to remain clear.
H_VARIANCE_SEASONALITY_STYLE = {
    "axis_label_size": 8.8,
    "tick_label_size": 7.0,
    "panel_label_size": 8.0,
    "legend_size": 7.2,
    "marker_legend_size": 6.8,
    "forecast_line_width": 1.15,
    "observation_line_width": 1.65,
    "forecast_marker_size": 2.4,
    "marker_edge_width": 0.60,
    "forecast_alpha": 0.82,
    "tick_length": 2.0,
    "tick_width": 0.55,
    "tick_pad": 1.4,
    "figure_width_mm": 183,
    "figure_height_mm": 270,
}

# Figure I compares annual covariance contributions. The line-panel and
# heatmap variants share typography but keep separate aspect ratios.
I_COVARIANCE_STYLE = {
    "axis_label_size": 8.6,
    "tick_label_size": 6.8,
    "panel_label_size": 7.6,
    "annotation_size": 6.4,
    "legend_size": 7.0,
    "line_width": 1.15,
    "reference_line_width": 0.55,
    "colorbar_label_size": 7.0,
    "colorbar_tick_size": 6.3,
    "colorbar_width": 0.010,
    "line_figure_width_mm": 183,
    "line_figure_height_mm": 255,
    "heatmap_figure_width_mm": 183,
    "heatmap_figure_height_mm": 150,
}

# Wong (2011) palette — distinguishable, colourblind-friendly, B&W-printable.
DATASET_COLORS = {
    "source_1":  "#000000",
    "source_2":  "#E69F00",
    "source_3":  "#56B4E9",
    "source_4":  "#009E73",
    "source_5":  "#F0E442",
    "source_6":  "#0072B2",
    "source_7":  "#D55E00",
    "source_8":  "#CC79A7",
    "source_9":  "#8C564B",
    "source_10": "#17BECF",
}

NMME_COLORS = ("#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#0072B2", "#999999")
EVENT_COLORS = ("#CC6677", "#DDCC77", "#44AA99", "#88CCEE", "#AA4499", "#332288", "#999933")


def configure_publication_style() -> None:
    """Apply the shared Matplotlib defaults before creating figures."""
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="sans-serif",
        rc={
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.45,
            "axes.edgecolor": "#2F2F2F",
        },
    )
    matplotlib.rcParams.update(
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
    """Ensure every source has a unique non-empty string id and label, with an assigned color."""
    source_ids = [source.get("id") for source in data_sources]
    labels = [source.get("label") for source in data_sources]
    if any(not isinstance(sid, str) or not sid for sid in source_ids):
        raise ValueError('Every data source must define a non-empty string "id".')
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Data-source IDs must be unique.")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError('Every data source must define a non-empty string "label".')
    missing_colors = [sid for sid in source_ids if sid not in DATASET_COLORS]
    if missing_colors:
        raise ValueError(f"No shared color configured for data-source IDs: {missing_colors}")


def dataset_color(dataset_id: str) -> str:
    try:
        return DATASET_COLORS[dataset_id]
    except KeyError as error:
        raise KeyError(f"No shared color configured for data-source ID {dataset_id!r}.") from error


def nmme_color_mapping(labels: list[str]) -> dict[str, str]:
    return {label: color for label, color in zip(labels, cycle(NMME_COLORS))}


def reorder_legend_handles_by_row(handles: list, labels: list[str], ncol: int) -> tuple[list, list[str]]:
    """Reorder legend entries so multi-column legends read left-to-right by row.

    Matplotlib lays multi-column legends out column-first by default.  This
    reorder makes the visual rows follow the original handle/label order.
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
    ncol: int = 4,
    bbox_to_anchor: tuple[float, float] = (0.5, 0.98),
    fontsize: float = LEGEND_SIZE,
    columnspacing: float = 0.40,
    handlelength: float = 1.2,
    labelspacing: float = 0.25,
    loc: str = "upper center",
):
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
    return f"({panel_letter}) {label}"


def panel_title_only(panel_letter: str) -> str:
    return f"({panel_letter})"


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
    """Create a 9-panel source layout with source_1 centered above a 4-by-2 grid.

    The returned axes are ordered for source_1, then source_2 through source_9.
    """
    grid = fig.add_gridspec(
        5, 2, left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace,
    )
    lower_cell = grid[1, 0].get_position(fig)
    top_row = grid[0, :].get_position(fig)
    top_left = top_row.x0 + (top_row.width - lower_cell.width) / 2
    top_bottom = top_row.y0 + (top_row.height - lower_cell.height) / 2
    return [
        fig.add_axes([top_left, top_bottom, lower_cell.width, lower_cell.height]),
        *[fig.add_subplot(grid[row, column]) for row in range(1, 5) for column in range(2)],
    ]


def source_panel_grid_5x2(
    fig: Figure,
    *,
    left: float = 0.08,
    right: float = 0.98,
    bottom: float = 0.08,
    top: float = 0.95,
    wspace: float = 0.14,
    hspace: float = 0.30,
) -> list[Axes]:
    """Create a 10-panel equal source layout ordered row-wise in a 5-by-2 grid."""
    grid = fig.add_gridspec(
        5, 2, left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace,
    )
    return [fig.add_subplot(grid[row, column]) for row in range(5) for column in range(2)]


def style_source_panel_axes_5x2(axes: list[Axes], *, n_visible: int | None = None) -> None:
    """Hide repeated tick labels in equal 5-by-2 source-panel layouts."""
    if n_visible is None:
        n_visible = len(axes)
    visible_indices = list(range(n_visible))
    bottom_row_start = max(0, n_visible - 2)

    for panel_index, ax in enumerate(axes):
        if panel_index >= n_visible:
            ax.set_visible(False)
            continue
        _, column_index = divmod(panel_index, 2)
        if column_index == 1:
            ax.tick_params(axis="y", labelleft=False)
        if panel_index not in visible_indices[bottom_row_start:]:
            ax.tick_params(axis="x", labelbottom=False)


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
    if xlabel:
        fig.supxlabel(xlabel, y=xlabel_y, fontsize=fontsize)
    if ylabel:
        fig.supylabel(ylabel, x=ylabel_x, fontsize=fontsize)


def mm_to_inches(value_mm: float) -> float:
    """Convert millimetres to inches for Matplotlib figure sizes."""
    return float(value_mm) / MM_PER_INCH


def figure_output_paths(
    output_base: Path,
    formats: tuple[str, ...] = DEFAULT_OUTPUT_FORMATS,
) -> list[Path]:
    """Return output paths with the requested publication file extensions."""
    return [output_base.with_suffix(f".{output_format}") for output_format in formats]


def save_publication_figure(
    fig: Figure,
    output_paths: list[Path],
    *,
    dpi: int = 600,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.02,
    print_paths: bool = False,
) -> list[Path]:
    for output_path in output_paths:
        save_kwargs = {"bbox_inches": bbox_inches, "pad_inches": pad_inches}
        if output_path.suffix.lower() == ".png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        if print_paths:
            print(f"Saved {output_path}")
    return output_paths


def style_light_grid(
    ax: Axes,
    *,
    axis: str = "y",
    color: str = LIGHT_GRID_COLOR,
    linewidth: float = LIGHT_GRID_LINEWIDTH,
    linestyle: str = ":",
    alpha: float = LIGHT_GRID_ALPHA,
) -> None:
    ax.grid(True, axis=axis, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)


def disable_axis_grid(ax: Axes) -> None:
    ax.grid(False)


def style_compact_ticks(
    ax: Axes,
    *,
    labelsize: float = COMPACT_TICK_LABEL_SIZE,
    length: float = COMPACT_TICK_LENGTH,
    width: float = COMPACT_TICK_WIDTH,
    pad: float = COMPACT_TICK_PAD,
) -> None:
    ax.tick_params(axis="both", direction="in", labelsize=labelsize, length=length, width=width, pad=pad)


def style_colorbar(
    colorbar: Colorbar,
    *,
    label: str | None = None,
    fontsize: float = COMPACT_TICK_LABEL_SIZE,
    tick_labelsize: float = COLORBAR_TICK_SIZE,
    labelpad: float = 3,
    tick_length: float = 2,
    tick_width: float = 0.55,
    tick_pad: float = 1.8,
) -> None:
    if label:
        colorbar.set_label(label, fontsize=fontsize, labelpad=labelpad)
    colorbar.ax.tick_params(labelsize=tick_labelsize, length=tick_length, width=tick_width, pad=tick_pad)
    style_boxed_axes(colorbar.ax)


def style_open_axes(ax: Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(AXES_LINEWIDTH)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="in", length=TICK_LENGTH, width=AXES_LINEWIDTH)


def style_boxed_axes(ax: Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXES_LINEWIDTH)
    ax.tick_params(axis="both", direction="in", length=TICK_LENGTH, width=AXES_LINEWIDTH)
