"""Lightweight checks for the shared Chapter 1 plotting style."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from plot_style import (
    AXES_LINEWIDTH,
    DATASET_COLORS,
    FONT_SANS_SERIF,
    TICK_LENGTH,
    add_shared_axis_labels,
    configure_publication_style,
    dataset_color,
    panel_title,
    reorder_legend_handles_by_row,
    source_panel_grid_1_plus_8,
    style_boxed_axes,
    style_open_axes,
    style_source_panel_axes,
    validate_data_sources,
)


def test_shared_style_and_axes_frames() -> None:
    """Validate shared rcParams and open/boxed frame helpers."""
    configure_publication_style()
    assert matplotlib.rcParams["font.sans-serif"][: len(FONT_SANS_SERIF)] == FONT_SANS_SERIF
    assert matplotlib.rcParams["xtick.direction"] == "in"
    assert matplotlib.rcParams["ytick.direction"] == "in"
    assert matplotlib.rcParams["xtick.major.size"] == TICK_LENGTH

    fig, axes = plt.subplots(1, 2)
    style_open_axes(axes[0])
    assert not axes[0].spines["top"].get_visible()
    assert not axes[0].spines["right"].get_visible()
    assert axes[0].spines["left"].get_linewidth() == AXES_LINEWIDTH

    style_boxed_axes(axes[1])
    assert axes[1].spines["top"].get_visible()
    assert axes[1].spines["right"].get_visible()
    assert axes[1].spines["top"].get_linewidth() == AXES_LINEWIDTH
    plt.close(fig)


def test_dataset_ids_titles_and_source_layout() -> None:
    """Validate fixed 9-source colors, label independence, and title/layout helpers."""
    sources = [
        {"id": f"source_{index}", "label": f"Renamed source {index}"}
        for index in range(1, 10)
    ]
    validate_data_sources(sources)

    assert set(DATASET_COLORS) == {f"source_{index}" for index in range(1, 10)}
    assert dataset_color("source_1") == "#000000"
    assert dataset_color("source_8") == "#0072B2"
    assert dataset_color("source_9") == "#999999"
    assert dataset_color(sources[0]["id"]) == DATASET_COLORS["source_1"]
    assert panel_title("a", "SST (NOAA)") == "(a) SST (NOAA)"

    fig = plt.figure(figsize=(7.2, 11.0))
    axes = source_panel_grid_1_plus_8(fig)
    assert len(axes) == 9
    assert axes[0].get_position().x0 > axes[1].get_position().x0
    assert axes[0].get_position().x1 < axes[2].get_position().x1
    assert axes[1].get_position().y0 > axes[3].get_position().y0
    assert axes[7].get_position().y0 == axes[8].get_position().y0
    style_source_panel_axes(axes, has_reference_top=True)
    fig.canvas.draw()
    assert axes[1].yaxis.get_ticklabels()[0].get_visible()
    assert not any(label.get_visible() for label in axes[2].yaxis.get_ticklabels())
    assert not any(label.get_visible() for label in axes[1].xaxis.get_ticklabels())
    assert axes[7].xaxis.get_ticklabels()[0].get_visible()

    add_shared_axis_labels(fig, xlabel="Shared x", ylabel="Shared y")
    assert fig._supxlabel.get_text() == "Shared x"
    assert fig._supylabel.get_text() == "Shared y"
    plt.close(fig)


def test_legend_reordering_reads_by_row() -> None:
    """Validate handle order needed for row-wise visual legend reading."""
    labels = [f"source_{index}" for index in range(1, 10)] + ["95% CI"]
    handles = list(range(len(labels)))
    reordered_handles, reordered_labels = reorder_legend_handles_by_row(handles, labels, ncol=4)
    assert reordered_labels == [
        "source_1",
        "source_5",
        "source_9",
        "source_2",
        "source_6",
        "95% CI",
        "source_3",
        "source_7",
        "source_4",
        "source_8",
    ]
    assert reordered_handles == [labels.index(label) for label in reordered_labels]
