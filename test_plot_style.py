"""Smoke tests for the shared publication plotting style."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_style import (
    AXES_LINEWIDTH,
    DATASET_COLORS,
    configure_publication_style,
    nmme_color_mapping,
    style_boxed_axes,
    style_open_axes,
    validate_data_sources,
)


def test_shared_style_and_axes_frames() -> None:
    configure_publication_style()
    fig, (open_ax, boxed_ax) = plt.subplots(1, 2)
    style_open_axes(open_ax)
    style_boxed_axes(boxed_ax)

    assert not open_ax.spines["top"].get_visible()
    assert not open_ax.spines["right"].get_visible()
    assert boxed_ax.spines["top"].get_visible()
    assert boxed_ax.spines["right"].get_visible()
    assert boxed_ax.spines["left"].get_linewidth() == AXES_LINEWIDTH

    colorbar = fig.colorbar(boxed_ax.imshow(np.arange(4).reshape(2, 2)), ax=boxed_ax)
    style_boxed_axes(colorbar.ax)
    assert colorbar.ax.spines["top"].get_visible()
    assert colorbar.ax.spines["right"].get_visible()
    assert matplotlib.rcParams["xtick.direction"] == "in"
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert DATASET_COLORS["source_1"] == "#0072B2"
    assert DATASET_COLORS["source_5"] == "#CC79A7"
    assert nmme_color_mapping(["Model A", "Model B"]) == {
        "Model A": "#56B4E9",
        "Model B": "#E69F00",
    }
    plt.close(fig)


def test_dataset_id_is_independent_of_display_label() -> None:
    data_sources = [
        {"id": "source_1", "label": "NOAA sea-surface temperature"},
        {"id": "source_2", "label": "HadISST"},
    ]
    validate_data_sources(data_sources)
    assert DATASET_COLORS[data_sources[0]["id"]] == "#0072B2"
    assert DATASET_COLORS[data_sources[1]["id"]] == "#E69F00"
