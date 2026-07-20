"""Lightweight checks for shared helpers and figure-local plotting styles."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from plot_style import (
    AXES_LINEWIDTH,
    COLORBAR_TICK_SIZE,
    DATASET_COLORS,
    DEFAULT_OUTPUT_FORMATS,
    FONT_SANS_SERIF,
    LIGHT_GRID_COLOR,
    TICK_LENGTH,
    add_shared_axis_labels,
    configure_publication_style,
    dataset_color,
    disable_axis_grid,
    figure_output_paths,
    mm_to_inches,
    panel_title,
    reorder_legend_handles_by_row,
    source_panel_grid_5x2,
    style_boxed_axes,
    style_colorbar,
    style_light_grid,
    style_open_axes,
    style_source_panel_axes_5x2,
    validate_data_sources,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def _literal_assignment(file_name: str, variable_name: str) -> dict:
    """Read a top-level literal assignment without importing the plotting script."""
    source_path = SCRIPT_DIR / file_name
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if variable_name in names:
                return ast.literal_eval(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == variable_name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{variable_name} not found in {source_path.name}")


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

    style_light_grid(axes[0], axis="y")
    assert axes[0].yaxis._major_tick_kw["gridOn"]
    assert axes[0].yaxis._major_tick_kw["grid_color"] == LIGHT_GRID_COLOR
    disable_axis_grid(axes[0])
    assert not axes[0].yaxis._major_tick_kw["gridOn"]

    image = axes[1].imshow([[0, 1], [1, 0]])
    colorbar = fig.colorbar(image, ax=axes[1])
    style_colorbar(colorbar, label="Shared colorbar")
    assert colorbar.ax.get_ylabel() == "Shared colorbar"
    assert colorbar.ax.yaxis.get_ticklabels()[0].get_fontsize() == COLORBAR_TICK_SIZE
    plt.close(fig)


def test_dataset_ids_titles_and_source_layout() -> None:
    """Validate fixed 10-source colors, label independence, and title/layout helpers."""
    sources = [
        {"id": f"source_{index}", "label": f"Renamed source {index}"}
        for index in range(1, 11)
    ]
    validate_data_sources(sources)

    assert set(DATASET_COLORS) == {f"source_{index}" for index in range(1, 11)}
    assert dataset_color("source_1") == "#000000"
    assert dataset_color("source_8") == "#CC79A7"
    assert dataset_color("source_10") == "#17BECF"
    assert dataset_color(sources[0]["id"]) == DATASET_COLORS["source_1"]
    assert panel_title("a", "SST (NOAA)") == "(a) SST (NOAA)"

    fig = plt.figure(figsize=(7.2, 11.0))
    axes = source_panel_grid_5x2(fig)
    assert len(axes) == 10
    assert axes[0].get_position().y0 == axes[1].get_position().y0
    assert axes[0].get_position().x0 < axes[1].get_position().x0
    assert axes[2].get_position().y0 > axes[4].get_position().y0
    assert axes[8].get_position().y0 == axes[9].get_position().y0
    style_source_panel_axes_5x2(axes)
    fig.canvas.draw()
    assert axes[0].yaxis.get_ticklabels()[0].get_visible()
    assert not any(label.get_visible() for label in axes[1].yaxis.get_ticklabels())
    assert not any(label.get_visible() for label in axes[0].xaxis.get_ticklabels())
    assert axes[8].xaxis.get_ticklabels()[0].get_visible()

    add_shared_axis_labels(fig, xlabel="Shared x", ylabel="Shared y")
    assert fig._supxlabel.get_text() == "Shared x"
    assert fig._supylabel.get_text() == "Shared y"
    plt.close(fig)


def test_legend_reordering_reads_by_row() -> None:
    """Validate handle order needed for row-wise visual legend reading."""
    labels = [f"source_{index}" for index in range(1, 11)] + ["95% CI"]
    handles = list(range(len(labels)))
    reordered_handles, reordered_labels = reorder_legend_handles_by_row(handles, labels, ncol=4)
    assert reordered_labels == [
        "source_1",
        "source_5",
        "source_9",
        "source_2",
        "source_6",
        "source_10",
        "source_3",
        "source_7",
        "95% CI",
        "source_4",
        "source_8",
    ]
    assert reordered_handles == [labels.index(label) for label in reordered_labels]


def test_publication_output_helpers() -> None:
    """Validate shared output helpers independently of figure-local settings."""
    assert mm_to_inches(25.4) == 1.0
    assert DEFAULT_OUTPUT_FORMATS == ("png", "pdf")
    assert figure_output_paths(Path("figure_a")) == [
        Path("figure_a.png"),
        Path("figure_a.pdf"),
    ]


def test_figure_styles_are_local_and_valid() -> None:
    """Validate local style dictionaries without executing data-heavy scripts."""
    style_locations = [
        ("A_plot_sampling_correlation.py", "A_SAMPLING_STYLE"),
        ("B_plot_dl_nmme_lead_correlation.py", "B_LEAD_CORRELATION_STYLE"),
        ("C_plot_target_season_correlation.py", "C_TARGET_HEATMAP_STYLE"),
        ("D_plot_enso_phase_mean_bias.py", "D_ENSO_PHASE_STYLE"),
        ("E_plot_enso_transition_frequency_vs_skill.py", "E_TRANSITION_STYLE"),
        ("F_plot_enso_event_errors.py", "F_EVENT_ERRORS_STYLE"),
        ("G_plot_enso_classification_metrics.py", "G_CLASSIFICATION_STYLE"),
        ("H_plot_nino34_variance_seasonality.py", "H_VARIANCE_SEASONALITY_STYLE"),
        ("I_plot_yearly_covariance_contribution.py", "I_COVARIANCE_STYLE"),
        ("I_1_plot_yearly_covariance_contribution.py", "I_COVARIANCE_STYLE"),
    ]
    styles = {
        (file_name, style_name): _literal_assignment(file_name, style_name)
        for file_name, style_name in style_locations
    }

    for style in styles.values():
        assert style
        assert all(isinstance(key, str) for key in style)

    assert styles[("A_plot_sampling_correlation.py", "A_SAMPLING_STYLE")]["lead_colors"]
    assert styles[("B_plot_dl_nmme_lead_correlation.py", "B_LEAD_CORRELATION_STYLE")]["marker_size"] == 3.0
    assert styles[("C_plot_target_season_correlation.py", "C_TARGET_HEATMAP_STYLE")]["cell_label_size"] == 5.4
    assert styles[("D_plot_enso_phase_mean_bias.py", "D_ENSO_PHASE_STYLE")]["figure_width_mm"] == 183
    assert styles[("E_plot_enso_transition_frequency_vs_skill.py", "E_TRANSITION_STYLE")]["point_size"] == 24
    assert styles[("F_plot_enso_event_errors.py", "F_EVENT_ERRORS_STYLE")]["axis_label_size"] == 11
    assert styles[("G_plot_enso_classification_metrics.py", "G_CLASSIFICATION_STYLE")]["figure_width_mm"] == 183

    h_style = styles[("H_plot_nino34_variance_seasonality.py", "H_VARIANCE_SEASONALITY_STYLE")]
    assert h_style["forecast_line_width"] == 1.15
    assert h_style["observation_line_width"] == 1.15

    assert styles[("I_plot_yearly_covariance_contribution.py", "I_COVARIANCE_STYLE")]["line_figure_width_mm"] == 183
    assert styles[("I_1_plot_yearly_covariance_contribution.py", "I_COVARIANCE_STYLE")]["heatmap_figure_width_mm"] == 183


def test_shared_module_has_no_figure_specific_style_dicts() -> None:
    """Keep per-figure configuration out of the shared style module."""
    source_path = SCRIPT_DIR / "plot_style.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    assigned_names = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    forbidden_names = {
        "A_SAMPLING_STYLE",
        "B_LEAD_CORRELATION_STYLE",
        "C_TARGET_HEATMAP_STYLE",
        "D_ENSO_PHASE_STYLE",
        "E_TRANSITION_STYLE",
        "F_EVENT_ERRORS_STYLE",
        "G_CLASSIFICATION_STYLE",
        "H_VARIANCE_SEASONALITY_STYLE",
        "I_COVARIANCE_STYLE",
    }
    assert assigned_names.isdisjoint(forbidden_names)
