"""ENSO transition frequency versus forecast skill.

Each pickle file is treated as one test dataset/window. Edit the configuration
block below to change data sources, leading time, transition mode, or output
location. The script is designed as a direct-run research plotting script, not a
command-line tool.
"""

from __future__ import annotations

import math
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import stats

from A_basic_sources import FIGURE_ROOT, get_dl_sources, list_pickle_files
from plot_style import configure_publication_style as configure_shared_publication_style, style_boxed_axes, style_open_axes, validate_data_sources


# =============================================================================
# User configuration
# =============================================================================

FIGURE_ID = "E"
FIGURE_NAME = "enso_transition_frequency_vs_skill"
OUTPUT_DIR = FIGURE_ROOT / f"{FIGURE_ID}_{FIGURE_NAME}"
FIGURE_DPI = 600
OUTPUT_FORMATS = ("png", "pdf") # "png" "pdf"
PUB_FIG_WIDTH_MM = 183
PUB_FIG_WIDTH_IN = PUB_FIG_WIDTH_MM / 25.4

MAKE_SKILL_RELATIONSHIP_PLOT = True
MAKE_TRANSITION_TIME_PLOT = True

LEAD = 6  # 1-based leading time. lead=6 uses array column index 5.
INPUT_MONTHS = 6  # input6 has 336 samples; input12 usually has 330 samples.

ENSO_THRESHOLD = 0.5
ENSO_EXIT_THRESHOLD = 0.4
EXTREME_ENSO_THRESHOLD = 1.5

# Transition method used by all supported transition modes:
#   "hysteresis"  -> ordinary ENSO runs use +/-0.5C entry and +/-0.4C exit;
#                   extreme modes then check raw Nino3.4 values inside each event run
#   "independent" -> ordinary ENSO runs use raw +/-0.5C thresholds;
#                   extreme ENSO runs use raw +/-1.5C thresholds
TRANSITION_METHOD = "independent"

# Options:
#   "el_nino_neutral_el_nino"
#   "la_nina_neutral_la_nina"
#   "neutral_el_nino_neutral"
#   "neutral_la_nina_neutral"
#   "extreme_el_nino_neutral_extreme_el_nino"
#   "extreme_la_nina_neutral_extreme_la_nina"
TRANSITION_MODE = "la_nina_neutral_la_nina"
ACC_WARNING_TOLERANCE = 0.02

SHOW_FIGURE = False
# Set to None to disable highlighted years.
ANNOTATE_YEAR_RANGE: tuple[int, int] | None = (1922, 1944) # None
X_AXIS_PADDING_FRACTION = 0.12
X_AXIS_MIN_PADDING = 0.0008

PLOT_STYLE = {
    "cmap": "PuOr",
    "point_size": 24,
    "point_alpha": 0.92,
    "point_edge_color": "#303030",
    "fit_line_color": "#1f1f1f",
    "grid_color": "#e6e6e6",
    "highlight_color": "#b2182b",
}

DATA_SOURCES = get_dl_sources()

ORDINARY_TRANSITION_MODES = {
    "el_nino_neutral_el_nino",
    "la_nina_neutral_la_nina",
    "neutral_el_nino_neutral",
    "neutral_la_nina_neutral",
}
EXTREME_TRANSITION_MODES = {
    "extreme_el_nino_neutral_extreme_el_nino",
    "extreme_la_nina_neutral_extreme_la_nina",
}

MODE_LABELS = {
    "el_nino_neutral_el_nino": "EN-Neutral-EN",
    "la_nina_neutral_la_nina": "LN-Neutral-LN",
    "neutral_el_nino_neutral": "Neutral-EN-Neutral",
    "neutral_la_nina_neutral": "Neutral-LN-Neutral",
    "extreme_el_nino_neutral_extreme_el_nino": "ExEN-Neutral-ExEN",
    "extreme_la_nina_neutral_extreme_la_nina": "ExLN-Neutral-ExLN",
}


@dataclass
class StateRun:
    state: int
    start: int
    end: int


@dataclass
class WindowPoint:
    dataset: str  # Immutable data-source ID used for grouping and matching.
    dataset_label: str  # Display text used only in figure titles.
    pickle_dir: Path
    file: Path
    label: str
    year: float
    transition_frequency: float
    acc: float
    recalculated_acc: float


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected a dict, got {type(data).__name__}")
    return data


def get_required_array(data: dict, field_name: str, path: Path) -> np.ndarray:
    if field_name not in data:
        raise KeyError(f"{path.name}: missing required field {field_name!r}")
    return np.asarray(data[field_name])


def validate_leading_time(lead_index: int, number_of_leads: int, field_name: str, path: Path) -> None:
    if lead_index < 0 or lead_index >= number_of_leads:
        raise ValueError(
            f"{path.name}: LEAD={LEAD} is outside available range for {field_name}. "
            f"Available leading times are 1-{number_of_leads}."
        )


def pearson_r(x_values: np.ndarray, y_values: np.ndarray) -> float:
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if valid.sum() < 2:
        return math.nan

    x_valid = x_values[valid].astype(float)
    y_valid = y_values[valid].astype(float)
    if np.nanstd(x_valid) == 0 or np.nanstd(y_valid) == 0:
        return math.nan
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def pearson_p_value(r_value: float, sample_count: int) -> float:
    if not np.isfinite(r_value) or sample_count < 3 or abs(r_value) >= 1:
        return math.nan
    t_value = r_value * math.sqrt((sample_count - 2) / (1 - r_value**2))
    return float(2 * stats.t.sf(abs(t_value), df=sample_count - 2))


def classify_enso_state_independent(nino34_values: np.ndarray) -> np.ndarray:
    states = np.zeros(nino34_values.shape, dtype=np.int8)
    states[nino34_values > ENSO_THRESHOLD] = 1
    states[nino34_values < -ENSO_THRESHOLD] = -1
    return states

def classify_enso_state_hysteresis(nino34_values: np.ndarray) -> np.ndarray:
    if ENSO_EXIT_THRESHOLD >= ENSO_THRESHOLD:
        raise ValueError("ENSO_EXIT_THRESHOLD must be smaller than ENSO_THRESHOLD.")

    states = np.zeros(nino34_values.shape, dtype=np.int8)
    current_state = 0
    for index, value in enumerate(nino34_values):
        if not np.isfinite(value):
            states[index] = current_state
            continue

        if current_state == 1:
            if value < -ENSO_THRESHOLD:
                current_state = -1
            elif value < ENSO_EXIT_THRESHOLD:
                current_state = 0
        elif current_state == -1:
            if value > ENSO_THRESHOLD:
                current_state = 1
            elif value > -ENSO_EXIT_THRESHOLD:
                current_state = 0
        else:
            if value > ENSO_THRESHOLD:
                current_state = 1
            elif value < -ENSO_THRESHOLD:
                current_state = -1

        states[index] = current_state

    return states


def classify_extreme_state_independent(nino34_values: np.ndarray) -> np.ndarray:
    states = np.zeros(nino34_values.shape, dtype=np.int8)
    states[nino34_values >= EXTREME_ENSO_THRESHOLD] = 1
    states[nino34_values <= -EXTREME_ENSO_THRESHOLD] = -1
    return states


def iter_state_runs(states: np.ndarray, valid_values: np.ndarray) -> list[StateRun]:
    runs: list[StateRun] = []
    start: int | None = None
    current_state: int | None = None

    for index, (state, is_valid) in enumerate(zip(states, valid_values)):
        if not is_valid:
            if start is not None and current_state is not None:
                runs.append(StateRun(int(current_state), start, index))
            start = None
            current_state = None
            continue

        state = int(state)
        if start is None:
            start = index
            current_state = state
        elif state != current_state:
            runs.append(StateRun(int(current_state), start, index))
            start = index
            current_state = state

    if start is not None and current_state is not None:
        runs.append(StateRun(int(current_state), start, len(states)))

    return runs


def run_is_extreme_event(run: StateRun, nino34_values: np.ndarray, event_state: int) -> bool:
    run_values = nino34_values[run.start : run.end]
    if event_state == 1:
        return bool(np.nanmax(run_values) >= EXTREME_ENSO_THRESHOLD)
    if event_state == -1:
        return bool(np.nanmin(run_values) <= -EXTREME_ENSO_THRESHOLD)
    raise ValueError("event_state must be 1 or -1.")


def ordinary_transition_runs(nino34_values: np.ndarray, valid_values: np.ndarray) -> list[StateRun]:
    if TRANSITION_METHOD == "hysteresis":
        states = classify_enso_state_hysteresis(nino34_values)
    elif TRANSITION_METHOD == "independent":
        states = classify_enso_state_independent(nino34_values)
    else:
        raise ValueError("TRANSITION_METHOD must be 'hysteresis' or 'independent'.")
    return iter_state_runs(states, valid_values)


def extreme_transition_runs(nino34_values: np.ndarray, valid_values: np.ndarray) -> list[StateRun]:
    if TRANSITION_METHOD == "hysteresis":
        states = classify_enso_state_hysteresis(nino34_values)
    elif TRANSITION_METHOD == "independent":
        states = classify_extreme_state_independent(nino34_values)
    else:
        raise ValueError("TRANSITION_METHOD must be 'hysteresis' or 'independent'.")
    return iter_state_runs(states, valid_values)


def count_event_neutral_boundaries(runs: list[StateRun], event_state: int) -> int:
    transition_count = 0
    for previous_run, next_run in zip(runs[:-1], runs[1:]):
        if previous_run.state == event_state and next_run.state == 0:
            transition_count += 1
        elif previous_run.state == 0 and next_run.state == event_state:
            transition_count += 1
    return transition_count


def count_event_neutral_event_sequences(runs: list[StateRun], event_state: int) -> int:
    transition_count = 0
    for first_run, middle_run, last_run in zip(runs[:-2], runs[1:-1], runs[2:]):
        if first_run.state == event_state and middle_run.state == 0 and last_run.state == event_state:
            transition_count += 1
    return transition_count


def count_neutral_event_neutral_sequences(runs: list[StateRun], event_state: int) -> int:
    """Count contiguous Neutral -> ENSO event -> Neutral state-run sequences."""
    transition_count = 0
    for first_run, middle_run, last_run in zip(runs[:-2], runs[1:-1], runs[2:]):
        if first_run.state == 0 and middle_run.state == event_state and last_run.state == 0:
            transition_count += 1
    return transition_count


def count_extreme_event_neutral_event_sequences(
    runs: list[StateRun],
    nino34_values: np.ndarray,
    event_state: int,
) -> int:
    transition_count = 0
    for first_run, middle_run, last_run in zip(runs[:-2], runs[1:-1], runs[2:]):
        if first_run.state != event_state or middle_run.state != 0 or last_run.state != event_state:
            continue
        if not run_is_extreme_event(first_run, nino34_values, event_state):
            continue
        if not run_is_extreme_event(last_run, nino34_values, event_state):
            continue
        transition_count += 1
    return transition_count


def calculate_transition_metric(nino34_values: np.ndarray, transition_mode: str) -> float:
    nino34_values = np.asarray(nino34_values, dtype=float).reshape(-1)
    if nino34_values.size < 2:
        return math.nan

    valid_values = np.isfinite(nino34_values)
    valid_pairs = valid_values[:-1] & valid_values[1:]
    if valid_pairs.sum() == 0:
        return math.nan

    if transition_mode in ORDINARY_TRANSITION_MODES:
        runs = ordinary_transition_runs(nino34_values, valid_values)
        if transition_mode == "el_nino_neutral_el_nino":
            transition_count = count_event_neutral_event_sequences(runs, event_state=1)
        elif transition_mode == "la_nina_neutral_la_nina":
            transition_count = count_event_neutral_event_sequences(runs, event_state=-1)
        elif transition_mode == "neutral_el_nino_neutral":
            transition_count = count_neutral_event_neutral_sequences(runs, event_state=1)
        elif transition_mode == "neutral_la_nina_neutral":
            transition_count = count_neutral_event_neutral_sequences(runs, event_state=-1)
    elif transition_mode in EXTREME_TRANSITION_MODES:
        runs = extreme_transition_runs(nino34_values, valid_values)
        if transition_mode == "extreme_el_nino_neutral_extreme_el_nino":
            transition_count = count_extreme_event_neutral_event_sequences(runs, nino34_values, event_state=1)
        elif transition_mode == "extreme_la_nina_neutral_extreme_la_nina":
            transition_count = count_extreme_event_neutral_event_sequences(runs, nino34_values, event_state=-1)
    else:
        valid_modes = sorted(ORDINARY_TRANSITION_MODES | EXTREME_TRANSITION_MODES)
        raise ValueError(f"Unknown TRANSITION_MODE={transition_mode!r}. Use one of {valid_modes}.")

    return float(transition_count / valid_pairs.sum())


def transition_mode_is_extreme() -> bool:
    return TRANSITION_MODE in EXTREME_TRANSITION_MODES


def transition_mode_label() -> str:
    return MODE_LABELS.get(TRANSITION_MODE, TRANSITION_MODE)


def parse_window_year(path: Path, fallback_index: int) -> tuple[str, float]:
    match = re.search(r"(?<!\d)(\d{4})(?!\d)", path.stem)
    if match:
        label = match.group(1)
        return label, float(label)
    return path.stem, float(fallback_index)


def collect_source_window_points(source: dict) -> list[WindowPoint]:
    """Collect transition-frequency and skill points for one data source."""
    if LEAD < 1:
        raise ValueError("LEAD must be >= 1 because leading time is 1-based.")

    dataset_id = source["id"]
    dataset_label = source["label"]
    pickle_dir = source["pickle_dir"]
    lead_index = LEAD - 1
    pickle_files = list_pickle_files(pickle_dir)

    points: list[WindowPoint] = []
    for file_index, pickle_path in enumerate(pickle_files):
        data = load_pickle(pickle_path)
        real_value = get_required_array(data, "real_value", pickle_path)
        predict_value = get_required_array(data, "predict_value", pickle_path)
        acc_values = get_required_array(data, "Pearson", pickle_path).reshape(-1)

        if real_value.ndim != 2:
            raise ValueError(f"{pickle_path.name}: real_value must be 2-D, got {real_value.shape}")
        if predict_value.ndim != 2:
            raise ValueError(f"{pickle_path.name}: predict_value must be 2-D, got {predict_value.shape}")
        if real_value.shape != predict_value.shape:
            raise ValueError(
                f"{pickle_path.name}: real_value shape {real_value.shape} does not match "
                f"predict_value shape {predict_value.shape}"
            )

        validate_leading_time(lead_index, real_value.shape[1], "real_value", pickle_path)
        validate_leading_time(lead_index, predict_value.shape[1], "predict_value", pickle_path)
        validate_leading_time(lead_index, acc_values.size, "Pearson", pickle_path)

        observed_series = real_value[:, lead_index].astype(float)
        predicted_series = predict_value[:, lead_index].astype(float)
        stored_acc = float(acc_values[lead_index])
        recalculated_acc = pearson_r(predicted_series, observed_series)

        if (
            np.isfinite(stored_acc)
            and np.isfinite(recalculated_acc)
            and abs(stored_acc - recalculated_acc) > ACC_WARNING_TOLERANCE
        ):
            warnings.warn(
                f"{pickle_path.name}: stored Pearson for lead {LEAD} = {stored_acc:.3f} differs "
                f"from recalculated ACC = {recalculated_acc:.3f}",
                RuntimeWarning,
            )

        label, year = parse_window_year(pickle_path, file_index)
        points.append(
            WindowPoint(
                dataset=dataset_id,
                dataset_label=dataset_label,
                pickle_dir=pickle_dir,
                file=pickle_path,
                label=label,
                year=year,
                transition_frequency=calculate_transition_metric(observed_series, TRANSITION_MODE),
                acc=stored_acc,
                recalculated_acc=recalculated_acc,
            )
        )

    years = sorted(point.year for point in points if np.isfinite(point.year))
    print("=" * 72)
    print(f"{source['label']}: found {len(points)} pickle files")
    print(f"Pickle directory: {pickle_dir}")
    if years:
        print(f"Years: {int(years[0])}-{int(years[-1])}")
    valid_transition = [
        point.transition_frequency
        for point in points
        if np.isfinite(point.transition_frequency)
    ]
    valid_acc = [point.acc for point in points if np.isfinite(point.acc)]
    print(
        f"Valid transition frequencies: {len(valid_transition)}; "
        f"valid ACC values: {len(valid_acc)}"
    )

    return points


def collect_all_window_points(data_sources: list[dict]) -> list[WindowPoint]:
    """Collect transition-frequency and skill points for all data sources."""
    all_points: list[WindowPoint] = []
    for source in data_sources:
        all_points.extend(collect_source_window_points(source))
    return all_points


def transition_mode_filename_token() -> str:
    """Return a compact transition-mode token for output filenames."""
    tokens = {
        "el_nino_neutral_el_nino": "el_nino",
        "la_nina_neutral_la_nina": "la_nina",
        "neutral_el_nino_neutral": "neutral_el_nino",
        "neutral_la_nina_neutral": "neutral_la_nina",
        "extreme_el_nino_neutral_extreme_el_nino": "extreme_el_nino",
        "extreme_la_nina_neutral_extreme_la_nina": "extreme_la_nina",
    }
    return tokens.get(TRANSITION_MODE, TRANSITION_MODE)


def make_output_paths(figure_name: str) -> list[Path]:
    """Return configured output paths for one figure name."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = (
        f"{FIGURE_ID}_{FIGURE_NAME}_{figure_name}_{TRANSITION_METHOD}_"
        f"{transition_mode_filename_token()}_lead{LEAD}"
    )
    return [OUTPUT_DIR / f"{output_stem}.{file_format}" for file_format in OUTPUT_FORMATS]


def data_source_labels(points: list[WindowPoint]) -> list[str]:
    """Return data-source IDs in first-seen order."""
    labels: list[str] = []
    for point in points:
        if point.dataset not in labels:
            labels.append(point.dataset)
    return labels


def points_for_dataset(points: list[WindowPoint], dataset: str) -> list[WindowPoint]:
    """Return points for one data source."""
    return [point for point in points if point.dataset == dataset]


def display_label_for_dataset(points: list[WindowPoint], dataset_id: str) -> str:
    """Return the configured display label for an immutable data-source ID."""
    for point in points:
        if point.dataset == dataset_id:
            return point.dataset_label
    raise KeyError(f"No display label available for data-source ID {dataset_id!r}.")


def transition_axis_label() -> str:
    """Return a y/x-axis label for the active transition mode."""
    if transition_mode_is_extreme():
        return "Extreme ENSO-neutral-Extreme ENSO transition frequency"
    if TRANSITION_MODE.startswith("neutral_"):
        return "Neutral-ENSO-Neutral transition frequency"
    return "ENSO event-neutral transition frequency"


def configure_publication_style() -> None:
    """Apply a compact, journal-ready Matplotlib style."""
    configure_shared_publication_style()


def save_publication_figure(fig: plt.Figure, output_paths: list[Path]) -> None:
    """Save publication figures with consistent raster and vector settings."""
    for output_path in output_paths:
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {output_path}")


def add_figure_header(fig: plt.Figure, title: str, show_highlight_note: bool) -> None:
    """Add the shared title, mode, and optional highlighted-year note."""
    fig.suptitle(title, fontsize=11, y=0.985)
    fig.text(
        0.5,
        0.958,
        f"Mode: {transition_mode_label()}",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    if show_highlight_note and ANNOTATE_YEAR_RANGE is not None:
        start_year, end_year = ANNOTATE_YEAR_RANGE
        fig.text(
            0.5,
            0.934,
            f"Open red circles: {start_year}-{end_year}",
            ha="center",
            va="center",
            fontsize=7.5,
            color=PLOT_STYLE["highlight_color"],
        )


def panel_title(dataset: str, panel_index: int) -> str:
    """Return a panel title with an attached panel identifier."""
    panel_label = chr(ord("a") + panel_index)
    return f"({panel_label}) {dataset}"


def padded_limits(
    values: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    """Return padded limits constrained by lower and upper bounds."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Need at least one finite value to calculate axis limits.")

    data_min = float(np.nanmin(finite_values))
    data_max = float(np.nanmax(finite_values))
    data_range = data_max - data_min
    padding = max(data_range * X_AXIS_PADDING_FRACTION, X_AXIS_MIN_PADDING)
    return max(lower_bound, data_min - padding), min(upper_bound, data_max + padding)


def clean_ticks(limits: tuple[float, float], nbins: int = 6) -> np.ndarray:
    """Return readable ticks contained within the requested axis limits."""
    locator = MaxNLocator(nbins=nbins)
    ticks = locator.tick_values(limits[0], limits[1])
    ticks = ticks[(ticks >= limits[0]) & (ticks <= limits[1])]
    if ticks.size < 2:
        return np.linspace(limits[0], limits[1], nbins)
    return ticks


def global_skill_plot_axes(
    points: list[WindowPoint],
) -> tuple[tuple[float, float], tuple[float, float], np.ndarray, np.ndarray]:
    """Return shared x/y limits and ticks for Figure 1 panels."""
    transition_frequency = np.asarray(
        [point.transition_frequency for point in points],
        dtype=float,
    )
    acc = np.asarray([point.acc for point in points], dtype=float)
    valid = np.isfinite(transition_frequency) & np.isfinite(acc)
    if valid.sum() < 2:
        raise ValueError("Need at least two valid points to calculate shared axes.")

    x_limits = padded_limits(transition_frequency[valid], 0.0, 1.0)
    y_limits = padded_limits(acc[valid], -1.0, 1.0)
    x_ticks = clean_ticks(x_limits)
    y_ticks = clean_ticks(y_limits)
    return x_limits, y_limits, x_ticks, y_ticks


def plot_transition_skill_panel(
    ax: plt.Axes,
    points: list[WindowPoint],
    dataset: str,
    year_min: float,
    year_max: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    x_ticks: np.ndarray,
    y_ticks: np.ndarray,
    panel_index: int,
):
    """Draw one transition-frequency versus skill panel."""
    transition_frequency = np.asarray([point.transition_frequency for point in points], dtype=float)
    acc = np.asarray([point.acc for point in points], dtype=float)
    years = np.asarray([point.year for point in points], dtype=float)

    valid = np.isfinite(transition_frequency) & np.isfinite(acc)
    if valid.sum() < 2:
        ax.text(
            0.5,
            0.5,
            "Fewer than two valid points",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_title(panel_title(display_label_for_dataset(points, dataset), panel_index), loc="left")
        return None

    r_value = pearson_r(transition_frequency[valid], acc[valid])
    p_value = pearson_p_value(r_value, int(valid.sum()))

    scatter = ax.scatter(
        transition_frequency[valid],
        acc[valid],
        c=years[valid],
        cmap=PLOT_STYLE["cmap"],
        vmin=year_min,
        vmax=year_max,
        s=PLOT_STYLE["point_size"],
        edgecolor=PLOT_STYLE["point_edge_color"],
        linewidth=0.35,
        alpha=PLOT_STYLE["point_alpha"],
    )

    if np.nanstd(transition_frequency[valid]) > 0:
        fit = np.polyfit(transition_frequency[valid], acc[valid], deg=1)
        x_line = np.linspace(
            float(np.nanmin(transition_frequency[valid])),
            float(np.nanmax(transition_frequency[valid])),
            100,
        )
        y_line = fit[0] * x_line + fit[1]
        ax.plot(x_line, y_line, color=PLOT_STYLE["fit_line_color"], linewidth=1.0, alpha=0.9)

    ax.set_title(panel_title(display_label_for_dataset(points, dataset), panel_index), loc="left")
    style_open_axes(ax)
    ax.grid(True, color=PLOT_STYLE["grid_color"], linewidth=0.45, alpha=0.8)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    p_text = "NA" if not np.isfinite(p_value) else f"{p_value:.2g}"
    stats_text = f"r = {r_value:.2f}\np = {p_text}"
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=7.5,
        bbox={
            "boxstyle": "square,pad=0.14",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.76,
        },
        clip_on=False,
    )

    if ANNOTATE_YEAR_RANGE is not None:
        annotate_start, annotate_end = ANNOTATE_YEAR_RANGE
        annotate_indices = np.where(valid & (years >= annotate_start) & (years <= annotate_end))[0]
        ax.scatter(
            transition_frequency[annotate_indices],
            acc[annotate_indices],
            s=72,
            facecolors="none",
            edgecolors=PLOT_STYLE["highlight_color"],
            linewidth=1.0,
            zorder=5,
        )

    return scatter


def plot_transition_skill_relationship(points: list[WindowPoint]) -> None:
    dataset_labels = data_source_labels(points)
    years = np.asarray([point.year for point in points], dtype=float)
    valid_years = years[np.isfinite(years)]
    if valid_years.size == 0:
        raise ValueError("Need at least one valid year to draw the relationship plot.")
    x_limits, y_limits, x_ticks, y_ticks = global_skill_plot_axes(points)

    fig_height = max(6.6, 1.55 * len(dataset_labels) + 1.25)
    fig, axes = plt.subplots(
        len(dataset_labels),
        1,
        figsize=(PUB_FIG_WIDTH_IN, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.88,
        bottom=0.075,
        top=0.905,
        hspace=0.23,
    )

    scatter = None
    for panel_index, dataset in enumerate(dataset_labels):
        dataset_points = points_for_dataset(points, dataset)
        scatter = plot_transition_skill_panel(
            axes[panel_index, 0],
            dataset_points,
            dataset,
            float(np.nanmin(valid_years)),
            float(np.nanmax(valid_years)),
            x_limits,
            y_limits,
            x_ticks,
            y_ticks,
            panel_index,
        ) or scatter

    if scatter is not None:
        colorbar = fig.colorbar(
            scatter,
            ax=axes[:, 0].tolist(),
            fraction=0.025,
            pad=0.025,
        )
        colorbar.set_label("Test-start year")
        style_boxed_axes(colorbar.ax)

    title = (
        "Extreme ENSO Transition Frequency vs Forecast Skill"
        if transition_mode_is_extreme()
        else "ENSO Transition Frequency vs Forecast Skill"
    )
    add_figure_header(fig, title, show_highlight_note=True)
    axes[-1, 0].set_xlabel(transition_axis_label())
    fig.text(
        0.07,
        0.5,
        f"ACC at lead {LEAD}",
        va="center",
        rotation="vertical",
    )

    save_publication_figure(fig, make_output_paths("frequency_vs_skill"))

    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


def plot_transition_time_panel(
    ax: plt.Axes,
    points: list[WindowPoint],
    dataset: str,
    year_min: float,
    year_max: float,
    freq_min: float,
    freq_max: float,
    panel_index: int,
):
    """Draw one transition-frequency over time panel."""
    transition_frequency = np.asarray([point.transition_frequency for point in points], dtype=float)
    years = np.asarray([point.year for point in points], dtype=float)

    valid = np.isfinite(years) & np.isfinite(transition_frequency)
    if valid.sum() < 2:
        ax.text(
            0.5,
            0.5,
            "Fewer than two valid points",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_title(panel_title(display_label_for_dataset(points, dataset), panel_index), loc="left")
        return None

    sorted_indices = np.argsort(years[valid])
    sorted_years = years[valid][sorted_indices]
    sorted_frequency = transition_frequency[valid][sorted_indices]

    ax.plot(
        sorted_years,
        sorted_frequency,
        color=PLOT_STYLE["fit_line_color"],
        linewidth=0.95,
        alpha=0.82,
        zorder=1,
    )
    scatter = ax.scatter(
        sorted_years,
        sorted_frequency,
        c=sorted_years,
        cmap=PLOT_STYLE["cmap"],
        vmin=year_min,
        vmax=year_max,
        s=PLOT_STYLE["point_size"],
        edgecolor=PLOT_STYLE["point_edge_color"],
        linewidth=0.35,
        alpha=PLOT_STYLE["point_alpha"],
        zorder=2,
    )

    ax.set_title(panel_title(display_label_for_dataset(points, dataset), panel_index), loc="left")
    style_open_axes(ax)
    ax.grid(True, color=PLOT_STYLE["grid_color"], linewidth=0.45, alpha=0.8)
    ax.set_xlim(year_min, year_max)
    ax.set_ylim(freq_min, freq_max)

    return scatter


def plot_transition_frequency_over_time(points: list[WindowPoint]) -> None:
    dataset_labels = data_source_labels(points)
    years = np.asarray([point.year for point in points], dtype=float)
    transition_frequency = np.asarray([point.transition_frequency for point in points], dtype=float)

    valid = np.isfinite(years) & np.isfinite(transition_frequency)
    if valid.sum() < 2:
        raise ValueError("Need at least two valid points to draw the transition frequency time plot.")

    year_min = float(np.nanmin(years[valid]))
    year_max = float(np.nanmax(years[valid]))
    freq_min_raw = float(np.nanmin(transition_frequency[valid]))
    freq_max_raw = float(np.nanmax(transition_frequency[valid]))
    freq_range = freq_max_raw - freq_min_raw
    freq_padding = max(freq_range * X_AXIS_PADDING_FRACTION, X_AXIS_MIN_PADDING)
    freq_min = max(0.0, freq_min_raw - freq_padding)
    freq_max = min(1.0, freq_max_raw + freq_padding)

    fig_height = max(5.8, 1.25 * len(dataset_labels) + 1.15)
    fig, axes = plt.subplots(
        len(dataset_labels),
        1,
        figsize=(PUB_FIG_WIDTH_IN, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.88,
        bottom=0.085,
        top=0.925,
        hspace=0.18,
    )

    scatter = None
    for panel_index, dataset in enumerate(dataset_labels):
        dataset_points = points_for_dataset(points, dataset)
        scatter = plot_transition_time_panel(
            axes[panel_index, 0],
            dataset_points,
            dataset,
            year_min,
            year_max,
            freq_min,
            freq_max,
            panel_index,
        ) or scatter

    if scatter is not None:
        colorbar = fig.colorbar(
            scatter,
            ax=axes[:, 0].tolist(),
            fraction=0.025,
            pad=0.025,
        )
        colorbar.set_label("Test-start year")
        style_boxed_axes(colorbar.ax)

    title = (
        f"Extreme ENSO Transition Frequency Over Time | Leading {LEAD}M"
        if transition_mode_is_extreme()
        else f"ENSO Transition Frequency Over Time | Leading {LEAD}M"
    )
    add_figure_header(fig, title, show_highlight_note=False)
    axes[-1, 0].set_xlabel("Test-start year")
    fig.text(
        0.07,
        0.5,
        transition_axis_label(),
        va="center",
        rotation="vertical",
    )

    save_publication_figure(fig, make_output_paths("frequency_over_time"))

    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


def main() -> None:
    configure_publication_style()
    validate_data_sources(DATA_SOURCES)
    points = collect_all_window_points(DATA_SOURCES)
    if MAKE_SKILL_RELATIONSHIP_PLOT:
        plot_transition_skill_relationship(points)
    if MAKE_TRANSITION_TIME_PLOT:
        plot_transition_frequency_over_time(points)

    valid_transition = [point.transition_frequency for point in points if np.isfinite(point.transition_frequency)]
    valid_acc = [point.acc for point in points if np.isfinite(point.acc)]
    datasets = data_source_labels(points)
    print("=" * 72)
    print(
        f"Processed {len(points)} pickle files from {len(datasets)} data sources "
        f"for leading time {LEAD}. "
        f"Valid transition frequencies: {len(valid_transition)}; valid ACC values: {len(valid_acc)}."
    )


if __name__ == "__main__":
    main()
