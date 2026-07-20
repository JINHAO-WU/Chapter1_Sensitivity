# Chapter 1 sensitivity analysis

Code for hyperparameter tuning, HamCNN sensitivity experiments, and figure generation for ENSO Niño 3.4 prediction.

## Project structure

| Folder | Purpose |
| --- | --- |
| `Parameter_Research_Optuna/` | Searches HamCNN training and architecture parameters with Optuna. Configure and run `optuna_tpe_search.py`. Results are stored in `optuna_runs/<SEARCH_VERSION>/`. |
| `Sensitivity_CNN_scripts/` | Trains and evaluates HamCNN across 30-year test windows using observations, CMIP6 data, or transfer learning. Set experiments in `A_Parameter_set.py` and run `A_Main_Run.py`. |
| `plot_scripts/` | Reads prediction pickles containing `predict_value` and `real_value`, calculates diagnostics, and creates publication figures. Shared paths and styles are defined in `A_basic_sources.py` and `plot_style.py`. |

### `Parameter_Research_Optuna/`

| Python file | Purpose |
| --- | --- |
| `optuna_tpe_search.py` | Runs the Optuna hyperparameter search and saves trial results. |
| `models.py` | Defines the HamCNN model used in the search. |
| `read_data.py` | Reads observational and CMIP6 data. |
| `data_utils.py` | Prepares datasets and provides scoring utilities. |

### `Sensitivity_CNN_scripts/`

| Python file | Purpose |
| --- | --- |
| `A_Main_Run.py` | Runs single or multiple ENSO experiments. |
| `A_Parameter_set.py` | Stores experiment, data, model, and output settings. |
| `data.py` | Loads data, slices train/test periods, and builds PyTorch datasets. |
| `Model_HamCNN.py` | Defines the HamCNN forecasting model. |
| `train_function.py` | Handles training, evaluation, and result serialisation. |
| `Z_plt_cut.py` | Plots forecast skill for an individual experiment. |

### `plot_scripts/`

| Python file | Purpose |
| --- | --- |
| `A_basic_sources.py` | Defines data sources and shared pickle-loading functions. |
| `A_plot_sampling_correlation.py` | Analyses sampling-based forecast correlations. |
| `B_plot_dl_nmme_lead_correlation.py` | Compares DL and NMME correlations across lead times. |
| `C_plot_target_season_correlation.py` | Plots correlation by target month and season. |
| `D_plot_enso_phase_mean_bias.py` | Calculates lead-dependent bias by ENSO phase. |
| `E_plot_enso_transition_frequency_vs_skill.py` | Relates ENSO transition frequency to forecast skill. |
| `F_plot_enso_event_errors.py` | Analyses ENSO event amplitude and peak-month errors. |
| `G_plot_enso_classification_metrics.py` | Produces ENSO classification metrics and confusion matrices. |
| `H_plot_nino34_variance_seasonality.py` | Analyses seasonal Niño 3.4 variance. |
| `I_plot_yearly_covariance_contribution.py` | Calculates annual covariance contributions for selected sources. |
| `I_1_plot_yearly_covariance_contribution.py` | Produces the expanded annual covariance contribution figure. |
| `plot_style.py` | Defines shared publication plotting styles and export settings. |
