# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (sparam-surrogate)
#     language: python
#     name: sparam-surrogate
# ---

# %% tags=["remove-input"]
"""
Train non-neural scalar and vector insertion-loss baseline models.
"""

# Reloads all modules every time before executing code, except explicitly
# excluded using ``%aimport -<package>``, like ``%aimport -numpy``.
# %load_ext autoreload
# %autoreload 2
# %aimport -pathlib
# %aimport -numpy

# ruff: noqa: E402 -- Configure filtered notebook output before remaining imports.
from sparam_surrogate.config import configure_stdio_relative_path

# Display paths relative to project root or user home for consistent output across
# platforms. It should be called before other imports to setup filters.
configure_stdio_relative_path()

# %%
import pandas as pd
from IPython.display import display

from sparam_surrogate.config import SurrogateConfig
from sparam_surrogate.data import (
    PointwiseDataset,
    TouchstoneLoader,
    random_simu_indices,
)
from sparam_surrogate.models import (
    PolynomialModel,
    RandomForestModel,
    ScalarRidgeModel,
    VectorRidgeModel,
)
from sparam_surrogate.outputs.runner import ModelRunRunner
from sparam_surrogate.utils.json_io import read_json
from sparam_surrogate.utils.model_prediction_plots import (
    plot_design_model_comparison_curves,
    plot_design_prediction_curves,
)
from sparam_surrogate.utils.non_neural_modelling_utils import (
    per_target_metrics,
    plot_model_mae_comparison_by_frequency,
    plot_scalar_mae_by_frequency,
    plot_scalar_prediction_band_by_frequency,
    plot_scalar_true_vs_predicted,
    plot_shared_target_mae_comparison,
    plot_shared_target_prediction_bands,
    plot_vector_mae_by_frequency,
    plot_vector_prediction_bands_by_frequency,
    plot_vector_true_vs_predicted,
    regression_metrics,
)


def per_target_split_metrics(
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Return run metrics keyed by target name and split.
    """
    validation_by_target = validation_metrics.set_index("target")
    test_by_target = test_metrics.set_index("target")
    return {
        str(target_name): {
            "validation": validation_by_target.loc[target_name].to_dict(),
            "test": test_by_target.loc[target_name].to_dict(),
        }
        for target_name in validation_by_target.index
    }

load_pointwise_datasets = PointwiseDataset.from_frequency_expanded_csv

# %% [markdown]
# # Non-Neural Network Modelling
#
# In this notebook, I train non-neural baseline models to predict insertion-loss
# targets derived from S-parameters. The first baseline predicts one scalar IL
# value for one port pair. The remaining baselines predict one vector containing
# the six configured through-path IL values.
#
# Every target uses the real-valued positive insertion-loss convention
# $IL_{ij,\mathrm{dB}}=-20\log_{10}|S_{ij}|$.
#
# Run-dependent metrics and selected hyperparameters are reported by output cells and
# persisted artifacts rather than duplicated in Markdown. The analysis text explains
# how to interpret those results so it remains valid when the notebook is rerun.
#
# The cleaned CSV stores only design features, frequency, split labels, simulation
# indices, and Touchstone paths. The `TouchstoneLoader` reads Touchstone files on
# demand, then this notebook materializes the target arrays before fitting because
# the `scikit-learn` Ridge baselines expect in-memory `NumPy` arrays.

# %%
cfg = SurrogateConfig.from_config()
random_seed = cfg.project.seed
scalar_il_loader = TouchstoneLoader("scalar", cfg, "il", 8)
vector_il_loader = TouchstoneLoader("vector", cfg, "il", 8)
scalar_target_index = 0  # pylint: disable=invalid-name
scalar_target_name = scalar_il_loader.target_names[scalar_target_index]
vector_target_names = tuple(vector_il_loader.target_names)

print(f"Name of raw dataset: {cfg.dataset.name}")
print(f"Raw data directory: {cfg.dataset.path}")
print(f"Processed directory: {cfg.paths.processed_data}")
print(f"Run output directory: {cfg.paths.runs}")
print(f"Model registry directory: {cfg.paths.models}")
print(f"Benchmark directory: {cfg.paths.benchmarks}")
print("Configured IL port pairs: ", *cfg.dataset.ports)
print(f"Scalar target: {scalar_target_name}")
print("Vector target names:", *vector_target_names, sep=", ")

# %% [markdown]
# ## Shared Notebook Utilities
#
# Reusable model classes live in `src/sparam_surrogate/models/`. Metrics,
# frequency summaries, and plotting helpers live in
# `src/sparam_surrogate/utils/non_neural_modelling_utils.py`.

# %% [markdown]
# ## Scalar Data Loading And Validation
#
# The scalar experiment owns train, validation, and test dataset views configured
# with the scalar IL loader. On the first run, each view materializes its target
# array from Touchstone files and writes a split cache. Later runs load the cache
# whenever it is newer than the cleaned CSV.

# %%
scalar_train_set, scalar_val_set, scalar_test_set = load_pointwise_datasets(
    cfg.preprocessing.freq_expanded_csv,
    target_loader=scalar_il_loader,
    cache=True,
)
scalar_data_interface = {
    "dataset_name": cfg.dataset.name,
    "input_features": scalar_train_set.feature_columns,
    "target_names": (scalar_target_name,),
    "target_scope": "scalar",
    "target_units": "dB",
    "target_representation": "insertion_loss_db",
}

print(f"Number of scalar model training   samples: {len(scalar_train_set)}")
print(f"Number of scalar model validation samples: {len(scalar_val_set)}")
print(f"Number of scalar model test       samples: {len(scalar_test_set)}")

# %% [markdown]
# Build the in-memory arrays used by `scikit-learn`.
#
# `X` comes from the cleaned CSV. It contains design parameters plus frequency.
# The scalar `y` arrays come from split-specific caches or, on a cache miss, from
# the Touchstone loader. `ScalarRidgeModel` expects one-dimensional targets, so
# the single-column cached arrays are flattened by selecting column zero.

# %%
# pylint: disable=invalid-name
X_train_scalar = scalar_train_set.features
X_val_scalar = scalar_val_set.features
X_test_scalar = scalar_test_set.features
print(f"Shape of training   features: {X_train_scalar.shape}")
print(f"Shape of validation features: {X_val_scalar.shape}")
print(f"Shape of test       features: {X_test_scalar.shape}")

y_train_scalar = scalar_train_set.targets[:, 0]
y_val_scalar = scalar_val_set.targets[:, 0]
y_test_scalar = scalar_test_set.targets[:, 0]
print(f"Shape of training    targets: {y_train_scalar.shape}")
print(f"Shape of validation  targets: {y_val_scalar.shape}")
print(f"Shape of test        targets: {y_test_scalar.shape}")
for split_name, target_values in {
    "training": y_train_scalar,
    "validation": y_val_scalar,
    "test": y_test_scalar,
}.items():
    if (target_values <= 0.0).any():
        raise RuntimeError(f"{split_name.title()} scalar IL targets must be positive.")
# pylint: disable=invalid-name

# %% [markdown]
# The Ridge baseline needs full `NumPy` arrays. `PointwiseDataset` persists
# targets on disk, while `TouchstoneLoader` temporarily caches parsed
# Touchstone networks only during a cold load. Its small in-memory network cache
# can be cleared once the scalar arrays are ready.

# %%
print(f"Scalar Touchstone cache info: {scalar_il_loader.cache_info()}")
scalar_il_loader.clear_cache()
print(f"Scalar Touchstone cache after clearing: {scalar_il_loader.cache_info()}")

# %% [markdown]
# ## 1. Scalar Insertion Loss Baseline
#
# This baseline trains a non-neural scalar regressor to predict one insertion-loss
# value from a design-frequency feature vector. Full IL curves are reconstructed by
# evaluating the trained scalar model across all frequency points for the same
# design.
#
# ### 1.1 Input-Output Definition
#
# The input to the model is a vector of design features and frequency, and the
# output is a scalar insertion-loss value for port-pair `(7, 1)`:
#
# $$
#   \mathbf{x}_{i,k} = [\text{design parameters}, f_k]
#   \rightarrow IL_{7,1}(f_k)
# $$
#
# So the supervised-learning sample is:
#
# $$
#   \mathbf{x}_{i,k} \in \mathbb{R}^{D+1}, \quad y_{i,k} = IL_{7,1}(f_k)
# $$
#
# where `i` is the design index and `k` is the frequency index.

# %% [markdown]
# ### 1.2 Key Evaluation Metrics
#
# Model performance is measured on held-out validation and test sets. The main
# measures are mean absolute error and root mean squared error.
#
# $$
#   MAE = \frac{1}{N} \sum_{n=1}^{N} |y_n - \hat{y}_n|
# $$
#
# $$
#   RMSE = \sqrt{\frac{1}{N} \sum_{n=1}^{N} (y_n - \hat{y}_n)^2}
# $$

# %% [markdown]
# ### 1.3 Ridge Regression
#
# Ridge regression is ordinary linear regression with an L2 penalty that
# discourages large weights:
#
# $$
#     L_{\text{Ridge}}(w)
#     =
#     \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
#     + \lambda \sum_{j=1}^{D} w_j^2
# $$
#
# The regularisation strength is selected using validation MAE.

# %% [markdown]
# ### 1.3.1 Ridge Alpha Grid
#
# The scalar Ridge alpha grid is configured in `configs/default.json` under
# `models.scalar_ridge.alphas`:

# %%
scalar_ridge_config = cfg.models.scalar_ridge
print("Scalar Ridge alpha grid = ", *scalar_ridge_config.alphas)

# %% [markdown]
# In `scikit-learn`'s `Ridge`, this value is called `alpha`. It has the same role as
# $\lambda$ in the Ridge objective above. A small `alpha` keeps the model close to
# ordinary least squares, while a large `alpha` shrinks coefficients more strongly
# and can reduce overfitting. The notebook fits one model for each candidate value,
# evaluates each candidate on the validation set, and selects the `alpha` with the
# lowest validation MAE. The test set is used only after this selection, so the test
# metrics remain held-out estimates.

# %% [markdown]
# ### 1.3.2 `Scikit-Learn` Model Design
#
# Each alpha candidate uses the same two-step `scikit-learn` pipeline:
#
# ```python
# Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", Ridge(alpha=alpha)),
# ])
# ```
#
# `StandardScaler()` standardises every feature column using the training split
# mean and standard deviation. This is important because the input features use
# different physical units and numeric ranges, such as GHz, dielectric constant,
# trace length, and loss tangent.
#
# `Ridge(alpha=alpha)` then fits the regularised linear regression model on the
# scaled feature matrix. The `alpha` value controls the L2 penalty strength.

# %%
example_scalar_model = ScalarRidgeModel(alphas=(scalar_ridge_config.alphas[0],))
print("Example instantiated model:")
print(f"- name={example_scalar_model.name}")
print(f"- alphas={example_scalar_model.alphas}")

# %%
scalar_runner = ModelRunRunner(cfg, ScalarRidgeModel.from_config(scalar_ridge_config))
scalar_model = scalar_runner.train(
    X_train_scalar,
    y_train_scalar,
    X_val_scalar,
    y_val_scalar,
)

scalar_alpha_results = scalar_model.validation_results
best_scalar_alpha = scalar_model.best_alpha
if scalar_alpha_results is None or best_scalar_alpha is None:
    raise RuntimeError("Scalar Ridge model did not record validation results.")

print(f"- Scalar target: {scalar_target_name}")
print("- Scalar Ridge validation sweep:")
print(scalar_alpha_results)
print(f"- Best scalar alpha: {best_scalar_alpha:g}")
print(f"- Selected scalar preprocessing: {scalar_model.pipeline.named_steps['scaler']}")
print(f"- Selected scalar regressor: {scalar_model.pipeline.named_steps['model']}")

# %% [markdown]
# ### 1.4 Evaluate On Held-Out Test Data

# %%
scalar_validation_metrics = scalar_runner.validate(X_val_scalar, y_val_scalar)
scalar_test_metrics = scalar_runner.test(X_test_scalar, y_test_scalar)

y_test_pred_scalar = scalar_model.predict(X_test_scalar)

scalar_metrics = pd.DataFrame([
    {"split": "validation", **scalar_validation_metrics},
    {"split": "test", **scalar_test_metrics},
])
print(scalar_metrics)

# %% [markdown]
# The held-out test error is moderately higher than the validation error. The
# difference is more pronounced for RMSE than for MAE, indicating that a subset of
# difficult test samples contributes disproportionately to the generalisation gap.
# The average absolute-error gap is nevertheless limited, so there is no evidence of
# severe overall instability. The following plots localise the remaining errors.

# %% [markdown]
# ### 1.5 Plotting: Scalar IL Distribution Across Test Designs
#
# This plot groups all held-out test rows by frequency and compares the true and
# predicted median curves, together with the 10th-90th percentile bands across
# design variants.

# %%
fig_scalar_distribution = plot_scalar_prediction_band_by_frequency(
    scalar_test_set.dataframe,
    y_test_scalar,
    y_test_pred_scalar,
    scalar_target_name,
)


# %%
# Randomly choose a list of simulation indices for plotting.
selected_simu_indices = random_simu_indices(scalar_test_set, 5, seed=random_seed)
fig_random_scalar_design_curves = plot_design_prediction_curves(
    scalar_model,
    scalar_test_set,
    scalar_il_loader,
    selected_simu_indices,
)

# %% [markdown]
# The predicted median follows the broad frequency trend, showing that Ridge learns
# the central response. However, the predicted percentile band is consistently much
# narrower than the true band, particularly toward the upper end of the frequency
# range. The model therefore behaves like an average-response predictor and
# underestimates design-to-design variation.

# %% [markdown]
# ### 1.6 Plotting: Predicted Vs True IL Values
#
# The x-axis is true IL and the y-axis is predicted IL. The diagonal reference
# line shows ideal predictions.

# %%
fig_scalar_scatter = plot_scalar_true_vs_predicted(
    y_test_scalar,
    y_test_pred_scalar,
    scalar_target_name,
)

# %% [markdown]
# Predictions occupy a narrower range than the true IL values and are compressed
# toward the centre of the distribution. The largest insertion-loss cases are
# systematically underestimated. This agrees with the percentile-band plot: the
# scalar Ridge model captures the broad trend but smooths away extreme
# design-frequency responses.
#

# %% [markdown]
# ### 1.7 Plotting: MAE By Frequency
#
# Group test rows by `FREQ_GHZ`:
#
# $$
#     MAE(f_k) = \frac{1}{N_k} \sum_i |IL_{i,k} - \widehat{IL}_{i,k}|
# $$
#
# This gives a frequency-dependent error curve.

# %%
fig_scalar_mae_frequency = plot_scalar_mae_by_frequency(
    scalar_test_set.dataframe,
    y_test_scalar,
    y_test_pred_scalar,
    scalar_target_name,
)

# %% [markdown]
# MAE rises smoothly across the frequency range rather than being dominated by a few
# isolated peaks. Prediction is therefore substantially easier at the low-frequency
# end and progressively harder toward the high-frequency end. Together with the
# compressed scatter and narrow prediction band, this is consistent with systematic
# underfitting by the global linear model rather than a small number of anomalous
# samples.

# %%
scalar_runner.manager.save_figure(
    fig_scalar_distribution,
    "prediction_band_by_frequency.png",
)
scalar_runner.manager.save_figure(
    fig_random_scalar_design_curves,
    "selected_design_curves_insertion_loss_db.png",
)
scalar_runner.manager.save_figure(
    fig_scalar_scatter,
    "true_vs_predicted.png",
)
scalar_runner.manager.save_figure(
    fig_scalar_mae_frequency,
    "mae_by_frequency.png",
)

scalar_artifact_paths = scalar_runner.persist(
    data_interface=scalar_data_interface,
    metric_units={"MAE": "dB", "RMSE": "dB"},
)
scalar_manifest = read_json(scalar_artifact_paths["manifest"])

print(f"Scalar Ridge run directory: {scalar_runner.manager.run_dir}")
print("Scalar Ridge artifacts:")
for artifact_name, artifact_path in scalar_artifact_paths.items():
    print(f"- {artifact_name}: {artifact_path}")
print("\nScalar Ridge manifest figures:", scalar_manifest.get("figures", {}))

# %% [markdown]
# The scalar experiment is now complete. Its figures and predictions are retained,
# but the large scalar dataset and feature arrays are released before loading the
# vector experiment. This keeps the two experiments independent without holding two
# complete sets of dataframe views in memory at the same time.

# %%
del scalar_train_set, scalar_val_set, scalar_test_set
del X_train_scalar, X_val_scalar, X_test_scalar
del y_train_scalar, y_val_scalar, y_test_scalar

# %% [markdown]
# ## 2. Vector Insertion Loss Baseline
#
# This baseline trains one multi-output non-neural regressor to predict a vector
# of six insertion-loss values from the same design-frequency feature vector.
# Full IL curves are reconstructed by evaluating the trained vector model across
# all frequency points for the same design.

# %% [markdown]
# The vector and polynomial experiments use a separate set of train, validation,
# and test views configured with the six-target vector IL loader. Their cache files
# are independent from the scalar experiment: a scalar cache can be rebuilt or
# removed without affecting vector model development.

# %%
vector_train_set, vector_val_set, vector_test_set = load_pointwise_datasets(
    cfg.preprocessing.freq_expanded_csv,
    target_loader=vector_il_loader,
    cache=True,
)
vector_data_interface = {
    "dataset_name": cfg.dataset.name,
    "input_features": vector_train_set.feature_columns,
    "target_names": vector_target_names,
    "target_scope": "vector",
    "target_units": "dB",
    "target_representation": "insertion_loss_db",
}

print(f"Number of training   samples: {len(vector_train_set)}")
print(f"Number of validation samples: {len(vector_val_set)}")
print(f"Number of test       samples: {len(vector_test_set)}")

# %%
# pylint: disable=invalid-name
X_train, Y_train = vector_train_set.features, vector_train_set.targets
X_val,   Y_val   = vector_val_set.features,   vector_val_set.targets
X_test,  Y_test  = vector_test_set.features,  vector_test_set.targets
# pylint: enable=invalid-name

print(f"Shape of training   features: {X_train.shape}")
print(f"Shape of validation features: {X_val.shape}")
print(f"Shape of test       features: {X_test.shape}")
print(f"Shape of training   targets: {Y_train.shape}")
print(f"Shape of validation targets: {Y_val.shape}")
print(f"Shape of test       targets: {Y_test.shape}")
for split_name, target_values in {
    "training": Y_train,
    "validation": Y_val,
    "test": Y_test,
}.items():
    if (target_values <= 0.0).any():
        raise RuntimeError(f"{split_name.title()} vector IL targets must be positive.")

# %%
print(f"Vector Touchstone cache: {vector_il_loader.cache_info()}")
vector_il_loader.clear_cache()
print(f"Vector Touchstone cache after clearing: {vector_il_loader.cache_info()}")

# %% [markdown]
# ### 2.1 Input-Output Definition
#
# The input is the same as the scalar baseline:
#
# $$
#   \mathbf{x}_{i,k} = [\text{design parameters}, f_k]
# $$
#
# The output is a six-value IL vector:
#
# $$
# Y_{i,k} =
# [
# IL_{7,1}(f_k),
# IL_{8,2}(f_k),
# IL_{9,3}(f_k),
# IL_{10,4}(f_k),
# IL_{11,5}(f_k),
# IL_{12,6}(f_k)
# ]
# $$
#
# Therefore:
#
# $$
# Y \in \mathbb{R}^{N \times 6}
# $$

# %%
print(f"Target names of vector model: {vector_target_names}")

# %% [markdown]
# ### 2.2 Target Loading
#
# The six dB IL targets are loaded with:
#
# ```python
# TouchstoneLoader(mode="vector", representation="il", config=cfg)
# ```
#
# On a cold load, the loader accesses Touchstone data while filling `Y_train`,
# `Y_val`, and `Y_test`, then each split is saved as an NPZ cache. On a warm load,
# the arrays come directly from the corresponding cache. Keeping the arrays in
# memory after loading is a deliberate trade-off for the scikit-learn baselines.

# %% [markdown]
# ### 2.3 Model Training
#
# The vector baseline uses one multi-output Ridge regression model. Validation MAE
# averaged across all six output columns selects the regularisation strength. It
# uses the same `StandardScaler()` plus `Ridge(alpha=alpha)` pipeline reported in
# the scalar modelling section.

# %%
vector_ridge_config = cfg.models.vector_ridge
vector_runner = ModelRunRunner(
    cfg,
    VectorRidgeModel.from_config(vector_ridge_config),
)
vector_model = vector_runner.train(X_train, Y_train, X_val, Y_val)

vector_alpha_results = vector_model.validation_results
best_vector_alpha = vector_model.best_alpha
if vector_alpha_results is None or best_vector_alpha is None:
    raise RuntimeError("Vector Ridge model did not record validation results.")

print("Vector Ridge validation sweep:", vector_alpha_results, sep="\n")
print(f"Best vector alpha: {best_vector_alpha:g}")
print(f"Selected vector preprocessing: {vector_model.pipeline.named_steps['scaler']}")
print(f"Selected vector regressor: {vector_model.pipeline.named_steps['model']}")

# %% [markdown]
# Validation performance is nearly unchanged across the tested alpha values, and the
# selected setting has only a marginal advantage. Regularisation strength is therefore
# not the main limitation. The model predicts all six outputs successfully, but the
# nearly flat sweep suggests that further alpha tuning will not overcome the limited
# capacity of a linear relationship.
#
# `StandardScaler()` remains appropriate because Ridge is sensitive to feature scale.
# It places geometry values, material properties, and frequency on comparable
# numerical scales before fitting.
#

# %% [markdown]
# ### 2.4 Vector Ridge Model Evaluation

# %%
vector_validation_metrics = vector_runner.validate(X_val, Y_val)
vector_test_metrics = vector_runner.test(X_test, Y_test)

Y_val_pred = vector_model.predict(X_val)  # pylint: disable=invalid-name
Y_test_pred = vector_model.predict(X_test)  # pylint: disable=invalid-name

vector_metrics = pd.DataFrame(
    [
        {"split": "validation", **vector_validation_metrics},
        {"split": "test", **vector_test_metrics},
    ]
)
per_target_validation_metrics = per_target_metrics(
    Y_val,
    Y_val_pred,
    vector_target_names,
)
per_target_test_metrics = per_target_metrics(
    Y_test,
    Y_test_pred,
    vector_target_names,
)
vector_per_target_run_metrics = per_target_split_metrics(
    per_target_validation_metrics,
    per_target_test_metrics,
)

print("Overall vector metrics:", vector_metrics, sep="\n")
print("\nPer-port-pair test metrics:", per_target_test_metrics, sep="\n")

# %% [markdown]
# Test performance is modestly worse than validation performance, with the RMSE gap
# more visible than the MAE gap. This again indicates that difficult held-out cases
# affect the squared-error measure more strongly. Per-target errors are closely
# grouped, so no individual through path dominates the aggregate result. The common
# limitation across outputs is consistent with underfitting by the linear model class.
#

# %% [markdown]
# ### 2.5 Plot Vector IL Distributions Across Test Designs

# %%
fig_vector_distributions = plot_vector_prediction_bands_by_frequency(
    vector_test_set.dataframe,
    Y_test,
    Y_test_pred,
    vector_target_names,
)

# %% [markdown]
# All six paths show the same broad behaviour. Their predicted medians follow the
# central frequency trends, but their percentile bands are much narrower than the
# corresponding true bands. The mismatch becomes more evident at higher frequencies,
# where design-to-design variation is strongest. Vector Ridge is therefore a stable
# multi-output baseline, but it still predicts an overly average-like response.

# %%
fig_random_vector_design_curves = plot_design_prediction_curves(
    vector_model,
    vector_test_set,
    vector_il_loader,
    selected_simu_indices,
)

# %% [markdown]
# ### 2.6 Plot Vector Predicted Vs True Scatter
#
# The diagonal means perfect prediction:
#
# $$
# \widehat{y}=y
# $$

# %%
fig_vector_scatter = plot_vector_true_vs_predicted(
    Y_test,
    Y_test_pred,
    vector_target_names,
)

# %% [markdown]
# Predictions are compressed toward the mean for every output, and strong-attenuation
# cases are systematically underestimated. The similar scatter pattern across all six
# paths agrees with the closely grouped per-target metrics: the main problem is shared
# model capacity rather than one uniquely difficult port pair.

# %% [markdown]
# ### 2.7 Plot Vector MAE By Frequency
#
# For each target column:
#
# $$
#     MAE_j(f_k) = mean_i |IL_j(i,k) - \widehat{IL}_j(i,k)|
# $$

# %%
fig_vector_mae_frequency = plot_vector_mae_by_frequency(
    vector_test_set.dataframe,
    Y_test,
    Y_test_pred,
    vector_target_names,
)

# %% [markdown]
# MAE increases smoothly with frequency for all six paths, and the curves remain close
# to one another. This confirms that the high-frequency weakness is common across the
# configured targets. The systematic trend, rather than isolated error spikes, further
# supports the conclusion that a more expressive nonlinear model is needed.
#

# %%
vector_runner.manager.save_figure(
    fig_vector_distributions,
    "prediction_bands_by_frequency.png",
)
vector_runner.manager.save_figure(
    fig_random_vector_design_curves,
    "selected_design_curves_insertion_loss_db.png",
)
vector_runner.manager.save_figure(
    fig_vector_scatter,
    "true_vs_predicted.png",
)
vector_runner.manager.save_figure(
    fig_vector_mae_frequency,
    "mae_by_frequency.png",
)

vector_artifact_paths = vector_runner.persist(
    data_interface=vector_data_interface,
    extra_metrics={"per_target": vector_per_target_run_metrics},
    metric_units={"MAE": "dB", "RMSE": "dB"},
)
vector_manifest = read_json(vector_artifact_paths["manifest"])

print(f"Vector Ridge run directory: {vector_runner.manager.run_dir}")
print("Vector Ridge artifacts:")
for artifact_name, artifact_path in vector_artifact_paths.items():
    print(f"- {artifact_name}: {artifact_path}")
print("\nVector Ridge manifest figures:", vector_manifest.get("figures", {}))

# %% [markdown]
# ## 3. Polynomial Vector Baseline
#
# This baseline keeps the same vector target definition as the vector Ridge model,
# but expands each input feature with powers before fitting a regularised linear
# model. The configured degree and regularisation grids are evaluated below, then
# the degree and Ridge regularisation strength with the lowest validation MAE are
# selected. For one feature, the powers-only expansion is:
#
# $$
# x_j \rightarrow [x_j, x_j^2, \ldots, x_j^D]
# $$
#
# Unlike full polynomial features, this does not create cross terms such as
# $x_1 x_2$. That keeps the feature matrix much smaller while still allowing each
# design or frequency feature to learn smooth nonlinear curvature.

# %% [markdown]
# ### 3.1 Train Polynomial Degree Sweep

# %%
polynomial_ridge_config = cfg.models.polynomial_ridge
polynomial_runner = ModelRunRunner(
    cfg,
    PolynomialModel.from_config(polynomial_ridge_config),
)
polynomial_model = polynomial_runner.train(X_train, Y_train, X_val, Y_val)

polynomial_validation_results = polynomial_model.validation_results
best_polynomial_degree = polynomial_model.best_degree
best_polynomial_alpha = polynomial_model.best_alpha
if (
    polynomial_validation_results is None
    or best_polynomial_degree is None
    or best_polynomial_alpha is None
):
    raise RuntimeError("Polynomial model did not record validation results.")

polynomial_step = polynomial_model.pipeline.named_steps["polynomial"]
expanded_feature_count = polynomial_step.n_output_features_
print(f"Polynomial validation sweep:\n{polynomial_validation_results}")

# %%
print(f"Best polynomial degree: {best_polynomial_degree}")
print(f"Best polynomial alpha: {best_polynomial_alpha:g}")
print(f"Selected expanded polynomial feature count: {expanded_feature_count}")
print(f"Selected polynomial pipeline:\n{polynomial_model.pipeline}")

# %% [markdown]
# The selected powers-only expansion gives the best validation result in the
# configured sweep, but the differences across degrees and regularisation strengths
# are small. This indicates that the additional nonlinear terms provide only a modest
# advantage and that regularisation tuning is not the main performance constraint.
# The held-out comparison below tests whether the small validation gain generalises.

# %% [markdown]
# ### 3.2 Polynomial Evaluation

# %%
polynomial_validation_metrics = polynomial_runner.validate(X_val, Y_val)
polynomial_test_metrics = polynomial_runner.test(X_test, Y_test)

Y_val_pred_poly = polynomial_model.predict(X_val)  # pylint: disable=invalid-name
Y_test_pred_poly = polynomial_model.predict(X_test)  # pylint: disable=invalid-name

polynomial_metrics = pd.DataFrame(
    [
        {"split": "validation", **polynomial_validation_metrics},
        {"split": "test", **polynomial_test_metrics},
    ]
)

print(f"Polynomial vector metrics:\n{polynomial_metrics}")

# %%
model_comparison = pd.DataFrame(
    [
        {"model": "Vector Ridge", **regression_metrics(Y_test, Y_test_pred)},
        {"model": "Polynomial", **regression_metrics(Y_test, Y_test_pred_poly)},
    ]
)

print(f"Overall model comparison:\n{model_comparison}")

# %% [markdown]
# Polynomial Ridge and Vector Ridge are effectively tied on the test set. Polynomial
# Ridge has a marginally lower MAE but a slightly higher RMSE, so neither model has a
# clear overall advantage. The validation gain therefore does not translate into a
# meaningful held-out aggregate improvement.

# %%
per_target_vector_comparison = per_target_test_metrics.copy()
per_target_vector_comparison.insert(0, "model", "Vector Ridge")

per_target_polynomial_metrics = per_target_metrics(
    Y_test,
    Y_test_pred_poly,
    vector_target_names,
)
per_target_polynomial_validation_metrics = per_target_metrics(
    Y_val,
    Y_val_pred_poly,
    vector_target_names,
)
polynomial_per_target_run_metrics = per_target_split_metrics(
    per_target_polynomial_validation_metrics,
    per_target_polynomial_metrics,
)
per_target_polynomial_comparison = per_target_polynomial_metrics.copy()
per_target_polynomial_comparison.insert(0, "model", "Polynomial")
per_target_model_comparison = pd.concat(
    [per_target_vector_comparison, per_target_polynomial_comparison],
    ignore_index=True,
)

print(f"Per-target model comparison:\n{per_target_model_comparison}")

# %% [markdown]
# Per-target changes are small and mixed across the six through paths rather than
# being driven by one output. Polynomial features slightly improve some paths and
# slightly worsen others, but they do not resolve the common modelling limitation.
#

# %% [markdown]
# ### 3.3 Plot Polynomial IL Distributions Across Test Designs

# %%
fig_polynomial_distributions = plot_vector_prediction_bands_by_frequency(
    vector_test_set.dataframe,
    Y_test,
    Y_test_pred_poly,
    vector_target_names,
    model_name="Polynomial",
)

# %%
# Randomly inspect held-out test designs with fresh Polynomial Ridge predictions.
fig_random_polynomial_design_curves = plot_design_prediction_curves(
    polynomial_model,
    vector_test_set,
    vector_il_loader,
    selected_simu_indices,
)

# %% [markdown]
# Polynomial Ridge produces slightly more realistic median curvature than Vector
# Ridge, showing that the added powers capture some nonlinear frequency dependence.
# However, the predicted percentile bands remain much narrower than the true bands.
# The feature expansion therefore refines the central trend without recovering the
# full design-to-design response spread.
#

# %% [markdown]
# ### 3.4 Compare Ridge And Polynomial MAE By Frequency

# %%
fig_model_mae_comparison_frequency = plot_model_mae_comparison_by_frequency(
    vector_test_set.dataframe,
    Y_test,
    {
        "Vector Ridge": Y_test_pred,
        "Polynomial": Y_test_pred_poly,
    },
    vector_target_names,
)

# %% [markdown]
# The Polynomial and Vector Ridge MAE curves almost overlap. Polynomial Ridge is
# slightly better in some frequency regions, but both models retain the same smooth
# increase in error toward higher frequencies. The expansion therefore provides a
# limited local refinement without changing the main frequency-dependent weakness.
#

# %%
polynomial_runner.manager.save_figure(
    fig_polynomial_distributions,
    "prediction_bands_by_frequency.png",
)
polynomial_runner.manager.save_figure(
    fig_random_polynomial_design_curves,
    "selected_design_curves_insertion_loss_db.png",
)
polynomial_runner.manager.save_figure(
    fig_model_mae_comparison_frequency,
    "ridge_polynomial_mae_by_frequency.png",
)

polynomial_artifact_paths = polynomial_runner.persist(
    data_interface=vector_data_interface,
    extra_metrics={"per_target": polynomial_per_target_run_metrics},
    metric_units={"MAE": "dB", "RMSE": "dB"},
)
polynomial_manifest = read_json(polynomial_artifact_paths["manifest"])

print(f"Polynomial Ridge run directory: {polynomial_runner.manager.run_dir}")
print("Polynomial Ridge artifacts:")
for artifact_name, artifact_path in polynomial_artifact_paths.items():
    print(f"- {artifact_name}: {artifact_path}")
print(
    "\nPolynomial Ridge manifest figures:",
    polynomial_manifest.get("figures", {}),
)

# %% [markdown]
# ### 3.5 Interpreting Polynomial Ridge Capacity
#
# Polynomial Ridge changes the validation and test results only modestly and does not
# recover the full predicted response spread. This is understandable because it is
# not simply fitting one curve; it learns a global relationship across many designs,
# frequencies, and output samples.
#
# %% [markdown]
# #### 1. The model is fitting many designs at once, not one curve
#
# For one fixed PCB design, a polynomial curve may fit the frequency response
# reasonably well. However, in this experiment, the model is learning the
# mapping:
#
# $$
# (\mathbf{u}, f) \rightarrow IL_{7,1}(f)
# $$
#
# Here, $\mathbf{u}$ represents the PCB design-parameter vector, and $(f)$
# represents frequency. Therefore, the model must learn not only how `IL_S7_1_DB`
# changes with frequency, but also how the curve changes when the PCB geometry
# and material parameters change. This is much harder than fitting one
# frequency-response curve for one design.
#
# %% [markdown]
# #### 2. Polynomial Ridge uses expanded features
#
# The Polynomial Ridge model can be written as:
#
# $$
# \hat{y}=\beta_0+\sum_{r=1}^{R}\beta_r\phi_r(\mathbf{u},f)
# $$
#
#
# where:
#
# * $\hat{y}$ is the predicted target value, such as predicted `IL_S7_1_DB`.
# * $\beta_0$ is the intercept term.
# * $R$ is the total number of expanded polynomial features.
# * $\beta_r$ is the learned coefficient for the (r)-th polynomial feature.
# * $\phi_r(\mathbf{u},f)$ is the (r)-th transformed feature generated
#   from design parameters and frequency.
#
# The selected degree and resulting expanded feature count are printed by the
# training cell above. This powers-only expansion gives the model more flexibility
# than plain Ridge while remaining compact because it does not add interactions
# between different input features.
#
# %% [markdown]
# #### 3. The role of regularisation
#
# Ridge regression penalises large coefficients:
#
# $$
# \min_{\boldsymbol{\beta}}
# \sum_{q=1}^{N}
# \left(\hat{y}^{(q)}-y^{(q)}\right)^2
# +
# \alpha
# \sum_{r=1}^{R}\beta_r^2
# $$
#
#
# Here, $\alpha$ controls the strength of regularisation. Validation error changes
# little across the tested values, so the smooth and average-like predictions are
# mainly a limitation of the powers-only representation rather than the selected
# regularisation strength.
#
# %% [markdown]
# #### 4. High-frequency behaviour is more complex than polynomial curvature
#
# At higher frequencies, the S-parameter response may be affected by stronger
# nonlinear effects, coupling, resonances, and sensitivity to small geometry
# changes. A low-capacity polynomial model may not represent these behaviours
# well.
#
# The median response is slightly less constrained than with Vector Ridge, but the
# predicted percentile band remains too narrow. Added polynomial curvature alone is
# therefore insufficient for the more complex response variation.
#
# %% [markdown]
# #### 5. The loss function encourages average predictions
#
# The model is trained to minimise the overall error across many designs and
# frequency points. When some extreme deep-loss samples are difficult to
# predict, the model can reduce total error by staying close to the central
# trend rather than fitting those extreme cases.
#
# The observed median and percentile bands show this central-tendency effect:
# Polynomial Ridge improves the median shape slightly but still misses much of the
# design-to-design variation.
#
# %% [markdown]
# #### Conclusion
#
# Polynomial Ridge is a modest refinement of Vector Ridge rather than a decisive
# improvement. It adds useful nonlinear curvature but remains a compact global model
# with average-like predictions. The next non-neural check uses a tree-based model to
# test whether local design-frequency partitions better represent the response.
#

# %% [markdown]
# ## 4. Random Forest Vector Baseline
#
# Random Forest regression is added as the final non-neural nonlinear baseline.
# Unlike Ridge and Polynomial Ridge, it does not fit one global linear model.
# Instead, it averages predictions from many decision trees, each of which
# partitions the design-frequency input space into local regions.
#
# The purpose of this section is to test whether the curvature missed by
# Polynomial Ridge is mainly caused by limited nonlinear model capacity. The
# model uses the same vector target, train/validation/test split, and raw input
# arrays as the previous vector baselines. The tree count and candidate leaf
# settings are configured in `configs/default.json` under `models.random_forest`.

# %% [markdown]
# ### 4.1 Train Random Forest Baseline

# %%
random_forest_config = cfg.models.random_forest
random_forest_runner = ModelRunRunner(
    cfg,
    RandomForestModel.from_config(random_forest_config),
)
random_forest_model = random_forest_runner.train(X_train, Y_train, X_val, Y_val)

random_forest_validation_results = random_forest_model.validation_results
if random_forest_validation_results is None:
    raise RuntimeError("Random Forest model did not record validation results.")

print(f"Random Forest validation sweep:\n{random_forest_validation_results}\n")
print(f"Selected Random Forest model:\n{random_forest_model.regressor}")

# %% [markdown]
# ### 4.2 Random Forest Evaluation

# %%
random_forest_train_metrics = random_forest_model.evaluate(X_train, Y_train)
random_forest_validation_metrics = random_forest_runner.validate(X_val, Y_val)
random_forest_test_metrics = random_forest_runner.test(X_test, Y_test)

Y_val_pred_rf = random_forest_model.predict(X_val)  # pylint: disable=invalid-name
Y_test_pred_rf = random_forest_model.predict(X_test)  # pylint: disable=invalid-name

random_forest_metrics = pd.DataFrame(
    [
        {"split": "train", **random_forest_train_metrics},
        {"split": "validation", **random_forest_validation_metrics},
        {"split": "test", **random_forest_test_metrics},
    ]
)
per_target_random_forest_validation_metrics = per_target_metrics(
    Y_val,
    Y_val_pred_rf,
    vector_target_names,
)
per_target_random_forest_metrics = per_target_metrics(
    Y_test,
    Y_test_pred_rf,
    vector_target_names,
)
random_forest_per_target_run_metrics = per_target_split_metrics(
    per_target_random_forest_validation_metrics,
    per_target_random_forest_metrics,
)

print(f"Random Forest vector metrics:\n{random_forest_metrics}")

# %% [markdown]
# Random Forest fits the training data much more closely than either held-out split.
# The large train-validation gap is clear evidence of overfitting: the forest captures
# training designs well but does not generalise that pointwise accuracy to unseen PCB
# designs. Validation and test performance are much closer to one another, so the main
# gap arises between fitted and unseen designs.

# %%
shared_target_model_comparison = pd.DataFrame(
    [
        {
            "model": "Scalar Ridge",
            "target": scalar_target_name,
            **regression_metrics(
                Y_test[:, scalar_target_index],
                y_test_pred_scalar,
            ),
        },
        {
            "model": "Vector Ridge",
            "target": scalar_target_name,
            **regression_metrics(
                Y_test[:, scalar_target_index],
                Y_test_pred[:, scalar_target_index],
            ),
        },
        {
            "model": "Polynomial Ridge",
            "target": scalar_target_name,
            **regression_metrics(
                Y_test[:, scalar_target_index],
                Y_test_pred_poly[:, scalar_target_index],
            ),
        },
        {
            "model": "Random Forest",
            "target": scalar_target_name,
            **regression_metrics(
                Y_test[:, scalar_target_index],
                Y_test_pred_rf[:, scalar_target_index],
            ),
        },
    ]
)

print(f"\n{scalar_target_name} model comparison:\n{shared_target_model_comparison}")

# %% [markdown]
# Random Forest has worse held-out MAE and RMSE than both Ridge-style vector models,
# and this weakness is consistent across the configured through paths. It is therefore
# not the strongest model by aggregate pointwise accuracy. However, it remains a useful
# nonlinear capacity check because the distribution and design-curve plots reveal
# response-shape behaviour that aggregate errors do not capture.

# %% [markdown]
# ### 4.3 Plot Random Forest Predicted Vs True Values

# %%
fig_random_forest_scatter = plot_vector_true_vs_predicted(
    Y_test,
    Y_test_pred_rf,
    vector_target_names,
    model_name="Random Forest",
)

# %% [markdown]
# ### 4.4 Plot Random Forest IL Distributions Across Test Designs

# %%
fig_random_forest_distributions = plot_vector_prediction_bands_by_frequency(
    vector_test_set.dataframe,
    Y_test,
    Y_test_pred_rf,
    vector_target_names,
    model_name="Random Forest",
)

# %% [markdown]
# Random Forest produces wider predicted percentile bands than the Ridge-style
# baselines and follows the true design-to-design spread more closely. Some variation,
# especially toward higher frequencies, is still underestimated. This is a qualitative
# improvement in distribution shape despite the worse aggregate MAE and RMSE, revealing
# a trade-off between pointwise accuracy and response-spread fidelity.
#
# %% [markdown]
# ### 4.5 Compare Vector-Model MAE By Frequency

# %%
fig_vector_model_mae_comparison_frequency = plot_model_mae_comparison_by_frequency(
    vector_test_set.dataframe,
    Y_test,
    {
        "Vector Ridge": Y_test_pred,
        "Polynomial Ridge": Y_test_pred_poly,
        "Random Forest": Y_test_pred_rf,
    },
    vector_target_names,
)

# %% [markdown]
# ### 4.6 Compare Held-Out Design Curves
#
# This is the main qualitative plot for the curvature question. It compares the
# true held-out curves against Polynomial Ridge and Random Forest predictions
# for the same selected test designs.

# %%
fig_polynomial_random_forest_design_comparison = (
    plot_design_model_comparison_curves(
        {
            "Polynomial Ridge": polynomial_model,
            "Random Forest": random_forest_model,
        },
        vector_test_set,
        vector_il_loader,
        selected_simu_indices,
    )
)

# %% [markdown]
# Random Forest represents substantially more local curvature than Polynomial Ridge
# and follows some bends, peaks, and recovery regions more closely. The improvement is
# uneven: some features remain misplaced or incorrectly scaled on the selected
# held-out designs. This richer shape is not accompanied by better aggregate pointwise
# error, so it is insufficient evidence of better generalisation.

# %% [markdown]
# ## 5. Four-Model Comparison On IL_S7_1_DB
#
# The scalar Ridge model only predicts `IL_S7_1_DB`, so the cleanest comparison is
# to evaluate all four fitted models on that shared target only. The scalar
# model contributes its direct prediction. The Vector Ridge, Polynomial Ridge,
# and Random Forest models contribute only their `IL_S7_1_DB` output column, even
# though they were trained on all six outputs.

# %%
shared_target_name = scalar_target_name
shared_target_index = scalar_target_index

shared_target_true = Y_test[:, shared_target_index]
shared_target_predictions = {
    "Scalar Ridge": y_test_pred_scalar,
    "Vector Ridge": Y_test_pred[:, shared_target_index],
    "Polynomial Ridge": Y_test_pred_poly[:, shared_target_index],
    "Random Forest": Y_test_pred_rf[:, shared_target_index],
}

# %% [markdown]
# First, compare the four `IL_S7_1_DB` MAE curves by frequency.

# %%
fig_s7_four_model_mae_frequency = plot_shared_target_mae_comparison(
    vector_test_set.dataframe,
    shared_target_true,
    shared_target_predictions,
    shared_target_name,
)

# %% [markdown]
# Second, compare the true distribution curve against the predicted distribution
# from each model. The true curve uses the same median and 10th-90th percentile
# band in all cases, while each model contributes its own predicted median and
# band. This makes it easier to see whether Random Forest improves the
# `IL_S7_1_DB` curve shape compared with the Ridge-style baselines.

# %%
fig_s7_four_model_distributions = plot_shared_target_prediction_bands(
    vector_test_set.dataframe,
    shared_target_true,
    shared_target_predictions,
    shared_target_name,
)

# %% [markdown]
# Random Forest does not improve held-out pointwise error on the shared target, even
# though it is the most flexible non-neural model considered here. Scalar and Vector
# Ridge perform very similarly, showing that multi-output fitting provides convenience
# and consistency but little shared-output benefit for this target. Polynomial Ridge
# changes the shared-target result only slightly and remains effectively tied with the
# linear Ridge predictions.
#
# The qualitative comparison tells a different story. Random Forest represents local
# curvature and the response distribution more realistically, while the Ridge-style
# models remain smoother and more compressed. Its large train-to-held-out gap shows
# that this additional flexibility currently overfits rather than improving pointwise
# generalisation.
#
# Overall, no single non-neural baseline wins every criterion. The Ridge models retain
# stronger aggregate accuracy, Polynomial Ridge adds limited nonlinear flexibility,
# and Random Forest better represents curve shape and design-to-design spread. This
# motivates a model class that can preserve the richer response structure while
# improving held-out pointwise accuracy.
#

# %%
random_forest_runner.manager.save_figure(
    fig_random_forest_scatter,
    "true_vs_predicted.png",
)
random_forest_runner.manager.save_figure(
    fig_random_forest_distributions,
    "prediction_bands_by_frequency.png",
)
random_forest_runner.manager.save_figure(
    fig_vector_model_mae_comparison_frequency,
    "vector_model_mae_by_frequency.png",
)
random_forest_runner.manager.save_figure(
    fig_polynomial_random_forest_design_comparison,
    "polynomial_random_forest_design_comparison.png",
)
random_forest_runner.manager.save_figure(
    fig_s7_four_model_mae_frequency,
    "s7_1_four_model_mae_by_frequency.png",
)
random_forest_runner.manager.save_figure(
    fig_s7_four_model_distributions,
    "s7_1_four_model_prediction_bands.png",
)

random_forest_artifact_paths = random_forest_runner.persist(
    data_interface=vector_data_interface,
    extra_metrics={"per_target": random_forest_per_target_run_metrics},
    metric_units={"MAE": "dB", "RMSE": "dB"},
)
random_forest_manifest = read_json(random_forest_artifact_paths["manifest"])

print(f"Random Forest run directory: {random_forest_runner.manager.run_dir}")
print("Random Forest artifacts:")
for artifact_name, artifact_path in random_forest_artifact_paths.items():
    print(f"- {artifact_name}: {artifact_path}")
print(
    "\nRandom Forest manifest figures:",
    random_forest_manifest.get("figures", {}),
)

# %% [markdown]
# ## Persisted Output Summary
#
# The notebook now writes into the planned output hierarchy:
#
# - `outputs/runs/<run_id>/` for immutable run artifacts.
# - `outputs/models/*.json` for latest and selected model pointers.
# - `outputs/benchmarks/*.csv` for comparison rows.

# %%
latest_model_registry = read_json(cfg.paths.models / "latest.json")
selected_model_registry = read_json(cfg.paths.models / "selected.json")
s7_latest_benchmark_path = (
    cfg.paths.benchmarks / "s7_1_insertion_loss_db_latest.csv"
)
s7_selected_benchmark_path = (
    cfg.paths.benchmarks / "s7_1_insertion_loss_db_selected.csv"
)
vector_latest_benchmark_path = (
    cfg.paths.benchmarks / "vector_insertion_loss_db_latest.csv"
)
vector_selected_benchmark_path = (
    cfg.paths.benchmarks / "vector_insertion_loss_db_selected.csv"
)

print("Latest model registry entries:")
print(sorted(latest_model_registry.get("models", {})))
print("\nSelected model registry entries:")
print(sorted(selected_model_registry.get("models", {})))

if s7_latest_benchmark_path.is_file():
    print("\nLatest S7_1 benchmark:")
    display(pd.read_csv(s7_latest_benchmark_path).style.hide(axis="index"))

if s7_selected_benchmark_path.is_file():
    print("\nSelected S7_1 benchmark:")
    display(pd.read_csv(s7_selected_benchmark_path).style.hide(axis="index"))

if vector_latest_benchmark_path.is_file():
    print("\nLatest vector benchmark:")
    display(pd.read_csv(vector_latest_benchmark_path).style.hide(axis="index"))

if vector_selected_benchmark_path.is_file():
    print("\nSelected vector benchmark:")
    display(pd.read_csv(vector_selected_benchmark_path).style.hide(axis="index"))
