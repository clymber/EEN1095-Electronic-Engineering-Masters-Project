# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: meng
#     language: python
#     name: python3
# ---

# %%
"""
Train non-neural scalar and vector insertion-loss baseline models.
"""
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sparam_surrogate.config import load_config, relative_to_project_root
from sparam_surrogate.data import DLDataset, TouchstoneLoader
from sparam_surrogate.utils.non_neural_modelling_utils import (
    RIDGE_ALPHA_GRID,
    fit_ridge_with_validation,
    per_target_metrics,
    plot_scalar_mae_by_frequency,
    plot_scalar_prediction_band_by_frequency,
    plot_scalar_residual_histogram,
    plot_scalar_residual_vs_frequency,
    plot_scalar_true_vs_predicted,
    plot_vector_mae_by_frequency,
    plot_vector_prediction_bands_by_frequency,
    plot_vector_residual_histograms,
    plot_vector_residual_vs_frequency,
    plot_vector_true_vs_predicted,
    regression_metrics,
)

DS_NAME = "linkOn8CavityStackBetween10x10Array_19_08_2021"
CLEANED_CSV = "sipi_dataset_cleaned.csv"

# %% [markdown]
# # Non-Neural Network Modelling
#
# In this notebook, I train non-neural baseline models to predict insertion-loss
# targets derived from S-parameters. The first baseline predicts one scalar IL
# value for one port pair. The second baseline predicts one vector containing the
# six configured through-path IL values.
#
# The cleaned CSV stores only design features, frequency, split labels, simulation
# indices, and Touchstone paths. The `TouchstoneLoader` reads Touchstone files on
# demand, then this notebook materializes the target arrays before fitting because
# the scikit-learn Ridge baselines expect in-memory NumPy arrays.

# %%
# Import and setup
cfg = load_config()
raw_data_dir = Path(cfg["paths"]["raw_data"]) / DS_NAME
processed_dir = Path(cfg["paths"]["processed_data"])
port_pairs = tuple(tuple(pair) for pair in cfg["dataset"]["ports"])

il_loader = TouchstoneLoader("scalar", cfg, "db", 512) # Insertion Loss loader
target_names = tuple(il_loader.target_names)

print(f"Dataset: {DS_NAME}")
print(f"Raw data directory: {relative_to_project_root(raw_data_dir)}")
print(f"Processed directory: {relative_to_project_root(processed_dir)}")
print("Configured IL port pairs: ", *port_pairs)
print("Target names:", *target_names, sep=", ")

# %% [markdown]
# ## Shared Notebook Utilities
#
# Reusable helpers for feature extraction, target materialization from
# Touchstone files, validation sweeps, metrics, and plotting live in
# `src/sparam_surrogate/utils/non_neural_modelling_utils.py`.

# %% [markdown]
# ## Data Loading And Validation
#
# Load the cleaned CSV as three dataset views.
# Each view owns one split.

# %%
train_set, val_set, test_set = DLDataset.from_cleaned_csv(processed_dir / CLEANED_CSV)

# %% [markdown]
# Build the in-memory arrays used by scikit-learn.
#
# `X` comes from the cleaned CSV. It contains design parameters plus frequency.
# `Y` comes from Touchstone files. The loader reads those files on demand.

# %%
# pylint: disable=invalid-name
X_train, Y_train = train_set.features, train_set.load_targets(il_loader)
X_val, Y_val = val_set.features, val_set.load_targets(il_loader)
X_test, Y_test = test_set.features, test_set.load_targets(il_loader)
# pylint: disable=invalid-name

print(f"Training samples: {len(train_set)}")
print(f"Validation samples: {len(val_set)}")
print(f"Test samples: {len(test_set)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# %% [markdown]
# The Ridge baselines need full NumPy arrays.
# Keeping `X` and `Y` in memory also keeps the later plots simple.
#
# `TouchstoneLoader` caches files while building `Y`.
# Once the targets are loaded, the cache can be cleared.

# %%
print(f"Y_train shape: {Y_train.shape}")
print(f"Y_val shape: {Y_val.shape}")
print(f"Y_test shape: {Y_test.shape}")
print(f"Touchstone cache: {il_loader.cache_info()}")

il_loader.clear_cache()
print(f"Touchstone cache after clearing: {il_loader.cache_info()}")

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
# `RIDGE_ALPHA_GRID` is the small set of candidate Ridge regularisation strengths
# tested before choosing the final baseline model:

# %%
print("RIDGE_ALPHA_GRID = ", RIDGE_ALPHA_GRID)

# %% [markdown]
# In scikit-learn's `Ridge`, this value is called `alpha`. It has the same role as
# $\lambda$ in the Ridge objective above. A small `alpha` keeps the model close to
# ordinary least squares, while a large `alpha` shrinks coefficients more strongly
# and can reduce overfitting. The notebook fits one model for each candidate value,
# evaluates each candidate on the validation set, and selects the `alpha` with the
# lowest validation MAE. The test set is used only after this selection, so the test
# metrics remain held-out estimates.

# %% [markdown]
# ### 1.3.2 Scikit-Learn Model Design
#
# Each alpha candidate uses the same two-step scikit-learn pipeline:
#
# ```python
# Pipeline(
#     [
#         ("scaler", StandardScaler()),
#         ("model", Ridge(alpha=alpha)),
#     ]
# )
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
def build_ridge_pipeline(alpha: float) -> Pipeline:
    """Return the scaler-plus-Ridge pipeline used by each alpha candidate."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )

print("Example instantiated pipeline:")
print(build_ridge_pipeline(alpha=RIDGE_ALPHA_GRID[0]))

# %%
scalar_target_index = 0  # pylint: disable=invalid-name
scalar_target_name = target_names[scalar_target_index]

y_train_scalar = Y_train[:, scalar_target_index]
y_val_scalar = Y_val[:, scalar_target_index]
y_test_scalar = Y_test[:, scalar_target_index]

scalar_model, scalar_alpha_results = fit_ridge_with_validation(
    X_train,
    y_train_scalar,
    X_val,
    y_val_scalar,
    RIDGE_ALPHA_GRID,
    build_ridge_pipeline,
)

best_scalar_alpha = scalar_alpha_results.loc[
    scalar_alpha_results["MAE"].idxmin(),
    "alpha",
]

print(f"Scalar target: {scalar_target_name}")
print("Scalar Ridge validation sweep:")
print(scalar_alpha_results)
print(f"Best scalar alpha: {best_scalar_alpha:g}")
print(f"Selected scalar preprocessing: {scalar_model.named_steps['scaler']}")
print(f"Selected scalar regressor: {scalar_model.named_steps['model']}")

# %% [markdown]
# ### 1.4 Evaluate On Held-Out Test Data

# %%
y_val_pred_scalar = cast(np.ndarray, scalar_model.predict(X_val))
y_test_pred_scalar = cast(np.ndarray, scalar_model.predict(X_test))

scalar_metrics = pd.DataFrame(
    [
        {"split": "validation", **regression_metrics(y_val_scalar, y_val_pred_scalar)},
        {"split": "test", **regression_metrics(y_test_scalar, y_test_pred_scalar)},
    ]
)
print(scalar_metrics)

# %% [markdown]
# ### 1.5 Plotting: Scalar IL Distribution Across Test Designs
#
# This plot groups all held-out test rows by frequency and compares the true and
# predicted median curves, together with the 10th-90th percentile bands across
# design variants.

# %%
fig_scalar_distribution = plot_scalar_prediction_band_by_frequency(
    test_set.dataframe,
    y_test_scalar,
    y_test_pred_scalar,
    scalar_target_name,
)

# %% [markdown]
# **Interpretation.** The scalar Ridge baseline captures the central
# insertion-loss trend well, as shown by the close agreement between the true and
# predicted medians. However, its predicted percentile band is much narrower than
# the true band, especially at high frequency. This indicates that the model
# underfits design-specific variation and cannot represent frequency-dependent
# resonant behaviour or the widening response envelope across variants.

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
# **Interpretation.** The diagonal line represents perfect prediction:
# $\widehat{IL} = IL$. Points close to this line mean the Ridge model predicts the
# true insertion loss accurately. In this plot, most predictions are compressed
# into a relatively narrow dB range, while the true values extend to much lower
# insertion-loss values. This means the scalar linear baseline is underestimating
# the severity of deep-loss cases: when the true IL is very negative, the model
# tends to predict a less negative value.
#
# The dense horizontal cloud also suggests that this simple Ridge model captures
# the broad average trend but not sharp frequency-dependent behaviour such as
# resonances or notches. Therefore, this plot should be read as evidence that the
# scalar Ridge baseline is useful as a simple benchmark, but it is not expressive
# enough to reproduce the full dynamic range of the S7_1_DB response.

# %% [markdown]
# ### 1.7 Plotting: Residual Histogram
#
# $$
#     e_n = \hat{y}_n - y_n
# $$
#
# This shows whether the model is biased or has heavy-error tails.

# %%
fig_scalar_residual_hist = plot_scalar_residual_histogram(
    y_test_scalar,
    y_test_pred_scalar,
    scalar_target_name,
)

# %% [markdown]
# **Interpretation.** Most residuals are close to zero, so many samples are
# predicted reasonably near their true values. The histogram is not symmetric,
# though: it has a long positive tail. Since residual is defined as
# `prediction - truth`, positive residuals mean the model predicts insertion loss
# values that are less negative than the true values. This confirms that deep-loss
# cases are often under-estimated by the scalar baseline.

# %% [markdown]
# ### 1.8 Plotting: Residual Vs Frequency
#
# The x-axis is frequency in GHz and the y-axis is residual. This shows whether
# the linear model fails more badly at high frequency or resonance-like regions.

# %%
fig_scalar_residual_frequency = plot_scalar_residual_vs_frequency(
    test_set.dataframe,
    y_test_scalar,
    y_test_pred_scalar,
    scalar_target_name,
)

# %% [markdown]
# **Interpretation.** The error spread increases with frequency. At low
# frequencies, residuals are tightly concentrated around zero, but at higher
# frequencies the residual band becomes wider and large positive outliers appear.
# This suggests that the linear scalar model is not failing uniformly; it becomes
# less reliable in high-frequency regions where the S-parameter response has more
# complex structure.

# %% [markdown]
# ### 1.9 Plotting: MAE By Frequency
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
    test_set.dataframe,
    y_test_scalar,
    y_test_pred_scalar,
    scalar_target_name,
)

# %% [markdown]
# **Interpretation.** The MAE rises steadily with frequency, from low error near
# the start of the sweep to much larger error near 100 GHz. This makes the
# frequency dependence of the baseline error clear: the scalar Ridge model is a
# much stronger approximation at low frequency than at high frequency.

# %% [markdown]
# ## 2. Vector Insertion Loss Baseline
#
# This baseline trains one multi-output non-neural regressor to predict a vector
# of six insertion-loss values from the same design-frequency feature vector.
# Full IL curves are reconstructed by evaluating the trained vector model across
# all frequency points for the same design.

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
print(f"Vector target names: {target_names}")
print(f"Y_train shape: {Y_train.shape}")
print(f"Y_val shape: {Y_val.shape}")
print(f"Y_test shape: {Y_test.shape}")

# %% [markdown]
# ### 2.2 Target Loading
#
# The six dB IL targets were loaded once with:
#
# ```python
# TouchstoneLoader(mode="scalar", representation="db", config=cfg)
# ```
#
# The loader accesses Touchstone data on demand while filling `Y_train`, `Y_val`,
# and `Y_test`, but those arrays remain in notebook memory after this step. This
# is a deliberate trade-off for the Ridge baseline: it keeps the fitting,
# validation, metric, and plotting code simple while leaving the cleaned CSV
# unchanged.

# %% [markdown]
# ### 2.3 Model Training
#
# The vector baseline uses one multi-output Ridge regression model. Validation MAE
# averaged across all six output columns selects the regularisation strength. It
# uses the same `StandardScaler()` plus `Ridge(alpha=alpha)` pipeline reported in
# the scalar modelling section.

# %%
vector_model, vector_alpha_results = fit_ridge_with_validation(
    X_train,
    Y_train,
    X_val,
    Y_val,
    RIDGE_ALPHA_GRID,
    build_ridge_pipeline,
)

best_vector_alpha = vector_alpha_results.loc[
    vector_alpha_results["MAE"].idxmin(),
    "alpha",
]

print("Vector Ridge validation sweep:")
print(vector_alpha_results)
print(f"Best vector alpha: {best_vector_alpha:g}")
print(f"Selected vector preprocessing: {vector_model.named_steps['scaler']}")
print(f"Selected vector regressor: {vector_model.named_steps['model']}")

# %% [markdown]
# ### 2.4 Vector Evaluation

# %%
Y_val_pred = cast(np.ndarray, vector_model.predict(X_val))  # pylint: disable=invalid-name
Y_test_pred = cast(np.ndarray, vector_model.predict(X_test))  # pylint: disable=invalid-name

vector_metrics = pd.DataFrame(
    [
        {"split": "validation", **regression_metrics(Y_val, Y_val_pred)},
        {"split": "test", **regression_metrics(Y_test, Y_test_pred)},
    ]
)
per_target_test_metrics = per_target_metrics(Y_test, Y_test_pred, target_names)

print("Overall vector metrics:")
print(vector_metrics)
print("\nPer-port-pair test metrics:")
print(per_target_test_metrics)

# %% [markdown]
# ### 2.5 Plot Vector IL Distributions Across Test Designs

# %%
fig_vector_distributions = plot_vector_prediction_bands_by_frequency(
    test_set.dataframe,
    Y_test,
    Y_test_pred,
    target_names,
)

# %% [markdown]
# **Interpretation.** The vector Ridge model captures the central frequency-loss
# trend for each through path, but the predicted percentile bands are much
# narrower than the true bands. This repeats the scalar-model conclusion across
# all six targets: the linear baseline learns the average response but underfits
# design-specific variation, especially where the high-frequency response
# envelope widens.

# %% [markdown]
# ### 2.6 Plot Vector Predicted Vs True Scatter

# %%
fig_vector_scatter = plot_vector_true_vs_predicted(
    Y_test,
    Y_test_pred,
    target_names,
)

# %% [markdown]
# **Interpretation.** All six scatter plots show predictions compressed into a
# narrower range than the true IL values. Points near the diagonal correspond to
# good predictions, but the deep-loss samples sit far away from the diagonal. This
# indicates that the vector Ridge baseline has the same main limitation for every
# through path: it smooths the response and does not reproduce the full dynamic
# range of the true S-parameter targets.

# %% [markdown]
# ### 2.7 Plot Vector Residual Histograms

# %%
fig_vector_residual_hists = plot_vector_residual_histograms(
    Y_test,
    Y_test_pred,
    target_names,
)

# %% [markdown]
# **Interpretation.** The residual distributions are broadly similar across the
# six targets. They peak close to zero, which means the model often gives a
# reasonable average prediction, but the asymmetric tails show that some samples
# have much larger errors. The repeated shape across targets suggests this is a
# model-capacity limitation rather than a problem with only one port pair.

# %% [markdown]
# ### 2.8 Plot Vector Residual Vs Frequency

# %%
fig_vector_residual_frequency = plot_vector_residual_vs_frequency(
    test_set.dataframe,
    Y_test,
    Y_test_pred,
    target_names,
)

# %% [markdown]
# **Interpretation.** For all six port pairs, the residual spread grows as
# frequency increases. The high-frequency region contains both a wider error band
# and more large positive outliers. This means the vector Ridge model's errors are
# frequency-dependent, with the weakest performance occurring where the true
# responses are most nonlinear and resonance-like.

# %% [markdown]
# ### 2.9 Plot Vector MAE By Frequency
#
# For each target column:
#
# $$
#     MAE_j(f_k) = mean_i |IL_j(i,k) - \widehat{IL}_j(i,k)|
# $$

# %%
fig_vector_mae_frequency = plot_vector_mae_by_frequency(
    test_set.dataframe,
    Y_test,
    Y_test_pred,
    target_names,
)

# %% [markdown]
# **Interpretation.** The six MAE curves are very close to each other and all
# increase with frequency. This shows that the six through-path targets have
# similar difficulty for the Ridge baseline. The steadily rising curves also
# confirm the main conclusion from the residual plots: high-frequency prediction
# is the dominant weakness of this non-neural linear baseline.

# %% [markdown]
# ## Validation Checks

# %%
assert y_train_scalar.shape == (len(train_set),)
assert Y_train.shape == (len(train_set), 6)
assert Y_val.shape == (len(val_set), 6)
assert Y_test.shape == (len(test_set), 6)
assert np.isfinite(y_test_pred_scalar).all()
assert np.isfinite(Y_test_pred).all()

print("Validation checks passed:")
print("- split integrity by SIMU_INDEX")
print("- scalar target shape")
print("- vector target shapes")
print("- finite features, targets, and predictions")
print("- target names match configured port pairs")
