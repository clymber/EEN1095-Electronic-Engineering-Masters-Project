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

from sparam_surrogate.config import SurrogateConfig
from sparam_surrogate.data import DLDataset, TouchstoneLoader, random_simu_indices
from sparam_surrogate.models import (
    PolynomialModel,
    RandomForestModel,
    ScalarRidgeModel,
    VectorRidgeModel,
)
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

# %% [markdown]
# # Non-Neural Network Modelling
#
# In this notebook, I train non-neural baseline models to predict insertion-loss
# targets derived from S-parameters. The first baseline predicts one scalar IL
# value for one port pair. The remaining baselines predict one vector containing
# the six configured through-path IL values.
#
# The cleaned CSV stores only design features, frequency, split labels, simulation
# indices, and Touchstone paths. The `TouchstoneLoader` reads Touchstone files on
# demand, then this notebook materializes the target arrays before fitting because
# the `scikit-learn` Ridge baselines expect in-memory `NumPy` arrays.

# %%
cfg = SurrogateConfig.from_csv()
random_seed = cfg.project.seed
scalar_db_loader = TouchstoneLoader("scalar", cfg, "db", 8)
vector_db_loader = TouchstoneLoader("vector", cfg, "db", 8)
scalar_target_index = 0  # pylint: disable=invalid-name
scalar_target_name = scalar_db_loader.target_names[scalar_target_index]
vector_target_names = tuple(vector_db_loader.target_names)

print(f"Name of raw dataset: {cfg.dataset.name}")
print(f"Raw data directory: {cfg.dataset.path}")
print(f"Processed directory: {cfg.paths.processed_data}")
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
# with the scalar dB loader. On the first run, each view materializes its target
# array from Touchstone files and writes a split cache. Later runs load the cache
# whenever it is newer than the cleaned CSV.

# %%
scalar_train_set, scalar_val_set, scalar_test_set = DLDataset.from_cleaned_csv(
    cfg.preprocessing.processed_csv,
    target_loader=scalar_db_loader,
    cache=True,
)

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
# pylint: disable=invalid-name

# %% [markdown]
# The Ridge baseline needs full `NumPy` arrays. `DLDataset` persists targets on
# disk, while `TouchstoneLoader` temporarily caches parsed Touchstone networks
# only during a cold load. Its small in-memory network cache can be cleared once
# the scalar arrays are ready.

# %%
print(f"Scalar Touchstone cache info: {scalar_db_loader.cache_info()}")
scalar_db_loader.clear_cache()
print(f"Scalar Touchstone cache after clearing: {scalar_db_loader.cache_info()}")

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
scalar_model = ScalarRidgeModel.from_config(scalar_ridge_config)
scalar_model.fit(X_train_scalar, y_train_scalar, X_val_scalar, y_val_scalar)

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
y_val_pred_scalar = scalar_model.predict(X_val_scalar)
y_test_pred_scalar = scalar_model.predict(X_test_scalar)

scalar_metrics = pd.DataFrame(
    [
        {"split": "validation", **regression_metrics(y_val_scalar, y_val_pred_scalar)},
        {"split": "test", **regression_metrics(y_test_scalar, y_test_pred_scalar)},
    ]
)
print(scalar_metrics)

# %% [markdown]
# For the single `S7_1_DB` target, the scalar model achieved MAE/RMSE values of
# 7.30/10.77 on the validation set and 7.35/10.90 on the held-out test set. The test
# performance is slightly worse, with an error increase of approximately 0.8% in MAE
# and 1.3% in RMSE. This indicates a small generalisation gap rather than severe
# overfitting.
#
# The RMSE remains higher than the MAE, which suggests that some samples still contain
# larger prediction errors. Therefore, the baseline model is useful as a pipeline
# validation step, but further analysis is needed to locate the high-error regions,
# especially across frequency and design-parameter space.

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
    scalar_db_loader,
    selected_simu_indices,
)

# %% [markdown]
# 1. __Overall trend captured__
#
#    The Ridge model follows the main downward trend of `S7_1_DB` as frequency
#    increases. This shows that the model has learned the broad frequency-dependent
#    behaviour.
#
# 2. __Spread is underestimated__
#
#    The true 10th–90th percentile band becomes much wider at high frequency, but the
#    predicted band remains very narrow. This means the model mostly predicts an
#    average-like response and does not capture design-to-design variation well.
#
# 3. __High-frequency region is harder__
#
#    The mismatch becomes more obvious above the mid/high-frequency range. This suggests
#    that the relationship between PCB parameters and response becomes more nonlinear at
#    higher frequencies.

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
# 1. **Perfect prediction reference**
#
#    The black diagonal line represents perfect prediction. Points close to this line
#    would mean the model predicts the true value accurately.
#
# 2. **Predictions are compressed**
#
#    The true values cover a very wide range, but the predicted values stay in a much
#    narrower range. This means the Ridge model cannot reproduce the full dynamic range
#    of `S7_1_DB`.
#
# 3. **Deep-loss samples are missed.**
#
#    For very negative true values, the model predicts values that are not negative
#    enough. In other words, strong attenuation cases are underestimated.
#
# 4. **Same issue as the percentile plot**
#
#    This confirms the result from Section 1.5: the model captures the broad
#    trend, but it smooths the response too much and misses extreme cases.
#
# The scalar Ridge model is useful as a baseline, but it is too limited for accurate
# sample-level prediction across the full response range.
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
# 1. **Error increases with frequency**
#
#    The MAE rises from about 1 dB at low frequency to about 13–14 dB near 100 GHz. This
#    shows that the model performs much better at low frequency than at high frequency.
#
# 2. **High-frequency prediction is the main weakness**
#
#    The error does not stay constant across the frequency range. Most of the prediction
#    difficulty comes from the mid-to-high-frequency region, especially above roughly 40
#    GHz.
#
# 3. **The increase is smooth, not random**
#
#    The MAE grows gradually rather than showing only a few isolated spikes.
#    This suggests a systematic modelling limitation, not just a small number of bad
#    samples.
#
# 4. **Likely cause: underfitting**
#
#    Ridge regression is a linear model, so it produces smooth predictions. At higher
#    frequencies, the S-parameter response becomes more nonlinear and more sensitive to
#    PCB design parameters. The model therefore cannot capture the full behaviour.
#
# 5. **Connection to previous plots**
#
#    This supports the observations from Sections 1.5 and 1.6. The Ridge model captures
#    the broad trend, but it underestimates the response spread and misses strong
#    attenuation cases, especially at high frequency.
#
# The scalar Ridge model is acceptable as an early baseline, but its accuracy degrades
# clearly with frequency. A nonlinear model is needed to improve high-frequency
# prediction.

# %% [markdown]
# The scalar experiment is now complete. Its figures and predictions are retained,
# but the large scalar dataset and feature arrays are released before loading the
# vector experiment. This keeps the two experiments independent without holding two
# complete sets of dataframe views in memory at the same time.

# %%
del scalar_train_set, scalar_val_set, scalar_test_set
del X_train_scalar, X_val_scalar, X_test_scalar
del y_train_scalar, y_val_scalar, y_test_scalar, y_val_pred_scalar

# %% [markdown]
# ## 2. Vector Insertion Loss Baseline
#
# This baseline trains one multi-output non-neural regressor to predict a vector
# of six insertion-loss values from the same design-frequency feature vector.
# Full IL curves are reconstructed by evaluating the trained vector model across
# all frequency points for the same design.

# %% [markdown]
# The vector and polynomial experiments use a separate set of train, validation,
# and test views configured with the six-target vector dB loader. Their cache files
# are independent from the scalar experiment: a scalar cache can be rebuilt or
# removed without affecting vector model development.

# %%
vector_train_set, vector_val_set, vector_test_set = DLDataset.from_cleaned_csv(
    cfg.preprocessing.processed_csv,
    target_loader=vector_db_loader,
    cache=True,
)

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

# %%
print(f"Vector Touchstone cache: {vector_db_loader.cache_info()}")
vector_db_loader.clear_cache()
print(f"Vector Touchstone cache after clearing: {vector_db_loader.cache_info()}")

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
# TouchstoneLoader(mode="vector", representation="db", config=cfg)
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
vector_model = VectorRidgeModel.from_config(vector_ridge_config)
vector_model.fit(X_train, Y_train, X_val, Y_val)

vector_alpha_results = vector_model.validation_results
best_vector_alpha = vector_model.best_alpha
if vector_alpha_results is None or best_vector_alpha is None:
    raise RuntimeError("Vector Ridge model did not record validation results.")

print("Vector Ridge validation sweep:", vector_alpha_results, sep="\n")
print(f"Best vector alpha: {best_vector_alpha:g}")
print(f"Selected vector preprocessing: {vector_model.pipeline.named_steps['scaler']}")
print(f"Selected vector regressor: {vector_model.pipeline.named_steps['model']}")

# %% [markdown]
# 1. **Validation performance is almost unchanged**
#
#    Across all tested `alpha` values, the MAE stays around `7.414 dB` and the RMSE
#    stays around `10.94 dB`, with only negligible differences between settings. The
#    selected value is `alpha = 10`, the strongest regularisation tested, but the
#    improvement over weaker settings is extremely small.
#
# 2. **Regularisation is not the main issue**
#
#    Changing `alpha` does not materially improve validation performance. This suggests
#    that the main limitation is not ordinary overfitting controlled by L2 shrinkage,
#    but underfitting: the linear model is too simple to capture the full behaviour.
#
# 3. **Vector output works, but remains limited**
#
#    The model predicts six output columns together, but Ridge still uses a linear
#    relationship between input features and outputs. It does not fully capture
#    nonlinear frequency-dependent effects or complex design-to-design variation. The
#    flat validation sweep further indicates that tuning `alpha` cannot significantly
#    improve performance, so meaningful gains will likely require a more expressive
#    model such as polynomial features, tree-based regression, or a neural network.
#
# 4. **StandardScaler is appropriate**
#
#    Using `StandardScaler()` is sensible because Ridge regression is sensitive to
#    feature scale. This keeps parameters such as geometry values and frequency on
#    comparable numerical scales.
#

# %% [markdown]
# ### 2.4 Vector Ridge Model Evaluation

# %%
Y_val_pred = vector_model.predict(X_val)  # pylint: disable=invalid-name
Y_test_pred = vector_model.predict(X_test)  # pylint: disable=invalid-name

vector_metrics = pd.DataFrame(
    [
        {"split": "validation", **regression_metrics(Y_val, Y_val_pred)},
        {"split": "test", **regression_metrics(Y_test, Y_test_pred)},
    ]
)
per_target_test_metrics = per_target_metrics(Y_test, Y_test_pred, vector_target_names)

print("Overall vector metrics:", vector_metrics, sep="\n")
print("\nPer-port-pair test metrics:", per_target_test_metrics, sep="\n")

# %% [markdown]
# 1. **Test performance is slightly worse than validation**
#
#    The validation MAE/RMSE are `7.41 dB` and `10.94 dB`, while the test MAE/RMSE
#    increase to `7.47 dB` and `11.08 dB`. This shows a small generalisation gap, but
#    not severe overfitting.
#
# 2. **RMSE increases more than MAE**
#
#    The test RMSE is noticeably higher than the test MAE. This suggests that some
#    held-out samples have relatively large prediction errors. In other words, the model
#    is not just making small uniform errors; it misses some difficult cases more
#    strongly.
#
# 3. **All six port pairs have similar difficulty**
#
#    The per-port-pair MAE values are all close, roughly between `7.26 dB` and
#    `7.64 dB`. This means no single output dominates the overall error. The Ridge model
#    has similar predictive difficulty across the six through-link responses.
#
# 4. **Vector Ridge is still a linear baseline**
#
#    Although the model predicts six outputs together, Ridge regression still learns a
#    linear mapping from the input features to each output. Therefore, it cannot fully
#    capture nonlinear frequency behaviour, resonance-like effects, or wide
#    design-to-design variation.
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
# 1. **All six links show a similar pattern**
#
#    The six predicted distributions have almost the same behaviour. The predicted
#    median follows the true median reasonably well for every port pair, so the vector
#    Ridge model has learned the broad frequency trend across all six outputs.
#
# 2. **Predicted spread is too narrow**
#
#    The true 10th–90th percentile bands become much wider as frequency increases,
#    especially above the mid-frequency range. However, the predicted percentile bands
#    remain very narrow. This means the model does not capture the full design-to-design
#    variation.
#
# 3. **High-frequency variation is missed**
#
#    At high frequencies, the true responses vary strongly between different test
#    designs, but the Ridge model mostly predicts a smooth average-like response. This
#    confirms that the model underfits the more complex high-frequency behaviour.
#
# 4. **Vector Ridge is stable but limited**
#
#    The model is stable across multiple outputs, which makes it a useful baseline.
#    However, the narrow predicted bands show that linear Ridge regression cannot model
#    the full response distribution.

# %%
fig_random_vector_design_curves = plot_design_prediction_curves(
    vector_model,
    vector_test_set,
    vector_db_loader,
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
# 1. **Predictions are compressed for all six outputs**
#
#    In every subplot, the true values cover a very wide range, but the predicted values
#    stay in a much narrower band. This means the vector Ridge model cannot reproduce
#    the full dynamic range of the six through-link responses.
#
# 2. **Deep-loss cases are not predicted correctly**
#
#    For very negative true values, the model predicts values that are not negative
#    enough. Strong attenuation cases are therefore underestimated.
#
# 3. **All port pairs show the same failure pattern**
#
#    The six scatter plots look very similar. This agrees with the per-port-pair
#    metrics: no single port pair is uniquely problematic; the limitation comes from the
#    model type.
#
# 4. **Same conclusion as the distribution plots**
#
#    This confirms Section 2.5 at the sample level. The model captures the broad trend,
#    but it predicts an average-like response and misses extreme design-frequency cases.
#
# 5. **Conclusion**
#
#    Vector Ridge is a useful multi-output baseline, but it is too simple for accurate
#    prediction across the full response range. A nonlinear model is needed to capture
#    stronger attenuation and wider design-to-design variation.

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
# 1. **Error increases with frequency**
#
#    The MAE is low at the beginning of the frequency range, around `1 dB`, but rises
#    steadily to about `13–14 dB` near `100 GHz`. This shows that the Vector Ridge model
#    becomes less accurate as frequency increases.
#
# 2. **All six links follow almost the same error trend**
#
#    The six MAE curves are very close to each other across the whole frequency range.
#    This means the model has similar difficulty across all six through-link targets,
#    rather than failing on only one specific port pair.
#
# 3. **High-frequency modelling is the main weakness**
#
#    The error growth is smooth and systematic, not caused by isolated spikes. This
#    suggests underfitting: the linear Ridge model cannot capture the more complex
#    high-frequency behaviour of the S-parameter responses.
#
# 4. **Conclusion**
#
#    The Vector Ridge model is a stable multi-output baseline, but its error increases
#    strongly with frequency. A more expressive nonlinear model is needed to improve
#    prediction accuracy, especially in the high-frequency region.
#

# %% [markdown]
# ## 3. Polynomial Vector Baseline
#
# This baseline keeps the same vector target definition as the vector Ridge model,
# but expands each input feature with powers before fitting a regularised linear
# model. The default follow-up experiment now sweeps degrees 3, 4, and 5 with a
# stronger regularisation search, then selects the degree and Ridge
# regularisation strength with the lowest validation MAE. For one feature, the
# powers-only expansion is:
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
polynomial_model = PolynomialModel.from_config(polynomial_ridge_config)
polynomial_model.fit(X_train, Y_train, X_val, Y_val)

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
# 1. **Degree 5 is selected in the current run**
#
#    The best validation result comes from `degree = 5` with `alpha = 1000`.
#    The selected powers-only expansion increases the feature count from 11 to
#    55. Higher powers help slightly, but the improvement remains modest.
#
# 2. **Regularisation helps, but only slightly**
#
#    The best model uses the largest tested `alpha` (`1000`), suggesting
#    stronger regularisation is beneficial. However, validation MAE changes
#    little across `alpha` values, so most of the gain comes from the polynomial
#    features.
#
# 3. **The gain is modest**
#
#    The best polynomial validation MAE is `7.3723 dB`, compared with `7.4142
#    dB` for Vector Ridge. The polynomial expansion improves the baseline, but
#    only slightly.
#
# 4. **Conclusion**
#
#    Polynomial features provide a small but consistent improvement while keeping the
#    model simple and interpretable. However, the gain is limited, suggesting that a
#    more expressive nonlinear model will be needed for larger improvements.

# %% [markdown]
# ### 3.2 Polynomial Evaluation

# %%
Y_val_pred_poly = polynomial_model.predict(X_val)  # pylint: disable=invalid-name
Y_test_pred_poly = polynomial_model.predict(X_test)  # pylint: disable=invalid-name

polynomial_metrics = pd.DataFrame(
    [
        {"split": "validation", **regression_metrics(Y_val, Y_val_pred_poly)},
        {"split": "test", **regression_metrics(Y_test, Y_test_pred_poly)},
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
# Test MAE drops from `7.4740 dB` to `7.4269 dB`, while RMSE decreases from `11.0796 dB`
# to `11.0532 dB`. The gain is small, but consistent.

# %%
per_target_vector_comparison = per_target_test_metrics.copy()
per_target_vector_comparison.insert(0, "model", "Vector Ridge")

per_target_polynomial_metrics = per_target_metrics(
    Y_test,
    Y_test_pred_poly,
    vector_target_names,
)
per_target_polynomial_comparison = per_target_polynomial_metrics.copy()
per_target_polynomial_comparison.insert(0, "model", "Polynomial")
per_target_model_comparison = pd.concat(
    [per_target_vector_comparison, per_target_polynomial_comparison],
    ignore_index=True,
)

print(f"Per-target model comparison:\n{per_target_model_comparison}")

# %% [markdown]
# 1. The improvement appears across all six targets.
#
#    Each port-pair prediction improves slightly, suggesting the polynomial features
#    provide a small overall benefit rather than helping only one specific link.
#
# 2. The main limitation remains.
#
#    The improvement is only a few hundredths of a dB, so the model is mainly refining
#    the Ridge baseline rather than addressing its underlying weaknesses.
#
# Overall, Polynomial Ridge is a slightly stronger non-neural baseline, but
#    the modest gain suggests that a more expressive nonlinear model is likely
#    needed for further improvement.
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
    vector_db_loader,
    selected_simu_indices,
)

# %% [markdown]
# 1. The Polynomial model follows the median trend well.
#
#    For all six port pairs, the predicted median follows the true median across
#    frequency. This shows that the polynomial features help the model represent
#    the main frequency-dependent trend.
#
# 2. The predicted spread is still too narrow.
#
#    The true 10th–90th percentile band becomes much wider at high frequency,
#    but the predicted band remains narrow. This means the model still
#    underestimates design-to-design variation.
#
# 3. The qualitative behaviour is similar to Vector Ridge.
#
#    Compared with Vector Ridge, the Polynomial model gives a small improvement.
#    The predicted median curves are less constrained to a straight-line trend
#    and show a more realistic curved frequency response, indicating that the
#    polynomial features capture some nonlinear frequency-dependent behaviour.
#    However, the improvement is modest, as the predicted distribution remains
#    much narrower than the true distribution and still fails to capture the full
#    response range.
#
# Overall, Polynomial Ridge is a useful refinement of the linear baseline, but
# it still underfits the wider high-frequency distribution. The next improvement
# likely requires a more flexible nonlinear model.
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
# 1. The Polynomial model is slightly better overall.
#
#    The Polynomial MAE curve is very close to the Vector Ridge curve, but it is
#    slightly lower in some frequency regions, especially at lower frequencies.
#    This agrees with the numerical results: the Polynomial model improves the
#    overall test MAE slightly, from `7.4740 dB` to `7.4269 dB`.
#
# 2. Both models show the same frequency-dependent error pattern.
#
#    The two curves have almost the same shape. MAE increases steadily from
#    around `1 dB` at low frequency to about `13–14 dB` near `100 GHz`. This
#    means the Polynomial model has not changed the main difficulty of the
#    problem: prediction becomes much harder at higher frequencies.
#
# 3. The improvement is useful but limited.
#
#    The Polynomial model adds some nonlinear flexibility, so it is a stronger
#    baseline than plain Vector Ridge. However, the error curve is only shifted
#    slightly and still rises strongly with frequency. This suggests that
#    polynomial features refine the model but do not fully solve the
#    high-frequency underfitting problem.
#
# Overall, Polynomial Ridge is a modest improvement over Vector Ridge. It should
# be kept as the stronger non-neural baseline, but the similar MAE-by-frequency
# shape shows that a more expressive nonlinear model is still needed for
# substantial improvement.
#

# %% [markdown]
# ### 3.5 Why Polynomial Ridge Only Gives A Limited Improvement
#
# The Polynomial Ridge model improves the curve shape slightly compared with
# Vector Ridge, but it still does not fit the full response distribution well.
# This is because the model is not simply fitting one curve; it is learning a
# global relationship across many designs, frequencies, and output samples.
#
# %% [markdown]
# #### 1. The model is fitting many designs at once, not one curve
#
# For one fixed PCB design, a polynomial curve may fit the frequency response
# reasonably well. However, in this experiment, the model is learning the
# mapping:
#
# $$
# (\mathbf{u}, f) \rightarrow S_{7,1,\mathrm{dB}}(f)
# $$
#
# Here, $\mathbf{u}$ represents the PCB design-parameter vector, and $(f)$
# represents frequency. Therefore, the model must learn not only how `S7_1_DB`
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
# * $\hat{y}$ is the predicted target value, such as predicted `S7_1_DB`.
# * $\beta_0$ is the intercept term.
# * $R$ is the total number of expanded polynomial features.
# * $\beta_r$ is the learned coefficient for the (r)-th polynomial feature.
# * $\phi_r(\mathbf{u},f)$ is the (r)-th transformed feature generated
#   from design parameters and frequency.
#
# In the current experiment, the original input feature count is 11, but the
# polynomial expansion produces 55 features:
#
# $$
# 11\ \text{original features} \rightarrow 55\ \text{polynomial features}
# $$
#
# Therefore, for the Polynomial Ridge model in this experiment:
#
# $$
# R=55
# $$
#
# This gives the model more flexibility than plain Ridge, but it is still a
# compact model rather than a highly expressive nonlinear model.
#
# %% [markdown]
# #### 3. Strong regularisation limits the curvature
#
# The best Polynomial Ridge model uses `alpha = 1000`, which means the model is
# strongly regularised. Ridge regression penalises large coefficients:
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
# Here, $\alpha$ controls the strength of regularisation. A larger $\alpha$
# shrinks the coefficients more strongly. This improves stability, but it also
# prevents the polynomial curve from bending too aggressively. As a result, the
# model still produces relatively smooth and average-like predictions.
#
# %% [markdown]
# #### 4. High-frequency behaviour is more complex than polynomial curvature
#
# At higher frequencies, the S-parameter response may be affected by stronger
# nonlinear effects, coupling, resonances, and sensitivity to small geometry
# changes. A low-capacity polynomial model may not represent these behaviours
# well.
#
# This explains why the Polynomial Ridge median curve is slightly less straight
# than Vector Ridge, but the predicted 10th–90th percentile band is still too
# narrow.
#
# %% [markdown]
# #### 5. The loss function encourages average predictions
#
# The model is trained to minimise the overall error across many designs and
# frequency points. When some extreme deep-loss samples are difficult to
# predict, the model can reduce total error by staying close to the central
# trend rather than fitting those extreme cases.
#
# Therefore, Polynomial Ridge improves the median curve shape slightly, but it
# still misses the full design-to-design variation.
#
# %% [markdown]
# #### Conclusion
#
# Polynomial Ridge is a useful improvement over Vector Ridge because it adds
# nonlinear feature terms and produces a less straight, more realistic median
# curve. However, it is still a compact global model, so it remains too smooth
# and average-like for the full design-to-design response spread. This motivates
# the next non-neural check: a more flexible tree-based model that can learn
# local design-frequency partitions.
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
random_forest_model = RandomForestModel.from_config(random_forest_config)
random_forest_model.fit(X_train, Y_train, X_val, Y_val)

random_forest_validation_results = random_forest_model.validation_results
if random_forest_validation_results is None:
    raise RuntimeError("Random Forest model did not record validation results.")

print(f"Random Forest validation sweep:\n{random_forest_validation_results}\n")
print(f"Selected Random Forest model:\n{random_forest_model.regressor}")

# %% [markdown]
# ### 4.2 Random Forest Evaluation

# %%
Y_train_pred_rf = random_forest_model.predict(X_train)  # pylint: disable=invalid-name
Y_val_pred_rf = random_forest_model.predict(X_val)  # pylint: disable=invalid-name
Y_test_pred_rf = random_forest_model.predict(X_test)  # pylint: disable=invalid-name

random_forest_metrics = pd.DataFrame(
    [
        {"split": "train", **regression_metrics(Y_train, Y_train_pred_rf)},
        {"split": "validation", **regression_metrics(Y_val, Y_val_pred_rf)},
        {"split": "test", **regression_metrics(Y_test, Y_test_pred_rf)},
    ]
)

print(f"Random Forest vector metrics:\n{random_forest_metrics}")

# %% [markdown]
# **Random Forest overfits the training data**
#
#    Training MAE/RMSE are only `0.4438/0.8707 dB`, but validation MAE/RMSE
#    rise to `7.7313/11.5134 dB` and test MAE/RMSE are `7.7570/11.5838 dB`.
#    This large train-validation gap shows that the forest fits the training
#    samples closely but does not generalise well to held-out PCB designs.

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
# 1. **Random Forest is worse by pointwise error metrics**
#
#    Test MAE is `7.7570 dB`, compared with `7.4740 dB` for Vector Ridge and
#    `7.4269 dB` for Polynomial Ridge. Test RMSE is also worse: `11.5838 dB`
#    versus `11.0796 dB` for Vector Ridge and `11.0532 dB` for Polynomial
#    Ridge. By aggregate pointwise error, Random Forest is not the strongest
#    model in this run.
#
# 2. **The metric weakness is consistent across all six targets**
#
#    Each per-target Random Forest MAE is higher than the corresponding Vector
#    Ridge and Polynomial Ridge value. This means the model is not only failing
#    on one difficult link; its pointwise error is higher across the configured
#    through paths.
#
# **Conclusion**
#
#    Random Forest provides a useful nonlinear tabular stress test. It is worse
#    by MAE/RMSE, but the distribution plots below show that it can represent
#    response spread and nonlinear curvature better than the Ridge-style
#    baselines. The remaining challenge is to keep this richer distribution and
#    curve-shape behaviour while improving held-out pointwise accuracy.

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
# 1. **The predicted distribution is a qualitative improvement**
#
#    Compared with the Ridge-style baselines, the Random Forest predicted
#    10th-90th percentile band is wider and follows the true response spread
#    more closely. The predicted median also tracks the true median well across
#    the six configured through paths.
#
# 2. **The spread is still not perfect**
#
#    The true high-frequency band remains wider than the predicted band in some
#    regions, so Random Forest still underestimates the full design-to-design
#    variation. However, the distribution shape is much more realistic than the
#    very narrow Ridge and Polynomial Ridge bands.
#
# 3. **Metric and distribution conclusions differ**
#
#    Random Forest is worse by aggregate MAE/RMSE, but better at representing
#    the held-out response distribution. This makes it useful evidence that
#    nonlinear model capacity helps the shape of the prediction, even if this
#    specific forest does not minimise pointwise error.
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
        vector_db_loader,
        selected_simu_indices,
    )
)

# %% [markdown]
# 1. **Random Forest greatly improves curve curvature**
#
#    Polynomial Ridge mostly produces smooth, nearly straight responses for each
#    held-out design. Random Forest follows local bends, dips, and recovery
#    regions much more closely, especially for designs with stronger
#    high-frequency curvature.
#
# 2. **The improvement is qualitative rather than metric-led**
#
#    The metric results above show that Random Forest does not improve aggregate
#    pointwise accuracy, but this comparison plot shows a clear qualitative
#    improvement in response curvature compared with Polynomial Ridge.

# %% [markdown]
# ## 5. Four-Model Comparison On S7_1_DB
#
# The scalar Ridge model only predicts `S7_1_DB`, so the cleanest comparison is
# to evaluate all four fitted models on that shared target only. The scalar
# model contributes its direct prediction. The Vector Ridge, Polynomial Ridge,
# and Random Forest models contribute only their `S7_1_DB` output column, even
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
# First, compare the four `S7_1_DB` MAE curves by frequency.

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
# `S7_1_DB` curve shape compared with the Ridge-style baselines.

# %%
fig_s7_four_model_distributions = plot_shared_target_prediction_bands(
    vector_test_set.dataframe,
    shared_target_true,
    shared_target_predictions,
    shared_target_name,
)

# %% [markdown]
# 1. **Random Forest does not improve pointwise error on the shared target**
#
#    On `S7_1_DB`, Random Forest test MAE/RMSE are `7.6914/11.4722 dB`. This
#    is worse than Scalar Ridge and Vector Ridge (`7.3544/10.9045 dB`) and
#    worse than Polynomial Ridge (`7.3154/10.8848 dB`). Therefore, the stronger
#    nonlinear tabular model does not improve pointwise MAE/RMSE for the shared
#    scalar target in this run.
#
# 2. **Vector Ridge does not clearly improve over Scalar Ridge on `S7_1_DB`**
#
#    The Scalar Ridge and Vector Ridge curves are almost overlapping. This
#    suggests that predicting all six outputs together does not significantly
#    improve the individual `S7_1_DB` prediction. For Ridge regression, the
#    multi-output model is therefore useful for convenience and consistency,
#    but it does not provide strong shared-output learning.
#
# 3. **Polynomial Ridge gives only a small improvement**
#
#    Polynomial Ridge is slightly better in some frequency regions, especially
#    near the low-frequency range, and its predicted median curve is slightly
#    less straight than the Ridge curves. This shows that polynomial features
#    add some nonlinear flexibility. However, the difference is small, so the
#    improvement is modest.
#
# 4. **Random Forest improves local curvature**
#
#    The Random Forest result suggests that local tree partitions help capture
#    nonlinear curvature, including bends and dips that Polynomial Ridge smooths
#    away. However, this curvature improvement does not yet generalise well
#    enough to improve pointwise MAE/RMSE on this held-out design split. The
#    very low train error and much higher validation/test error still point to
#    overfitting.
#
# 5. **Random Forest improves the predicted distribution shape**
#
#    The distribution comparison shows that Random Forest gives a wider and more
#    realistic predicted band than the Ridge-style models. It still may not
#    capture the full high-frequency design-to-design variation, but it is a
#    clear qualitative improvement in distribution shape even though its
#    pointwise error metrics are worse.
#
# The four-model comparison shows a clear progression: Scalar Ridge
# establishes the single-target baseline, Vector Ridge extends the same idea to
# multiple outputs, Polynomial Ridge adds limited nonlinear flexibility, and
# Random Forest tests a stronger non-neural tabular model. In this run, Random
# Forest improves both the qualitative distribution and the local curve
# curvature, but it does not improve aggregate held-out MAE/RMSE. This
# strengthens the case for a model class that can preserve the richer response
# shape while improving pointwise generalisation.
#
