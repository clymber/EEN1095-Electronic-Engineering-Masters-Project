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

import pandas as pd

from sparam_surrogate.config import load_config, relative_to_project_root
from sparam_surrogate.data import DLDataset, TouchstoneLoader
from sparam_surrogate.models import (
    RIDGE_ALPHA_GRID,
    PolynomialModel,
    ScalarRidgeModel,
    VectorRidgeModel,
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

DS_NAME = "linkOn8CavityStackBetween10x10Array_19_08_2021"
CLEANED_CSV = "sipi_dataset_cleaned.csv"

# %% [markdown]
# # Non-Neural Network Modelling
#
# In this notebook, I train non-neural baseline models to predict insertion-loss
# targets derived from S-parameters. The first baseline predicts one scalar IL
# value for one port pair. The second and third baselines predict one vector
# containing the six configured through-path IL values.
#
# The cleaned CSV stores only design features, frequency, split labels, simulation
# indices, and Touchstone paths. The `TouchstoneLoader` reads Touchstone files on
# demand, then this notebook materializes the target arrays before fitting because
# the `scikit-learn` Ridge baselines expect in-memory `NumPy` arrays.

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
# Reusable model classes live in `src/sparam_surrogate/models/`. Metrics,
# frequency summaries, and plotting helpers live in
# `src/sparam_surrogate/utils/non_neural_modelling_utils.py`.

# %% [markdown]
# ## Data Loading And Validation
#
# Load the cleaned CSV as three dataset views: training set, evaluation set and
# test set.
# Each view owns one split.

# %%
train_set, val_set, test_set = DLDataset.from_cleaned_csv(processed_dir / CLEANED_CSV)

# %% [markdown]
# Build the in-memory arrays used by `scikit-learn`.
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
# The Ridge baselines need full `NumPy` arrays.
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
print("RIDGE_ALPHA_GRID = ", *RIDGE_ALPHA_GRID)

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
example_scalar_model = ScalarRidgeModel(alphas=(RIDGE_ALPHA_GRID[0],))
print("Example instantiated model:")
print(f"- name={example_scalar_model.name}")
print(f"- alphas={example_scalar_model.alphas}")

# %%
scalar_target_index = 0  # pylint: disable=invalid-name
scalar_target_name = target_names[scalar_target_index]

y_train_scalar = Y_train[:, scalar_target_index]
y_val_scalar = Y_val[:, scalar_target_index]
y_test_scalar = Y_test[:, scalar_target_index]

scalar_model = ScalarRidgeModel()
scalar_model.fit(X_train, y_train_scalar, X_val, y_val_scalar)

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
y_val_pred_scalar = scalar_model.predict(X_val)
y_test_pred_scalar = scalar_model.predict(X_test)

scalar_metrics = pd.DataFrame(
    [
        {"split": "validation", **regression_metrics(y_val_scalar, y_val_pred_scalar)},
        {"split": "test", **regression_metrics(y_test_scalar, y_test_pred_scalar)},
    ]
)
print(scalar_metrics)

# %% [markdown]
# For the single `S7_1_DB` target, the scalar model achieved MAE/RMSE values of
# 7.20/11.06 on the validation set and 7.68/12.45 on the held-out test set. The test
# performance is slightly worse, with an error increase of approximately 6.6% in MAE and
# 12.5% in RMSE. This indicates a modest generalisation gap, but not severe overfitting.
#
# The larger relative increase in RMSE suggests that some test samples contain larger
# prediction errors. Therefore, the baseline model is useful as a pipeline validation
# step, but further analysis is needed to locate the high-error regions, especially
# across frequency and design-parameter space.

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
    test_set.dataframe,
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
vector_model = VectorRidgeModel()
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
#    Across all tested `alpha` values, the MAE stays around `7.325 dB` and the RMSE
#    stays around `11.23 dB`, with only negligible differences between settings. The
#    selected
#    value is `alpha = 1e-05`, the weakest regularisation tested, indicating that the
#    model performs best when Ridge behaves almost like ordinary linear regression.
#
# 2. **Regularisation is not the main issue**
#
#    Increasing `alpha` does not improve validation performance. This suggests that the
#    main limitation is not overfitting, but underfitting: the linear model is too
#    simple to capture the full behaviour.
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
per_target_test_metrics = per_target_metrics(Y_test, Y_test_pred, target_names)

print("Overall vector metrics:", vector_metrics, sep="\n")
print("\nPer-port-pair test metrics:", per_target_test_metrics, sep="\n")

# %% [markdown]
# 1. **Test performance is slightly worse than validation**
#
#    The validation MAE/RMSE are `7.33 dB` and `11.23 dB`, while the test MAE/RMSE
#    increase to `7.80 dB` and `12.61 dB`. This shows a modest generalisation gap, but
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
#    The per-port-pair MAE values are all close, roughly between `7.58 dB` and
#    `7.96 dB`. This means no single output dominates the overall error. The Ridge model
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
    test_set.dataframe,
    Y_test,
    Y_test_pred,
    target_names,
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
    target_names,
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
    test_set.dataframe,
    Y_test,
    Y_test_pred,
    target_names,
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
polynomial_model = PolynomialModel()
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

print("Polynomial validation sweep:")
print(polynomial_validation_results)
print(f"Best polynomial degree: {best_polynomial_degree}")
print(f"Best polynomial alpha: {best_polynomial_alpha:g}")
print(f"Selected expanded polynomial feature count: {expanded_feature_count}")
print("Selected polynomial pipeline:")
print(polynomial_model.pipeline)

# %% [markdown]
# 1. **Degree 4 seems to be the sweet spot**
#
#    The best validation result comes from `degree = 4` with `alpha = 1000`. Degree 3 is
#    close behind, while degree 5 offers no further improvement. Nonlinear interactions
#    help, but extra complexity does not.
#
# 2. **Regularisation helps, but only slightly**
#
#    The best model uses the largest tested `alpha` (`1000`), suggesting stronger
#    regularisation is beneficial. However, validation MAE changes little across `alpha`
#    values, so most of the gain comes from the polynomial features.
#
# 3. **The gain is modest**
#
#    The best polynomial validation MAE is `7.2918 dB`, compared with about `7.3250 dB`
#    for Vector Ridge. The polynomial expansion improves the baseline, but only
#    slightly.
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
per_target_polynomial_metrics = per_target_metrics(
    Y_test,
    Y_test_pred_poly,
    target_names,
)

model_comparison = pd.DataFrame(
    [
        {"model": "Vector Ridge", **regression_metrics(Y_test, Y_test_pred)},
        {"model": "Polynomial", **regression_metrics(Y_test, Y_test_pred_poly)},
    ]
)

per_target_vector_comparison = per_target_test_metrics.copy()
per_target_vector_comparison.insert(0, "model", "Vector Ridge")
per_target_polynomial_comparison = per_target_polynomial_metrics.copy()
per_target_polynomial_comparison.insert(0, "model", "Polynomial")
per_target_model_comparison = pd.concat(
    [per_target_vector_comparison, per_target_polynomial_comparison],
    ignore_index=True,
)

print("Polynomial vector metrics:")
print(polynomial_metrics)
print("\nOverall model comparison:")
print(model_comparison)
print("\nPer-target model comparison:")
print(per_target_model_comparison)

# %% [markdown]
# 1. The Polynomial model performs slightly better than Vector Ridge.
#
#    Test MAE drops from `7.80 dB` to `7.76 dB`, while RMSE decreases from `12.61 dB`
#    to `12.59 dB`. The gain is small, but consistent.
#
# 2. The improvement appears across all six targets.
#
#    Each port-pair prediction improves slightly, suggesting the polynomial features
#    provide a small overall benefit rather than helping only one specific link.
#
# 3. The main limitation remains.
#
#    The improvement is only a few hundredths of a dB, so the model is mainly refining
#    the Ridge baseline rather than addressing its underlying weaknesses.
#
# 4. Overall, Polynomial Ridge is a slightly stronger non-neural baseline, but the
#    modest gain suggests that a more expressive nonlinear model is likely needed for
#    further improvement.
#

# %% [markdown]
# ### 3.3 Plot Polynomial IL Distributions Across Test Designs

# %%
fig_polynomial_distributions = plot_vector_prediction_bands_by_frequency(
    test_set.dataframe,
    Y_test,
    Y_test_pred_poly,
    target_names,
    model_name="Polynomial",
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
    test_set.dataframe,
    Y_test,
    {
        "Vector Ridge": Y_test_pred,
        "Polynomial": Y_test_pred_poly,
    },
    target_names,
)

# %% [markdown]
# 1. The Polynomial model is slightly better overall.
#
#    The Polynomial MAE curve is very close to the Vector Ridge curve, but it is
#    slightly lower in some frequency regions, especially at lower frequencies.
#    This agrees with the numerical results: the Polynomial model improves the
#    overall test MAE slightly, from about `7.80 dB` to `7.76 dB`.
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
# ## 4. Three-Model Comparison On S7_1_DB
#
# The scalar Ridge model only predicts `S7_1_DB`, so the cleanest comparison is
# to evaluate all three fitted models on that shared target only. The scalar
# model contributes its direct prediction. The Vector Ridge and Polynomial Ridge
# models contribute only their `S7_1_DB` output column, even though they were
# trained on all six outputs.

# %%
shared_target_name = scalar_target_name #  It should be "S7_1_DB"
shared_target_index = scalar_target_index

shared_target_true = Y_test[:, shared_target_index]
shared_target_predictions = {
    "Scalar Ridge": y_test_pred_scalar,
    "Vector Ridge": Y_test_pred[:, shared_target_index],
    "Polynomial Ridge": Y_test_pred_poly[:, shared_target_index],
}

# %% [markdown]
# First, compare the three `S7_1_DB` MAE curves by frequency.

# %%
fig_s7_three_model_mae_frequency = plot_shared_target_mae_comparison(
    test_set.dataframe,
    shared_target_true,
    shared_target_predictions,
    shared_target_name,
)

# %% [markdown]
# Second, compare the true distribution curve against the predicted distribution
# from each model. The true curve uses the same median and 10th-90th percentile
# band in all cases, while each model contributes its own predicted median and
# band. This makes it easier to see whether Polynomial Ridge improves the
# `S7_1_DB` curve shape compared with Vector Ridge.

# %%
fig_s7_three_model_distributions = plot_shared_target_prediction_bands(
    test_set.dataframe,
    shared_target_true,
    shared_target_predictions,
    shared_target_name,
)

# %% [markdown]
# 1. **All three models show almost the same frequency-dependent error trend**
#
#    In the MAE-by-frequency plot, the three curves are very close to each
#    other. The error is low at low frequency and rises steadily toward high
#    frequency, reaching about `13–14 dB` near `100 GHz`. This confirms that
#    high-frequency prediction remains the main difficulty for all three
#    models.
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
# 4. **The predicted distribution is still too narrow**
#
#    In the distribution comparison, all three models produce much narrower
#    predicted bands than the true 10th–90th percentile band. This means none
#    of the three models captures the full design-to-design variation. The
#    models still behave like average-response predictors.
#
# The three-model comparison shows a clear progression: Scalar Ridge
# establishes the single-target baseline, Vector Ridge extends the same idea to
# multiple outputs, and Polynomial Ridge adds limited nonlinear flexibility.
# However, the improvement from each step is small. The dominant weakness
# remains high-frequency underfitting and failure to capture the full response
# spread.
#
#

# %% [markdown]
# ### Why Polynomial Ridge Only Gives a Limited Improvement
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
# polynomial expansion produces 44 features:
#
# $$
# 11\ \text{original features} \rightarrow 44\ \text{polynomial features}
# $$
#
# Therefore, for the Polynomial Ridge model in this experiment:
#
# $$
# R=44
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
# curve.
#
# But the design-specific curvature is partly averaged out during global
# training. As a result, the predicted curve becomes smoother than the true
# responses, especially at high frequency where different PCB designs show
# stronger variation.
#
