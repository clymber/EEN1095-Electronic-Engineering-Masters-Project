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

# %% tag=["remove-input"]
"""
Train a fundamental neural-network insertion-loss baseline.
"""

# %load_ext autoreload
# %autoreload 2
# %aimport -pathlib
# %aimport -numpy

# ruff: noqa: E402 -- Configure filtered notebook output before remaining imports.
from sparam_surrogate.config import configure_stdio_relative_path

# Display paths relative to project root or user home.
configure_stdio_relative_path()

# %%
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from sparam_surrogate.config import SurrogateConfig
from sparam_surrogate.data import DLDataset, TouchstoneLoader, random_simu_indices
from sparam_surrogate.models.neural_mlp import (
    PolynomialVectorMLP,
    VectorMLP,
)
from sparam_surrogate.utils.model_prediction_plots import plot_design_prediction_curves
from sparam_surrogate.utils.non_neural_modelling_utils import (
    per_target_metrics,
    regression_metrics,
)

# %% [markdown]
# # Neural Network Vector Baseline
#
# This notebook trains a fundamental fully connected neural-network baseline for
# the same six-output insertion-loss task used by the vector models in
# `nb03_non_neural_modelling.py`.
#
# The model uses the same cleaned design-frequency rows, the same split labels,
# and the same six dB through-path targets. This keeps the neural baseline
# comparable to Vector Ridge, Polynomial Ridge, and Random Forest.
#
# The first neural baseline is intentionally simple:
#
# ```text
# Input:  11 features
#         10 design parameters + FREQ_GHZ
#
# Hidden: Dense(128) + ReLU
#         Dense(128) + ReLU
#         Dense(64)  + ReLU
#
# Output: Dense(6), linear activation
#         six IL targets in dB
# ```
#
# Features and targets are scaled with train-only `StandardScaler` instances
# before training. Keras therefore optimizes MSE in scaled target units. Final
# predictions are inverse-transformed back to dB before computing MAE/RMSE and
# before plotting.

# %%
cfg = SurrogateConfig.from_csv()
random_seed = cfg.project.seed
neural_mlp_config = cfg.models.neural_mlp
polynomial_neural_mlp_config = cfg.models.polynomial_neural_mlp
keras.utils.set_random_seed(random_seed)

print(f"Name of raw dataset: {cfg.dataset.name}")
print(f"Raw data directory: {cfg.dataset.path}")
print(f"Processed directory: {cfg.paths.processed_data}")
print(f"Configured IL port pairs: {cfg.dataset.ports}")
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow physical devices: {tf.config.list_physical_devices()}")
print(f"Random seed: {random_seed}")
print(f"Batch size: {neural_mlp_config.batch_size}")
print(f"Max epochs: {neural_mlp_config.epochs}")
print(f"Adam learning rate: {neural_mlp_config.learning_rate:g}")
print(f"Gradient clip norm: {neural_mlp_config.gradient_clip_norm:g}")
print(f"Early stopping patience: {neural_mlp_config.early_stopping_patience}")
print(f"Reduce LR patience: {neural_mlp_config.reduce_lr_patience}")
print(f"Reduce LR factor: {neural_mlp_config.reduce_lr_factor:g}")
print(f"Minimum learning rate: {neural_mlp_config.min_learning_rate:g}")
print(f"Polynomial neural degree: {polynomial_neural_mlp_config.polynomial_degree}")

# %% [markdown]
# ## 1. Vector Data Loading
#
# The target loader matches the vector dB experiment in `nb03`. The cached target
# arrays are reused when they are newer than the cleaned CSV, so repeated neural
# experiments do not need to parse every Touchstone file again.

# %%
vector_db_loader = TouchstoneLoader("vector", cfg, "db", 8)
vector_target_names = tuple(vector_db_loader.target_names)

vector_train_set, vector_val_set, vector_test_set = DLDataset.from_cleaned_csv(
    cfg.preprocessing.processed_csv,
    target_loader=vector_db_loader,
    cache=True,
)

print(f"Number of training   samples: {len(vector_train_set)}")
print(f"Number of validation samples: {len(vector_val_set)}")
print(f"Number of test       samples: {len(vector_test_set)}")
print("Vector target names:", ", ".join(vector_target_names))

# %%
# pylint: disable=invalid-name
X_train = vector_train_set.features
X_val = vector_val_set.features
X_test = vector_test_set.features

Y_train = vector_train_set.targets
Y_val = vector_val_set.targets
Y_test = vector_test_set.targets
# pylint: enable=invalid-name

print(f"Shape of training   features: {X_train.shape}")
print(f"Shape of validation features: {X_val.shape}")
print(f"Shape of test       features: {X_test.shape}")
print(f"Shape of training   targets: {Y_train.shape}")
print(f"Shape of validation targets: {Y_val.shape}")
print(f"Shape of test       targets: {Y_test.shape}")

if Y_train.shape[1] != len(vector_target_names):
    raise RuntimeError("Training targets do not match configured target names.")
if Y_val.shape[1] != len(vector_target_names):
    raise RuntimeError("Validation targets do not match configured target names.")
if Y_test.shape[1] != len(vector_target_names):
    raise RuntimeError("Test targets do not match configured target names.")

# %%
vector_db_loader.clear_cache()


# %% [markdown]
# ## 2. Model Classes
#
# Reusable neural model classes live in `src/sparam_surrogate/models/`.
# `VectorMLP` and `PolynomialVectorMLP` subclass the common `SparamModel`
# interface, so shared helpers can call `predict`, `evaluate`, and
# `model_name` without special cases.


# %% [markdown]
# ## 3. Train Neural Baseline
#
# The Keras loss shown during fitting is MSE after target scaling. It is useful
# for optimization and early stopping, but it is not a dB-space metric. All final
# tables below use inverse-transformed predictions.

# %%
neural_model = VectorMLP.from_config(neural_mlp_config)
neural_model.fit(X_train, Y_train, X_val, Y_val)

# %%
neural_model.keras_model.summary()
print("Keras history losses are MSE values in scaled target units.")

# %%
if neural_model.history is None:
    raise RuntimeError("Neural model did not record training history.")

fig_training_history = neural_model.plot_training_history()

# %%
selected_simu_indices = random_simu_indices(vector_test_set, 5, seed=random_seed)
fig_random_neural_design_curves = plot_design_prediction_curves(
    neural_model,
    vector_test_set,
    vector_db_loader,
    selected_simu_indices,
)

# %% [markdown]
# ## 4. Neural Baseline Evaluation
#
# `VectorMLP.predict()` transforms features with the train-fitted
# input scaler, calls the Keras model, and inverse-transforms the six scaled
# outputs back to dB. The metrics below are therefore directly comparable to
# the non-neural dB-space metrics in `nb03`.

# %%
# pylint: disable=invalid-name
Y_train_pred_nn = neural_model.predict(X_train)
Y_val_pred_nn = neural_model.predict(X_val)
Y_test_pred_nn = neural_model.predict(X_test)
# pylint: enable=invalid-name

assert Y_test_pred_nn.shape == Y_test.shape

neural_metrics = pd.DataFrame(
    [
        {"split": "train", **regression_metrics(Y_train, Y_train_pred_nn)},
        {"split": "validation", **regression_metrics(Y_val, Y_val_pred_nn)},
        {"split": "test", **regression_metrics(Y_test, Y_test_pred_nn)},
    ]
)

per_target_neural_metrics = per_target_metrics(
    Y_test,
    Y_test_pred_nn,
    vector_target_names,
)

negative_prediction_count = int(np.sum(Y_test_pred_nn < 0.0))
negative_prediction_ratio = negative_prediction_count / Y_test_pred_nn.size
minimum_predicted_il = float(np.min(Y_test_pred_nn))

print("Neural MLP vector metrics:", neural_metrics, sep="\n")
print("\nNeural MLP per-target test metrics:", per_target_neural_metrics, sep="\n")
print(f"\nNegative IL prediction count: {negative_prediction_count:,}")
print(f"Negative IL prediction ratio: {negative_prediction_ratio:.4%}")
print(f"Minimum predicted IL: {minimum_predicted_il:.4f} dB")

# %% [markdown]
# ## Small modelling experiments

# %%
sample_sizes = [256*4, 256*8, 256*16, 256*32, 256*40, 256*64, 256*128, 256*256]
small_models: list[VectorMLP] = [
    VectorMLP(
        batch_size=512,
        epochs=100,
        learning_rate=1e-4,
        gradient_clip_norm=1.0,
        random_state=random_seed,
    )
    for _ in sample_sizes
]


def fit_small_model(
    n_samples: int,
    small_model: VectorMLP,
) -> tuple[int, VectorMLP]:
    """
    Fit one small neural baseline and return it with its sample count.
    """
    print(f"\nTraining VectorMLP with {n_samples} samples...")

    X_small = X_train[:n_samples]
    Y_small = Y_train[:n_samples]

    small_model.fit(
        X_small,
        Y_small,
        X_small,
        Y_small,
        verbose=0,
    )
    return n_samples, small_model


# %%
for n_samples, small_model in zip(sample_sizes, small_models, strict=True):
    try:
        time_begin = pd.Timestamp.now()
        fit_small_model(n_samples, small_model)
        time_end = pd.Timestamp.now()

        elapsed_time = time_end - time_begin
        print(f"Completed VectorMLP with {n_samples} samples in {elapsed_time}.")

        if small_model.history is None:
            raise RuntimeError(
                f"Small model {n_samples} did not record training history."
            )
        print(f"Plotting training history for small model with {n_samples} samples...")
        fig_training_history = small_model.plot_training_history()
        plt.show()
    except Exception as exc:
        raise RuntimeError(
            f"Training failed for small model with {n_samples} samples."
        ) from exc

print("\nTraining complete for all sample sizes.")

# %%
# Print the number of epochs required to reach the best validation loss for each model.
best_epochs = []

print("\nBest validation loss epochs for small models:")
for n_samples, small_model in zip(sample_sizes, small_models, strict=True):
    if small_model.history is None:
        raise RuntimeError(f"Small model {n_samples} did not record training history.")
    history_frame = pd.DataFrame(small_model.history.history)
    best_epoch = int(history_frame["val_loss"].idxmin()) + 1
    best_val_loss = float(history_frame["val_loss"].min())
    best_epochs.append(best_epoch)

print("|" + "|".join(f"{best_epoch:>7}" for best_epoch in best_epochs) + "|")

# %% [markdown]
# | 1024  | 2048  | 4096  | 8192  | 10240 | 16384 | 32768 | 65536 |
# | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
# |    100|     69|     41|     19|     17|     10|      5|      4|
# |     22|     62|     73|     15|     12|     13|      7|      5|
# |     93|     40|     40|     15|     13|      6|      4|      2|
# |     73|     62|     83|     32|     94|     32|     22|      4|

# %% [markdown]
# ## Polynomial Neural Variant
#
# This follow-up trains the same MLP on the powers-only polynomial feature
# representation used by the `nb03` Polynomial Ridge baseline. The raw
# design-frequency features are scaled first, expanded as
# `[x, x^2, ...]` without cross terms, then scaled again before Keras sees them.
# Target scaling, callbacks, optimizer settings, and train/validation/test splits
# stay aligned with the raw-feature neural baseline.

# %%
polynomial_neural_model = PolynomialVectorMLP.from_config(
    polynomial_neural_mlp_config
)
polynomial_neural_model.fit(X_train, Y_train, X_val, Y_val)

# %%
if polynomial_neural_model.expanded_feature_count_ is None:
    raise RuntimeError("Polynomial neural model did not record feature count.")

print(
    "Polynomial neural expanded feature count: "
    f"{X_train.shape[1]} -> {polynomial_neural_model.expanded_feature_count_}"
)
polynomial_neural_model.keras_model.summary()
print("Keras history losses are MSE values in scaled target units.")

# %%
if polynomial_neural_model.history is None:
    raise RuntimeError("Polynomial neural model did not record training history.")

fig_polynomial_mlp_history = polynomial_neural_model.plot_training_history()

# %%
fig_random_polynomial_neural_design_curves = plot_design_prediction_curves(
    polynomial_neural_model,
    vector_test_set,
    vector_db_loader,
    selected_simu_indices,
)

# %% [markdown]
# ### Polynomial Neural Evaluation
#
# `PolynomialVectorMLP.predict()` applies the train-fitted input
# scaler, polynomial expansion, expanded-feature scaler, and target inverse
# transform. The metrics below are therefore reported in original dB units.

# %%
# pylint: disable=invalid-name
Y_train_pred_poly_nn = polynomial_neural_model.predict(X_train)
Y_val_pred_poly_nn = polynomial_neural_model.predict(X_val)
Y_test_pred_poly_nn = polynomial_neural_model.predict(X_test)
# pylint: enable=invalid-name

assert Y_test_pred_poly_nn.shape == Y_test.shape

polynomial_neural_metrics = pd.DataFrame(
    [
        {"split": "train", **regression_metrics(Y_train, Y_train_pred_poly_nn)},
        {"split": "validation", **regression_metrics(Y_val, Y_val_pred_poly_nn)},
        {"split": "test", **regression_metrics(Y_test, Y_test_pred_poly_nn)},
    ]
)

per_target_polynomial_neural_metrics = per_target_metrics(
    Y_test,
    Y_test_pred_poly_nn,
    vector_target_names,
)

polynomial_neural_negative_count = int(np.sum(Y_test_pred_poly_nn < 0.0))
polynomial_neural_negative_ratio = (
    polynomial_neural_negative_count / Y_test_pred_poly_nn.size
)
polynomial_neural_minimum_il = float(np.min(Y_test_pred_poly_nn))

print("Polynomial Neural MLP vector metrics:", polynomial_neural_metrics, sep="\n")
print(
    "\nPolynomial Neural MLP per-target test metrics:",
    per_target_polynomial_neural_metrics,
    sep="\n",
)
print(f"\nNegative IL prediction count: {polynomial_neural_negative_count:,}")
print(f"Negative IL prediction ratio: {polynomial_neural_negative_ratio:.4%}")
print(f"Minimum predicted IL: {polynomial_neural_minimum_il:.4f} dB")

# %%
neural_test_metrics = neural_metrics.loc[neural_metrics["split"] == "test"].iloc[0]
polynomial_neural_test_metrics = polynomial_neural_metrics.loc[
    polynomial_neural_metrics["split"] == "test"
].iloc[0]

neural_variant_comparison = pd.DataFrame(
    [
        {"model": "Vector Ridge", "MAE": 7.4740, "RMSE": 11.0796},
        {"model": "Polynomial Ridge", "MAE": 7.4269, "RMSE": 11.0532},
        {
            "model": "Neural MLP",
            "MAE": float(neural_test_metrics["MAE"]),
            "RMSE": float(neural_test_metrics["RMSE"]),
        },
        {
            "model": "Polynomial Neural MLP",
            "MAE": float(polynomial_neural_test_metrics["MAE"]),
            "RMSE": float(polynomial_neural_test_metrics["RMSE"]),
        },
    ]
)

print("Neural variant test comparison:", neural_variant_comparison, sep="\n")
