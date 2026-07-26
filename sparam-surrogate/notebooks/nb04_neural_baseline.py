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

from sparam_surrogate.config import SurrogateConfig
from sparam_surrogate.data import DLDataset, TouchstoneLoader, random_simu_indices
from sparam_surrogate.models.neural_mlp import (
    PolynomialVectorMLP,
    VectorMLP,
)
from sparam_surrogate.outputs.runner import ModelRunRunner
from sparam_surrogate.utils.json_io import read_json
from sparam_surrogate.utils.model_prediction_plots import plot_design_prediction_curves
from sparam_surrogate.utils.non_neural_modelling_utils import per_target_metrics


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
cfg = SurrogateConfig.from_config()
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
vector_data_interface = {
    "dataset_name": cfg.dataset.name,
    "input_features": vector_train_set.feature_columns,
    "target_names": vector_target_names,
    "target_scope": "vector",
    "target_units": "dB",
}

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
neural_runner = ModelRunRunner(cfg, VectorMLP.from_config(neural_mlp_config))
neural_model = neural_runner.train(X_train, Y_train, X_val, Y_val)

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
neural_design_curve_path = neural_runner.manager.save_figure(
    fig_random_neural_design_curves,
    "selected_design_curves_magnitude_db.png",
)
print(f"Saved Neural MLP design-curve plot: {neural_design_curve_path}")

# %% [markdown]
# ## 4. Neural Baseline Evaluation
#
# `VectorMLP.predict()` transforms features with the train-fitted
# input scaler, calls the Keras model, and inverse-transforms the six scaled
# outputs back to dB. The metrics below are therefore directly comparable to
# the non-neural dB-space metrics in `nb03`.

# %%
neural_train_metrics = neural_model.evaluate(X_train, Y_train)
neural_validation_metrics = neural_runner.validate(X_val, Y_val)
neural_test_metrics = neural_runner.test(X_test, Y_test)

# pylint: disable=invalid-name
Y_val_pred_nn = neural_model.predict(X_val)
Y_test_pred_nn = neural_model.predict(X_test)
# pylint: enable=invalid-name

assert Y_test_pred_nn.shape == Y_test.shape

neural_metrics = pd.DataFrame(
    [
        {"split": "train", **neural_train_metrics},
        {"split": "validation", **neural_validation_metrics},
        {"split": "test", **neural_test_metrics},
    ]
)

per_target_neural_metrics = per_target_metrics(
    Y_test,
    Y_test_pred_nn,
    vector_target_names,
)
per_target_neural_validation_metrics = per_target_metrics(
    Y_val,
    Y_val_pred_nn,
    vector_target_names,
)
neural_per_target_run_metrics = per_target_split_metrics(
    per_target_neural_validation_metrics,
    per_target_neural_metrics,
)

negative_prediction_count = int(np.sum(Y_test_pred_nn < 0.0))
negative_prediction_ratio = negative_prediction_count / Y_test_pred_nn.size
minimum_predicted_il = float(np.min(Y_test_pred_nn))

print("Neural MLP vector metrics:", neural_metrics, sep="\n")
print("\nNeural MLP per-target test metrics:", per_target_neural_metrics, sep="\n")
print(f"\nNegative IL prediction count: {negative_prediction_count:,}")
print(f"Negative IL prediction ratio: {negative_prediction_ratio:.4%}")
print(f"Minimum predicted IL: {minimum_predicted_il:.4f} dB")

# %%
neural_artifact_paths = neural_runner.persist(
    data_interface=vector_data_interface,
    extra_metrics={"per_target": neural_per_target_run_metrics},
    metric_units={"MAE": "dB", "RMSE": "dB"},
)
neural_manifest = read_json(neural_artifact_paths["manifest"])

print(f"Neural MLP run directory: {neural_runner.manager.run_dir}")
print("Neural MLP artifacts:")
for artifact_name, artifact_path in neural_artifact_paths.items():
    print(f"- {artifact_name}: {artifact_path}")
print("\nNeural MLP manifest figures:", neural_manifest.get("figures", {}))

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
polynomial_neural_runner = ModelRunRunner(
    cfg,
    PolynomialVectorMLP.from_config(polynomial_neural_mlp_config),
)
polynomial_neural_model = polynomial_neural_runner.train(
    X_train,
    Y_train,
    X_val,
    Y_val,
)

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
polynomial_neural_design_curve_path = polynomial_neural_runner.manager.save_figure(
    fig_random_polynomial_neural_design_curves,
    "selected_design_curves_magnitude_db.png",
)
print(
    "Saved Polynomial Neural MLP design-curve plot: "
    f"{polynomial_neural_design_curve_path}"
)

# %% [markdown]
# ### Polynomial Neural Evaluation
#
# `PolynomialVectorMLP.predict()` applies the train-fitted input
# scaler, polynomial expansion, expanded-feature scaler, and target inverse
# transform. The metrics below are therefore reported in original dB units.

# %%
polynomial_neural_train_metrics = polynomial_neural_model.evaluate(
    X_train,
    Y_train,
)
polynomial_neural_validation_metrics = polynomial_neural_runner.validate(
    X_val,
    Y_val,
)
polynomial_neural_test_metrics = polynomial_neural_runner.test(X_test, Y_test)

# pylint: disable=invalid-name
Y_val_pred_poly_nn = polynomial_neural_model.predict(X_val)
Y_test_pred_poly_nn = polynomial_neural_model.predict(X_test)
# pylint: enable=invalid-name

assert Y_test_pred_poly_nn.shape == Y_test.shape

polynomial_neural_metrics = pd.DataFrame(
    [
        {"split": "train", **polynomial_neural_train_metrics},
        {"split": "validation", **polynomial_neural_validation_metrics},
        {"split": "test", **polynomial_neural_test_metrics},
    ]
)

per_target_polynomial_neural_metrics = per_target_metrics(
    Y_test,
    Y_test_pred_poly_nn,
    vector_target_names,
)
per_target_polynomial_neural_validation_metrics = per_target_metrics(
    Y_val,
    Y_val_pred_poly_nn,
    vector_target_names,
)
polynomial_neural_per_target_run_metrics = per_target_split_metrics(
    per_target_polynomial_neural_validation_metrics,
    per_target_polynomial_neural_metrics,
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
polynomial_neural_artifact_paths = polynomial_neural_runner.persist(
    data_interface=vector_data_interface,
    extra_metrics={"per_target": polynomial_neural_per_target_run_metrics},
    metric_units={"MAE": "dB", "RMSE": "dB"},
)
polynomial_neural_manifest = read_json(
    polynomial_neural_artifact_paths["manifest"]
)

print(
    "Polynomial Neural MLP run directory: "
    f"{polynomial_neural_runner.manager.run_dir}"
)
print("Polynomial Neural MLP artifacts:")
for artifact_name, artifact_path in polynomial_neural_artifact_paths.items():
    print(f"- {artifact_name}: {artifact_path}")
print(
    "\nPolynomial Neural MLP manifest figures:",
    polynomial_neural_manifest.get("figures", {}),
)

# %%
comparison_model_names = {
    "vector_ridge": "Vector Ridge",
    "polynomial_ridge": "Polynomial Ridge",
    "neural_mlp": "Neural MLP",
    "polynomial_neural_mlp": "Polynomial Neural MLP",
}
vector_latest_benchmark_path = (
    cfg.paths.benchmarks / "vector_magnitude_db_latest.csv"
)
neural_variant_comparison = (
    pd.read_csv(vector_latest_benchmark_path)
    .set_index("model_name")
    .loc[
        list(comparison_model_names),
        ["test_mae_db", "test_rmse_db"],
    ]
    .rename(
        index=comparison_model_names,
        columns={"test_mae_db": "MAE", "test_rmse_db": "RMSE"},
    )
    .rename_axis("model")
    .reset_index()
)

print("Neural variant test comparison:", neural_variant_comparison, sep="\n")

# %% [markdown]
# ## Persisted Output Summary
#
# The two main neural runs above now write into the planned output hierarchy:
#
# - `outputs/runs/<run_id>/` for immutable artifacts.
# - `outputs/models/*.json` for latest and selected model pointers.
# - `outputs/benchmarks/*.csv` for notebook-to-notebook comparison rows.

# %%
latest_model_registry = read_json(cfg.paths.models / "latest.json")
selected_model_registry = read_json(cfg.paths.models / "selected.json")
s7_latest_benchmark_path = cfg.paths.benchmarks / "s7_1_magnitude_db_latest.csv"
s7_selected_benchmark_path = (
    cfg.paths.benchmarks / "s7_1_magnitude_db_selected.csv"
)
vector_latest_benchmark_path = (
    cfg.paths.benchmarks / "vector_magnitude_db_latest.csv"
)
vector_selected_benchmark_path = (
    cfg.paths.benchmarks / "vector_magnitude_db_selected.csv"
)

print("Latest model registry entries:")
print(sorted(latest_model_registry.get("models", {})))
print("\nSelected model registry entries:")
print(sorted(selected_model_registry.get("models", {})))

if s7_latest_benchmark_path.is_file():
    print("\nLatest S7_1 benchmark:")
    print(pd.read_csv(s7_latest_benchmark_path))

if s7_selected_benchmark_path.is_file():
    print("\nSelected S7_1 benchmark:")
    print(pd.read_csv(s7_selected_benchmark_path))

if vector_latest_benchmark_path.is_file():
    print("\nLatest vector benchmark:")
    print(pd.read_csv(vector_latest_benchmark_path))

if vector_selected_benchmark_path.is_file():
    print("\nSelected vector benchmark:")
    print(pd.read_csv(vector_selected_benchmark_path))
