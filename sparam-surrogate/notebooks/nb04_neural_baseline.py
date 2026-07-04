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
from typing import Any

from sparam_surrogate.config import configure_stdio_relative_path

# Display paths relative to project root or user home.
configure_stdio_relative_path()

# %%
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, layers
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler

from sparam_surrogate.config import SurrogateConfig
from sparam_surrogate.data import DLDataset, TouchstoneLoader, random_simu_indices
from sparam_surrogate.models import SparamModel
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
keras.utils.set_random_seed(random_seed)

# MAX_EPOCHS = cfg.training.epochs
PREDICTION_BATCH_SIZE = 4096
BATCH_SIZE = 512
LEARNING_RATE = 3e-5
GRADIENT_CLIP_NORM = 0.5
EARLY_STOPPING_PATIENCE = 18
REDUCE_LR_PATIENCE = 6
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6
MAX_EPOCHS = 100
# Keras accepts float callback deltas, but Pyright infers int from the
# EarlyStopping runtime default of 0.
CALLBACK_MIN_DELTA: Any = 1e-4

print(f"Name of raw dataset: {cfg.dataset.name}")
print(f"Raw data directory: {cfg.dataset.path}")
print(f"Processed directory: {cfg.paths.processed_data}")
print(f"Configured IL port pairs: {cfg.dataset.ports}")
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow physical devices: {tf.config.list_physical_devices()}")
print(f"Random seed: {random_seed}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {MAX_EPOCHS}")
print(f"Adam learning rate: {LEARNING_RATE:g}")
print(f"Gradient clip norm: {GRADIENT_CLIP_NORM:g}")
print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print(f"Reduce LR patience: {REDUCE_LR_PATIENCE}")
print(f"Reduce LR factor: {REDUCE_LR_FACTOR:g}")
print(f"Minimum learning rate: {MIN_LEARNING_RATE:g}")

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
print(f"Vector Touchstone cache before clearing: {vector_db_loader.cache_info()}")
vector_db_loader.clear_cache()
print(f"Vector Touchstone cache after clearing: {vector_db_loader.cache_info()}")


# %% [markdown]
# ## 2. Model Definition
#
# `NeuralVectorBaseline` subclasses the same `SparamModel` interface used by the
# non-neural baselines. This means common helpers can call `predict`,
# `evaluate`, and `model_name` without special cases.

# %%
def build_vector_mlp(input_width: int, output_width: int) -> keras.Model:
    inputs = Input(
        shape=(input_width,),
        name="design_frequency_features",
    )

    hidden = layers.Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        name="dense_128_a",
    )(inputs)

    hidden = layers.Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        name="dense_128_b",
    )(hidden)

    hidden = layers.Dense(
        64,
        activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        name="dense_64",
    )(hidden)

    outputs = layers.Dense(
        output_width,
        activation="linear",
        kernel_initializer=keras.initializers.RandomNormal(
            mean=0.0,
            stddev=1e-2
        ), # type: ignore[arg-type]
        bias_initializer="zeros",
        name="s_db_outputs",
    )(hidden)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="vector_mlp_baseline",
    )


class NeuralVectorBaseline(SparamModel):
    """
    Keras MLP wrapper implementing the common surrogate model interface.
    """

    name = "neural_mlp"

    def __init__(
        self,
        *,
        batch_size: int = BATCH_SIZE,
        epochs: int = MAX_EPOCHS,
        prediction_batch_size: int = PREDICTION_BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        gradient_clip_norm: float = GRADIENT_CLIP_NORM,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
        reduce_lr_patience: int = REDUCE_LR_PATIENCE,
        reduce_lr_factor: float = REDUCE_LR_FACTOR,
        min_learning_rate: float = MIN_LEARNING_RATE,
        random_state: int = random_seed,
    ) -> None:
        """
        Store training controls and scaler state.
        """
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.prediction_batch_size = int(prediction_batch_size)
        self.learning_rate = float(learning_rate)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.early_stopping_patience = int(early_stopping_patience)
        self.reduce_lr_patience = int(reduce_lr_patience)
        self.reduce_lr_factor = float(reduce_lr_factor)
        self.min_learning_rate = float(min_learning_rate)
        self.random_state = int(random_state)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model: keras.Model | None = None
        self.history: keras.callbacks.History | None = None

    def model_name(self) -> str:
        """
        Return the plot label with the expected MLP capitalization.
        """
        return "Neural MLP"

    def fit(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,  # pylint: disable=invalid-name
        y_val: np.ndarray | None = None,
        verbose: str | int = 2,
    ) -> "NeuralVectorBaseline":
        """
        Fit the neural baseline using scaled features and scaled targets.
        """
        if X_val is None or y_val is None:
            raise ValueError("NeuralVectorBaseline requires validation data.")

        keras.utils.set_random_seed(self.random_state)
        X_train_scaled = self.x_scaler.fit_transform(  # pylint: disable=invalid-name
            np.asarray(X_train, dtype=float)
        ).astype(np.float32)
        X_val_scaled = self.x_scaler.transform(  # pylint: disable=invalid-name
            np.asarray(X_val, dtype=float)
        ).astype(np.float32)
        y_train_scaled = self.y_scaler.fit_transform(
            np.asarray(y_train, dtype=float)
        ).astype(np.float32)
        y_val_scaled = self.y_scaler.transform(np.asarray(y_val, dtype=float)).astype(
            np.float32
        )

        self.model = build_vector_mlp(
            input_width=X_train_scaled.shape[1],
            output_width=y_train_scaled.shape[1],
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=self.gradient_clip_norm,
            ),
            loss="mse",
            steps_per_execution=8,
        )

        self.history = self.model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=self.reduce_lr_factor,
                    patience=self.reduce_lr_patience,
                    min_delta=CALLBACK_MIN_DELTA,
                    min_lr=self.min_learning_rate,
                    verbose=1,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    min_delta=CALLBACK_MIN_DELTA,
                    restore_best_weights=True,
                    verbose=1,
                ),
            ],
            shuffle=True,
            verbose=verbose,  # type: ignore[assignment]
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Return inverse-transformed dB predictions from the fitted Keras model.
        """
        X_scaled = self.x_scaler.transform(np.asarray(X, dtype=float)).astype(  # noqa: N806
            np.float32
        )
        y_pred_scaled = self.keras_model.predict(
            X_scaled,
            batch_size=self.prediction_batch_size,
            verbose=0,  # type: ignore[assignment]
        )
        return self.y_scaler.inverse_transform(np.asarray(y_pred_scaled, dtype=float))

    @property
    def keras_model(self) -> keras.Model:
        """
        Return the fitted Keras model.
        """
        if self.model is None:
            raise RuntimeError("NeuralVectorBaseline must be fitted before prediction.")
        return self.model


def plot_training_history(history: keras.callbacks.History) -> Figure:
    """
    Plot scaled-unit training and validation MSE histories.
    """
    history_frame = pd.DataFrame(history.history)
    best_epoch = int(history_frame["val_loss"].idxmin()) + 1
    best_val_loss = float(history_frame["val_loss"].min())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history_frame.index + 1, history_frame["loss"], label="train loss")
    ax.plot(history_frame.index + 1, history_frame["val_loss"], label="val loss")
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.scatter(
        [best_epoch],
        [best_val_loss],
        color="black",
        s=30,
        zorder=3,
        label=f"best val epoch {best_epoch}",
    )
    ax.set_title("Neural MLP Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled target units)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# %% [markdown]
# ## 3. Train Neural Baseline
#
# The Keras loss shown during fitting is MSE after target scaling. It is useful
# for optimization and early stopping, but it is not a dB-space metric. All final
# tables below use inverse-transformed predictions.

# %%
neural_model = NeuralVectorBaseline()
neural_model.fit(X_train, Y_train, X_val, Y_val)

# %%
neural_model.keras_model.summary()
print("Keras history losses are MSE values in scaled target units.")

# %%
if neural_model.history is None:
    raise RuntimeError("Neural model did not record training history.")

fig_training_history = plot_training_history(neural_model.history)

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
# `NeuralVectorBaseline.predict()` transforms features with the train-fitted
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
small_models: list[NeuralVectorBaseline] = [
    NeuralVectorBaseline(
        batch_size=512,
        epochs=100,
        learning_rate=1e-4,
        gradient_clip_norm=1.0,
    )
    for _ in sample_sizes
]


def fit_small_model(
    n_samples: int,
    small_model: NeuralVectorBaseline,
) -> tuple[int, NeuralVectorBaseline]:
    """
    Fit one small neural baseline and return it with its sample count.
    """
    print(f"\nTraining NeuralVectorBaseline with {n_samples} samples...")

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
        print(f"Completed NeuralVectorBaseline with {n_samples} samples in {elapsed_time}.")

        if small_model.history is None:
            raise RuntimeError(f"Small model {n_samples} did not record training history.")
        print(f"Plotting training history for small model with {n_samples} samples...")
        fig_training_history = plot_training_history(small_model.history)
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
