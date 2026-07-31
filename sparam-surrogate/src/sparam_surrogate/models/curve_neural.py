"""
S-TCNN-style neural decoder for whole insertion-loss curves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import keras
import numpy as np
from keras import Input, layers
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler

from sparam_surrogate.models.neural import NeuralModel
from sparam_surrogate.utils.non_neural_modelling_utils import regression_metrics

if TYPE_CHECKING:
    from sparam_surrogate.config.surrogate_config import CurveNeuralModelConfig

FrequencyEncoding = Literal["linear", "fourier"]
CurveFrequencyEncoding = Literal["none", "linear", "fourier"]

# Keras accepts float callback deltas, but its EarlyStopping stub infers int
# from the runtime default.
CALLBACK_MIN_DELTA: Any = 1e-3


def frequency_features(
    frequencies_ghz: np.ndarray,
    encoding: FrequencyEncoding,
    *,
    fourier_order: int = 4,
) -> np.ndarray:
    """
    Return normalized explicit features for one frequency grid.
    """
    frequencies = np.asarray(frequencies_ghz, dtype=float)
    if frequencies.ndim != 1 or frequencies.size < 2:
        raise ValueError(
            "Frequency grid must be one-dimensional with at least two values."
        )
    if not np.isfinite(frequencies).all():
        raise ValueError("Frequency grid must contain only finite values.")

    frequency_range = np.ptp(frequencies)
    if frequency_range == 0.0:
        raise ValueError("Frequency grid must not be constant.")

    normalized = (
        2.0 * (frequencies - frequencies.min()) / frequency_range - 1.0
    )
    if encoding == "linear":
        return normalized[:, np.newaxis].astype(np.float32)
    if encoding != "fourier":
        raise ValueError(f"Unknown frequency encoding: {encoding!r}.")
    if fourier_order < 1:
        raise ValueError("Fourier order must be at least one.")

    features = [normalized]
    for harmonic in range(1, fourier_order + 1):
        features.extend(
            (
                np.sin(np.pi * harmonic * normalized),
                np.cos(np.pi * harmonic * normalized),
            )
        )
    return np.column_stack(features).astype(np.float32)


def build_curve_decoder(
    *,
    input_width: int,
    n_frequencies: int,
    n_targets: int,
    latent_dim: int = 128,
    decoder_channels: Sequence[int] = (128, 64, 32),
    kernel_size: int = 5,
    frequency_feature_width: int = 0,
) -> keras.Model:
    """
    Build a design-to-curve decoder with optional explicit frequency inputs.
    """
    channels = tuple(int(width) for width in decoder_channels)
    if len(channels) != 3:
        raise ValueError("Exactly three decoder channel widths are required.")
    if n_frequencies <= 0 or n_frequencies % 8:
        raise ValueError("Frequency count must be positive and divisible by eight.")
    if min(input_width, n_targets, latent_dim, kernel_size, *channels) <= 0:
        raise ValueError("Model dimensions and kernel size must be positive.")
    if frequency_feature_width < 0:
        raise ValueError("Frequency feature width must not be negative.")

    initial_length = n_frequencies // 8
    design_input = Input(shape=(input_width,), name="design_parameters")
    decoded = layers.Dense(
        latent_dim,
        activation="relu",
        kernel_initializer="he_normal",
        name="dense_encoder",
    )(design_input)
    decoded = layers.Dense(
        initial_length * channels[0],
        activation="relu",
        kernel_initializer="he_normal",
        name="dense_projection",
    )(decoded)
    decoded = layers.Reshape(
        (initial_length, channels[0]),
        name="initial_curve_features",
    )(decoded)

    for stage, width in enumerate(channels, start=1):
        decoded = layers.Conv1DTranspose(
            width,
            kernel_size,
            strides=2,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name=f"upsample_{stage}",
        )(decoded)

    model_inputs: keras.KerasTensor | list[keras.KerasTensor] = design_input
    if frequency_feature_width:
        frequency_input = Input(
            shape=(n_frequencies, frequency_feature_width),
            name="frequency_features",
        )
        decoded = layers.Concatenate(name="add_frequency_features")(
            [decoded, frequency_input]
        )
        model_inputs = [design_input, frequency_input]

    decoded = layers.Conv1D(
        channels[-1],
        kernel_size,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="curve_refinement",
    )(decoded)
    outputs = layers.Conv1D(
        n_targets,
        1,
        padding="same",
        activation="linear",
        name="insertion_loss_outputs",
    )(decoded)

    suffix = "frequency_aware" if frequency_feature_width else "baseline"
    return keras.Model(
        inputs=model_inputs,
        outputs=outputs,
        name=f"stcnn_style_curve_decoder_{suffix}",
    )


@keras.saving.register_keras_serializable(package="sparam_surrogate")
class CurveAwareMSE(keras.losses.Loss):
    """
    Combine point-wise curve MSE with an optional first-difference term.
    """

    def __init__(
        self,
        derivative_weight: float = 0.0,
        reduction: Any = "sum_over_batch_size",
        name: str = "curve_aware_mse",
    ) -> None:
        """
        Store the derivative-loss weight.
        """
        super().__init__(reduction=reduction, name=name)
        if derivative_weight < 0.0:
            raise ValueError("Derivative loss weight must not be negative.")
        self.derivative_weight = float(derivative_weight)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """
        Return per-design data and first-difference mean-squared errors.
        """
        error = y_pred - y_true
        data_mse = keras.ops.mean(keras.ops.square(error), axis=(1, 2))
        derivative_error = error[:, 1:, :] - error[:, :-1, :]
        derivative_mse = keras.ops.mean(
            keras.ops.square(derivative_error),
            axis=(1, 2),
        )
        return data_mse + self.derivative_weight * derivative_mse

    def get_config(self) -> dict[str, Any]:
        """
        Return the serializable loss configuration.
        """
        return {
            **super().get_config(),
            "derivative_weight": self.derivative_weight,
        }


@keras.saving.register_keras_serializable(package="sparam_surrogate")
class UnscaledMeanAbsoluteError(keras.metrics.Mean):
    """
    Measure curve MAE in original target units from standardized targets.
    """

    def __init__(
        self,
        target_scale: Sequence[float],
        name: str = "mae_db",
        dtype: Any = None,
    ) -> None:
        """
        Store the channel-wise target scales used to undo scaled errors.
        """
        scale = np.asarray(target_scale, dtype=np.float32)
        if scale.ndim != 1 or scale.size == 0:
            raise ValueError("Target scale must be a non-empty vector.")
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("Target scale must contain positive finite values.")
        super().__init__(name=name, dtype=dtype)
        self.target_scale = scale

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Any = None,
    ) -> Any:
        """
        Accumulate per-curve MAE after restoring each channel's scale.
        """
        scale = keras.ops.convert_to_tensor(
            self.target_scale,
            dtype=self.dtype,
        )
        absolute_error = keras.ops.abs(y_pred - y_true) * scale
        per_curve_mae = keras.ops.mean(absolute_error, axis=(1, 2))
        return super().update_state(per_curve_mae, sample_weight=sample_weight)

    def get_config(self) -> dict[str, Any]:
        """
        Return the serializable metric configuration.
        """
        return {
            **super().get_config(),
            "target_scale": self.target_scale.tolist(),
        }


@dataclass
class CurveNeuralModel(NeuralModel):
    """
    Common model wrapper for S-TCNN-style whole-curve prediction.
    """

    frequencies_ghz: np.ndarray
    latent_dim: int = 32
    decoder_channels: Sequence[int] = (32, 16, 8)
    kernel_size: int = 5
    frequency_encoding: CurveFrequencyEncoding = "fourier"
    fourier_order: int = 4
    weight_decay: float = 0.0
    derivative_loss_weight: float = 11.626038
    batch_size: int = 64
    epochs: int = 100
    prediction_batch_size: int = 4096
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 0.5
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    random_state: int = 128
    x_scaler: StandardScaler = field(init=False, repr=False)
    y_scaler: StandardScaler = field(init=False, repr=False)
    model: keras.Model | None = field(default=None, init=False, repr=False)
    history: keras.callbacks.History | None = field(
        default=None,
        init=False,
        repr=False,
    )
    selected_epoch_: int | None = field(default=None, init=False)

    name: ClassVar[str] = "curve_neural"

    def __post_init__(self) -> None:
        """
        Normalize and validate constructor controls and initialize scalers.
        """
        frequencies = np.asarray(self.frequencies_ghz, dtype=float)
        if frequencies.ndim != 1 or frequencies.size < 2:
            raise ValueError("Frequency grid must be one-dimensional.")
        if not np.isfinite(frequencies).all():
            raise ValueError("Frequency grid must contain only finite values.")
        if np.any(np.diff(frequencies) <= 0.0):
            raise ValueError("Frequency grid must be strictly increasing.")

        channels = tuple(int(width) for width in self.decoder_channels)
        if len(channels) != 3:
            raise ValueError("Exactly three decoder channel widths are required.")
        if self.frequency_encoding not in {"none", "linear", "fourier"}:
            raise ValueError("Frequency encoding must be none, linear, or fourier.")
        if min(
            self.latent_dim,
            self.kernel_size,
            self.fourier_order,
            *channels,
        ) <= 0:
            raise ValueError("Architecture dimensions must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("Weight decay must not be negative.")
        if self.derivative_loss_weight < 0.0:
            raise ValueError("Derivative loss weight must not be negative.")

        self.frequencies_ghz = frequencies.copy()
        self.decoder_channels = channels
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    @classmethod
    def from_config(
        cls,
        cfg: CurveNeuralModelConfig,
        *,
        frequencies_ghz: np.ndarray,
    ) -> CurveNeuralModel:
        """
        Return a curve model initialized from typed configuration.
        """
        return cls(  # type: ignore[arg-type]
            frequencies_ghz=frequencies_ghz,
            **asdict(cfg),
        )

    def model_name(self) -> str:
        """
        Return the human-readable curve-model label.
        """
        return "S-TCNN-Style Curve Neural"

    def fit(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,  # pylint: disable=invalid-name
        y_val: np.ndarray | None = None,
        verbose: str | int = 2,
    ) -> CurveNeuralModel:
        """
        Fit the curve decoder using wrapper-owned train-only scaling.
        """
        if X_val is None or y_val is None:
            raise ValueError("CurveNeuralModel requires validation data.")
        X_train_array, y_train_array = self._validated_arrays(  # noqa: N806
            X_train,
            y_train,
            split_name="training",
        )
        X_val_array, y_val_array = self._validated_arrays(  # noqa: N806
            X_val,
            y_val,
            split_name="validation",
        )
        if X_val_array.shape[1] != X_train_array.shape[1]:
            raise ValueError("Training and validation feature widths differ.")
        if y_val_array.shape[2] != y_train_array.shape[2]:
            raise ValueError("Training and validation target widths differ.")

        keras.utils.set_random_seed(self.random_state)
        X_train_scaled = self.x_scaler.fit_transform(X_train_array).astype(  # noqa: N806
            np.float32
        )
        X_val_scaled = self.x_scaler.transform(X_val_array).astype(  # noqa: N806
            np.float32
        )
        n_targets = y_train_array.shape[-1]
        y_train_scaled = self.y_scaler.fit_transform(
            y_train_array.reshape(-1, n_targets)
        ).reshape(y_train_array.shape).astype(np.float32)
        y_val_scaled = self.y_scaler.transform(
            y_val_array.reshape(-1, n_targets)
        ).reshape(y_val_array.shape).astype(np.float32)

        encoded_grid = self._encoded_frequency_grid()
        feature_width = 0 if encoded_grid is None else encoded_grid.shape[1]
        self.model = build_curve_decoder(
            input_width=X_train_scaled.shape[1],
            n_frequencies=len(self.frequencies_ghz),
            n_targets=n_targets,
            latent_dim=self.latent_dim,
            decoder_channels=self.decoder_channels,
            kernel_size=self.kernel_size,
            frequency_feature_width=feature_width,
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=self.gradient_clip_norm,
                weight_decay=self.weight_decay,
            ),
            loss=CurveAwareMSE(self.derivative_loss_weight),
            metrics=[UnscaledMeanAbsoluteError(self.y_scaler.scale_)],
            steps_per_execution=8,
        )
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_mae_db",
            patience=self.early_stopping_patience,
            min_delta=CALLBACK_MIN_DELTA,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        )
        self.history = self.model.fit(
            self._model_inputs(X_train_scaled, encoded_grid),
            y_train_scaled,
            validation_data=(
                self._model_inputs(X_val_scaled, encoded_grid),
                y_val_scaled,
            ),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_mae_db",
                    factor=self.reduce_lr_factor,
                    patience=self.reduce_lr_patience,
                    min_delta=CALLBACK_MIN_DELTA,
                    mode="min",
                    min_lr=self.min_learning_rate,
                    verbose=1,
                ),
                early_stopping,
            ],
            shuffle=True,
            verbose=verbose,  # type: ignore[assignment]
        )
        self.selected_epoch_ = int(early_stopping.best_epoch) + 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Return unscaled whole-curve predictions in dB.
        """
        X_array = np.asarray(X, dtype=float)  # noqa: N806
        if X_array.ndim != 2:
            raise ValueError("Prediction features must be a two-dimensional array.")
        X_scaled = self.x_scaler.transform(X_array).astype(np.float32)  # noqa: N806
        prediction_scaled = self.keras_model.predict(
            self._model_inputs(X_scaled, self._encoded_frequency_grid()),
            batch_size=self.prediction_batch_size,
            verbose=0,  # type: ignore[assignment]
        )
        n_targets = prediction_scaled.shape[-1]
        return self.y_scaler.inverse_transform(
            np.asarray(prediction_scaled).reshape(-1, n_targets)
        ).reshape(prediction_scaled.shape)

    def evaluate(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
        y: np.ndarray,
    ) -> dict[str, float]:
        """
        Return aggregate dB metrics after flattening design and frequency axes.
        """
        targets = np.asarray(y, dtype=float)
        predictions = self.predict(X)
        if targets.shape != predictions.shape:
            raise ValueError("Prediction and target curve shapes differ.")
        n_targets = targets.shape[-1]
        return regression_metrics(
            targets.reshape(-1, n_targets),
            predictions.reshape(-1, n_targets),
        )

    def plot_training_history(self) -> Figure:
        """
        Plot the training objective and dB MAE with the restored epoch marked.
        """
        if self.history is None or self.selected_epoch_ is None:
            raise RuntimeError(f"{self.name} has no recorded training history.")

        history = self.history.history
        required_keys = {"loss", "val_loss", "mae_db", "val_mae_db"}
        missing_keys = required_keys.difference(history)
        if missing_keys:
            names = ", ".join(sorted(missing_keys))
            raise ValueError(f"Curve training history is missing: {names}")

        epochs = np.arange(1, len(history["loss"]) + 1)
        selected_index = self.selected_epoch_ - 1
        if selected_index not in range(len(epochs)):
            raise ValueError("Selected epoch is outside the training history.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        panels = (
            (
                axes[0],
                "loss",
                "val_loss",
                "Optimization objective",
                "Composite loss (scaled target units)",
            ),
            (
                axes[1],
                "mae_db",
                "val_mae_db",
                "Selection metric",
                "MAE (dB)",
            ),
        )
        for axis, train_key, validation_key, title, ylabel in panels:
            axis.plot(epochs, history[train_key], label="training")
            axis.plot(epochs, history[validation_key], label="validation")
            axis.axvline(
                self.selected_epoch_,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=f"restored epoch {self.selected_epoch_}",
            )
            axis.scatter(
                [self.selected_epoch_],
                [history[validation_key][selected_index]],
                color="black",
                s=30,
                zorder=3,
            )
            axis.set(
                xlabel="Epoch",
                ylabel=ylabel,
                title=title,
            )
            axis.grid(True, alpha=0.3)
            axis.legend()

        fig.suptitle(f"{self.model_name()} Training History")
        fig.tight_layout()
        return fig

    @property
    def keras_model(self) -> keras.Model:
        """
        Return the fitted Keras decoder.
        """
        if self.model is None:
            raise RuntimeError("CurveNeuralModel must be fitted before prediction.")
        return self.model

    def _validated_arrays(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
        y: np.ndarray,
        *,
        split_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate and return one design/curve split.
        """
        features = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float)
        if features.ndim != 2:
            raise ValueError(f"{split_name.title()} features must be two-dimensional.")
        if targets.ndim != 3:
            raise ValueError(f"{split_name.title()} targets must be three-dimensional.")
        if len(features) != len(targets):
            raise ValueError(f"{split_name.title()} features and targets differ.")
        if targets.shape[1] != len(self.frequencies_ghz):
            raise ValueError(f"{split_name.title()} frequency count differs.")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError(f"{split_name.title()} arrays contain non-finite values.")
        return features, targets

    def _encoded_frequency_grid(self) -> np.ndarray | None:
        """
        Return the configured explicit frequency features.
        """
        if self.frequency_encoding == "none":
            return None
        return frequency_features(
            self.frequencies_ghz,
            self.frequency_encoding,
            fourier_order=self.fourier_order,
        )

    @staticmethod
    def _model_inputs(
        features: np.ndarray,
        encoded_grid: np.ndarray | None,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Pair scaled designs with a repeated explicit frequency grid.
        """
        if encoded_grid is None:
            return features
        repeated_grid = np.repeat(
            encoded_grid[np.newaxis, :, :],
            len(features),
            axis=0,
        )
        return [features, repeated_grid]
