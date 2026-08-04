"""
Frequency-conditioned neural modelling for complete complex S-matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import keras
import numpy as np
from keras import Input, layers
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.signal import hilbert
from sklearn.preprocessing import StandardScaler

from sparam_surrogate.models.curve_neural import CALLBACK_MIN_DELTA, frequency_features
from sparam_surrogate.models.neural import NeuralModel


def real_imag_channels_to_smatrix(
    channels: np.ndarray,
    n_ports: int,
) -> np.ndarray:
    """
    Convert row-major real-then-imaginary channels to complex S-matrices.
    """
    values = np.asarray(channels)
    n_entries = n_ports**2
    real = values[..., :n_entries]
    imag = values[..., n_entries:]
    return (real + 1j * imag).reshape(*values.shape[:-1], n_ports, n_ports)


def complex_smatrix_to_channels(matrices: np.ndarray) -> np.ndarray:
    """
    Flatten complex S-matrices as all real entries followed by all imaginary.
    """
    values = np.asarray(matrices)
    flattened = values.reshape(*values.shape[:-2], -1)
    return np.concatenate((flattened.real, flattened.imag), axis=-1).astype(np.float32)


def upper_triangle_entry_indices(n_ports: int) -> np.ndarray:
    """
    Return row-major flattened indices for the upper triangle.
    """
    rows, columns = np.triu_indices(n_ports)
    return (rows * n_ports + columns).astype(np.int32)


def _full_to_upper_mapping(n_ports: int) -> np.ndarray:
    """
    Map every row-major matrix entry to its unique upper-triangle entry.
    """
    upper = upper_triangle_entry_indices(n_ports)
    positions = {int(entry): index for index, entry in enumerate(upper)}
    return np.asarray(
        [
            positions[min(row, column) * n_ports + max(row, column)]
            for row in range(n_ports)
            for column in range(n_ports)
        ],
        dtype=np.int32,
    )


def _output_entry_positions(
    n_ports: int,
    reciprocal: bool,
    entry_indices: tuple[int, ...],
) -> np.ndarray:
    """
    Map full-matrix entry indices to positions in the model output.
    """
    indices = np.asarray(entry_indices, dtype=np.int32)
    if reciprocal:
        return _full_to_upper_mapping(n_ports)[indices]
    return indices


def _select_complex_entries(
    channels: np.ndarray,
    entry_indices: np.ndarray,
) -> np.ndarray:
    """
    Select complex entries while retaining real-then-imaginary ordering.
    """
    values = np.asarray(channels)
    n_entries = values.shape[-1] // 2
    return np.concatenate(
        (
            values[..., :n_entries][..., entry_indices],
            values[..., n_entries:][..., entry_indices],
        ),
        axis=-1,
    )


def _expand_reciprocal_channels(
    channels: np.ndarray,
    n_ports: int,
) -> np.ndarray:
    """
    Mirror unique upper-triangle channels into the full external contract.
    """
    values = np.asarray(channels)
    n_upper = values.shape[-1] // 2
    mapping = _full_to_upper_mapping(n_ports)
    return np.concatenate(
        (
            values[..., :n_upper][..., mapping],
            values[..., n_upper:][..., mapping],
        ),
        axis=-1,
    )


@dataclass
class ComplexRMSScaler:
    """
    Scale each complex matrix entry by one shared real/imaginary RMS magnitude.
    """

    scale_: np.ndarray | None = field(default=None, init=False)

    def fit(self, targets: np.ndarray) -> ComplexRMSScaler:
        """
        Fit entry-wise RMS magnitudes from training targets.
        """
        values = np.asarray(targets, dtype=np.float32)
        n_entries = values.shape[-1] // 2
        squared_magnitude = values[..., :n_entries] ** 2 + values[..., n_entries:] ** 2
        axes = tuple(range(squared_magnitude.ndim - 1))
        self.scale_ = np.maximum(
            np.sqrt(np.mean(squared_magnitude, axis=axes)),
            1e-12,
        ).astype(np.float32)
        return self

    def transform(
        self,
        targets: np.ndarray,
        entry_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Scale full targets or a selected set of complex entries.
        """
        values = np.asarray(targets, dtype=np.float32)
        if entry_indices is not None:
            values = _select_complex_entries(values, entry_indices)
        scale = self.entry_scale(entry_indices)
        return values / np.concatenate((scale, scale))

    def inverse_transform(
        self,
        targets: np.ndarray,
        entry_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Restore scaled full targets or selected complex entries.
        """
        scale = self.entry_scale(entry_indices)
        return np.asarray(targets) * np.concatenate((scale, scale))

    def entry_scale(
        self,
        entry_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return fitted scales for all or selected entries.
        """
        if self.scale_ is None:
            raise RuntimeError("ComplexRMSScaler must be fitted before use.")
        if entry_indices is None:
            return self.scale_
        return self.scale_[entry_indices]


@keras.saving.register_keras_serializable(package="sparam_surrogate")
class FixedFrequencyFeatures(layers.Layer):
    """
    Repeat fixed Fourier and optional Gaussian frequency features per design.
    """

    def __init__(
        self,
        frequencies_ghz: list[float] | np.ndarray,
        fourier_order: int = 4,
        rbf_count: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Store the frequency grid and its fixed encoding.
        """
        super().__init__(**kwargs)
        self.frequencies_ghz = np.asarray(frequencies_ghz, dtype=np.float32)
        self.fourier_order = int(fourier_order)
        self.rbf_count = int(rbf_count)
        self.features = frequency_features(
            self.frequencies_ghz,
            "fourier",
            fourier_order=self.fourier_order,
        )
        if self.rbf_count:
            normalized = frequency_features(
                self.frequencies_ghz,
                "linear",
            )
            centers = np.linspace(-1.0, 1.0, self.rbf_count, dtype=np.float32)
            spacing = 2.0 / max(self.rbf_count - 1, 1)
            width = 1.5 * spacing
            rbf_features = np.exp(
                -0.5 * ((normalized - centers[np.newaxis, :]) / width) ** 2
            ).astype(np.float32)
            self.features = np.concatenate((self.features, rbf_features), axis=1)

    def call(self, inputs: Any) -> Any:
        """
        Return the fixed grid repeated over the dynamic batch dimension.
        """
        features = keras.ops.convert_to_tensor(self.features)
        features = keras.ops.expand_dims(features, axis=0)
        output_shape = (
            keras.ops.shape(inputs)[0],
            self.features.shape[0],
            self.features.shape[1],
        )
        return keras.ops.broadcast_to(features, output_shape)

    def compute_output_shape(self, input_shape: Any) -> tuple[Any, int, int]:
        """
        Return the batched frequency-feature shape.
        """
        return input_shape[0], self.features.shape[0], self.features.shape[1]

    def get_config(self) -> dict[str, Any]:
        """
        Return the serializable layer configuration.
        """
        return {
            **super().get_config(),
            "frequencies_ghz": self.frequencies_ghz.tolist(),
            "fourier_order": self.fourier_order,
            "rbf_count": self.rbf_count,
        }


def build_frequency_residual_model(
    *,
    input_width: int,
    frequencies_ghz: np.ndarray,
    output_width: int,
    hidden_width: int = 128,
    residual_blocks: int = 3,
    fourier_order: int = 4,
    frequency_rbf_count: int = 0,
) -> keras.Model:
    """
    Build a frequency-conditioned residual MLP for complete design curves.
    """
    design_input = Input(shape=(input_width,), name="design_parameters")
    repeated_design = layers.RepeatVector(
        len(frequencies_ghz),
        name="repeat_design_over_frequency",
    )(design_input)
    encoded_frequency = FixedFrequencyFeatures(
        frequencies_ghz,
        fourier_order,
        rbf_count=frequency_rbf_count,
        name="fixed_frequency_features",
    )(design_input)
    trunk = layers.Concatenate(name="design_and_frequency")(
        [repeated_design, encoded_frequency]
    )
    trunk = layers.Dense(
        hidden_width,
        activation="relu",
        kernel_initializer="he_normal",
        name="input_projection",
    )(trunk)
    for block in range(1, residual_blocks + 1):
        residual = layers.Dense(
            hidden_width,
            activation="relu",
            kernel_initializer="he_normal",
            name=f"residual_{block}_hidden",
        )(trunk)
        residual = layers.Dense(
            hidden_width,
            kernel_initializer="he_normal",
            name=f"residual_{block}_output",
        )(residual)
        trunk = layers.Add(name=f"residual_{block}_add")([trunk, residual])
        trunk = layers.Activation(
            "relu",
            name=f"residual_{block}_activation",
        )(trunk)
    outputs = layers.Dense(
        output_width,
        name="scaled_real_imag_outputs",
    )(trunk)
    return keras.Model(
        design_input,
        outputs,
        name="frequency_conditioned_full_smatrix",
    )


def build_insertion_loss_residual_head(
    *,
    input_width: int,
    frequencies_ghz: np.ndarray,
    n_paths: int,
    hidden_width: int = 64,
    fourier_order: int = 4,
) -> keras.Model:
    """
    Build a small head that corrects configured insertion-loss predictions.
    """
    design_input = Input(shape=(input_width,), name="scaled_design_parameters")
    baseline_input = Input(
        shape=(len(frequencies_ghz), n_paths),
        name="scaled_baseline_insertion_loss",
    )
    repeated_design = layers.RepeatVector(
        len(frequencies_ghz),
        name="repeat_design_over_frequency",
    )(design_input)
    encoded_frequency = FixedFrequencyFeatures(
        frequencies_ghz,
        fourier_order,
        name="fourier_frequency_features",
    )(design_input)
    head = layers.Concatenate(name="residual_head_features")(
        [repeated_design, encoded_frequency, baseline_input]
    )
    head = layers.Dense(
        hidden_width,
        activation="relu",
        kernel_initializer="he_normal",
        name="residual_head_hidden_1",
    )(head)
    head = layers.Dense(
        hidden_width,
        activation="relu",
        kernel_initializer="he_normal",
        name="residual_head_hidden_2",
    )(head)
    corrections = layers.Dense(
        n_paths,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="insertion_loss_correction_db",
    )(head)
    return keras.Model(
        [design_input, baseline_input],
        corrections,
        name="six_path_insertion_loss_residual_head",
    )


def build_complex_residual_head(
    *,
    input_width: int,
    frequencies_ghz: np.ndarray,
    n_paths: int,
    hidden_width: int = 64,
    fourier_order: int = 4,
    frequency_rbf_count: int = 0,
) -> keras.Model:
    """
    Build a head that adds trainable complex residuals to baseline paths.
    """
    design_input = Input(shape=(input_width,), name="scaled_design_parameters")
    baseline_input = Input(
        shape=(len(frequencies_ghz), 2 * n_paths),
        name="scaled_baseline_complex_paths",
    )
    repeated_design = layers.RepeatVector(
        len(frequencies_ghz),
        name="repeat_design_over_frequency",
    )(design_input)
    encoded_frequency = FixedFrequencyFeatures(
        frequencies_ghz,
        fourier_order,
        rbf_count=frequency_rbf_count,
        name="fixed_frequency_features",
    )(design_input)
    head = layers.Concatenate(name="complex_residual_head_features")(
        [repeated_design, encoded_frequency, baseline_input]
    )
    head = layers.Dense(
        hidden_width,
        activation="relu",
        kernel_initializer="he_normal",
        name="complex_residual_hidden_1",
    )(head)
    head = layers.Dense(
        hidden_width,
        activation="relu",
        kernel_initializer="he_normal",
        name="complex_residual_hidden_2",
    )(head)
    residual = layers.Dense(
        2 * n_paths,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="scaled_complex_residual",
    )(head)
    corrected = layers.Add(name="corrected_scaled_complex_paths")(
        [baseline_input, residual]
    )
    return keras.Model(
        [design_input, baseline_input],
        corrected,
        name="six_path_complex_residual_head",
    )


@keras.saving.register_keras_serializable(package="sparam_surrogate")
class ComplexPhysicsLoss(keras.losses.Loss):
    """
    Combine normalized complex MSE with log-magnitude and passivity terms.
    """

    def __init__(
        self,
        entry_scale: list[float] | np.ndarray,
        n_ports: int,
        reciprocal: bool = False,
        log_magnitude_weight: float = 0.0,
        log_magnitude_entry_indices: tuple[int, ...] | None = None,
        log_magnitude_floor: float = 1e-6,
        targeted_log_magnitude_weight: float = 0.0,
        targeted_log_magnitude_entry_indices: tuple[int, ...] | None = None,
        deep_null_log_magnitude_weight: float = 0.0,
        deep_null_threshold_magnitude: float = 0.0,
        deep_null_weight: float = 0.0,
        passivity_weight: float = 0.0,
        reduction: Any = "sum_over_batch_size",
        name: str = "complex_physics_loss",
    ) -> None:
        """
        Store output scaling and optional physics weights.
        """
        super().__init__(reduction=reduction, name=name)
        self.entry_scale = np.asarray(entry_scale, dtype=np.float32)
        self.n_ports = int(n_ports)
        self.reciprocal = bool(reciprocal)
        self.log_magnitude_weight = float(log_magnitude_weight)
        self.log_magnitude_entry_indices = tuple(
            int(index) for index in (log_magnitude_entry_indices or ())
        )
        self.log_magnitude_floor = float(log_magnitude_floor)
        self.targeted_log_magnitude_weight = float(targeted_log_magnitude_weight)
        self.targeted_log_magnitude_entry_indices = tuple(
            int(index) for index in (targeted_log_magnitude_entry_indices or ())
        )
        self.deep_null_log_magnitude_weight = float(deep_null_log_magnitude_weight)
        self.deep_null_threshold_magnitude = float(deep_null_threshold_magnitude)
        self.deep_null_weight = float(deep_null_weight)
        self.passivity_weight = float(passivity_weight)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """
        Return one composite loss value per design.
        """
        error = y_pred - y_true
        total = keras.ops.mean(keras.ops.square(error), axis=(1, 2))
        if self.log_magnitude_weight:
            total = total + self.log_magnitude_weight * self._log_magnitude_loss(
                y_true,
                y_pred,
            )
        if self.targeted_log_magnitude_weight:
            total = total + (
                self.targeted_log_magnitude_weight
                * self._log_magnitude_loss(
                    y_true,
                    y_pred,
                    entry_indices=self.targeted_log_magnitude_entry_indices,
                    apply_deep_weight=False,
                )
            )
        if self.deep_null_log_magnitude_weight:
            total = total + (
                self.deep_null_log_magnitude_weight
                * self._log_magnitude_loss(
                    y_true,
                    y_pred,
                    entry_indices=self.targeted_log_magnitude_entry_indices,
                    deep_only=True,
                    apply_deep_weight=False,
                )
            )
        if self.passivity_weight:
            total = total + self.passivity_weight * self._passivity_loss(y_pred)
        return total

    def _unscaled_components(self, values: Any) -> tuple[Any, Any]:
        """
        Return unscaled real and imaginary output entries.
        """
        n_entries = len(self.entry_scale)
        scale = keras.ops.convert_to_tensor(self.entry_scale)
        return (
            values[..., :n_entries] * scale,
            values[..., n_entries:] * scale,
        )

    def _log_magnitude_loss(
        self,
        y_true: Any,
        y_pred: Any,
        *,
        entry_indices: tuple[int, ...] | None = None,
        deep_only: bool = False,
        apply_deep_weight: bool = True,
    ) -> Any:
        """
        Return targeted Huber loss on stable, optionally tail-weighted logs.
        """
        true_real, true_imag = self._unscaled_components(y_true)
        pred_real, pred_imag = self._unscaled_components(y_pred)
        positions = self._log_magnitude_positions(entry_indices)
        true_real = keras.ops.take(true_real, positions, axis=-1)
        true_imag = keras.ops.take(true_imag, positions, axis=-1)
        pred_real = keras.ops.take(pred_real, positions, axis=-1)
        pred_imag = keras.ops.take(pred_imag, positions, axis=-1)
        floor_squared = self.log_magnitude_floor**2
        true_magnitude_squared = true_real**2 + true_imag**2
        pred_magnitude_squared = pred_real**2 + pred_imag**2
        true_log = 0.5 * keras.ops.log(true_magnitude_squared + floor_squared)
        pred_log = 0.5 * keras.ops.log(pred_magnitude_squared + floor_squared)
        absolute_error = keras.ops.abs(pred_log - true_log)
        huber = keras.ops.where(
            absolute_error <= 1.0,
            0.5 * keras.ops.square(absolute_error),
            absolute_error - 0.5,
        )
        threshold_squared = self.deep_null_threshold_magnitude**2
        deep_null_mask = keras.ops.cast(
            true_magnitude_squared <= threshold_squared,
            huber.dtype,
        )
        if deep_only:
            numerator = keras.ops.sum(huber * deep_null_mask, axis=(1, 2))
            denominator = keras.ops.maximum(
                keras.ops.sum(deep_null_mask, axis=(1, 2)),
                1.0,
            )
            return numerator / denominator
        weights = keras.ops.ones_like(huber)
        if apply_deep_weight and self.deep_null_weight:
            weights = weights + self.deep_null_weight * deep_null_mask
        return keras.ops.sum(huber * weights, axis=(1, 2)) / keras.ops.sum(
            weights,
            axis=(1, 2),
        )

    def _log_magnitude_positions(
        self,
        entry_indices: tuple[int, ...] | None = None,
    ) -> Any:
        """
        Return targeted or default off-diagonal output positions.
        """
        selected_entries = (
            self.log_magnitude_entry_indices if entry_indices is None else entry_indices
        )
        if selected_entries:
            positions = _output_entry_positions(
                self.n_ports,
                self.reciprocal,
                selected_entries,
            )
        elif self.reciprocal:
            rows, columns = np.triu_indices(self.n_ports)
            positions = np.flatnonzero(rows != columns)
        else:
            rows, columns = np.indices((self.n_ports, self.n_ports))
            positions = np.flatnonzero(rows.ravel() != columns.ravel())
        return keras.ops.convert_to_tensor(positions.astype(np.int32))

    def _passivity_loss(self, y_pred: Any) -> Any:
        """
        Return the mean squared largest-singular-value excess per design.
        """
        real, imag = self._unscaled_components(y_pred)
        if self.reciprocal:
            mapping = keras.ops.convert_to_tensor(_full_to_upper_mapping(self.n_ports))
            real = keras.ops.take(real, mapping, axis=-1)
            imag = keras.ops.take(imag, mapping, axis=-1)
        batch_size = keras.ops.shape(real)[0]
        n_frequencies = keras.ops.shape(real)[1]
        matrix_shape = (
            batch_size,
            n_frequencies,
            self.n_ports,
            self.n_ports,
        )
        real = keras.ops.reshape(real, matrix_shape)
        imag = keras.ops.reshape(imag, matrix_shape)
        block_matrix = keras.ops.concatenate(
            (
                keras.ops.concatenate((real, -imag), axis=-1),
                keras.ops.concatenate((imag, real), axis=-1),
            ),
            axis=-2,
        )
        singular_values = keras.ops.linalg.svd(
            block_matrix,
            compute_uv=False,
        )
        excess = keras.ops.relu(keras.ops.max(singular_values, axis=-1) - 1.0)
        return keras.ops.mean(keras.ops.square(excess), axis=1)

    def get_config(self) -> dict[str, Any]:
        """
        Return the serializable loss configuration.
        """
        return {
            **super().get_config(),
            "entry_scale": self.entry_scale.tolist(),
            "n_ports": self.n_ports,
            "reciprocal": self.reciprocal,
            "log_magnitude_weight": self.log_magnitude_weight,
            "log_magnitude_entry_indices": self.log_magnitude_entry_indices,
            "log_magnitude_floor": self.log_magnitude_floor,
            "targeted_log_magnitude_weight": self.targeted_log_magnitude_weight,
            "targeted_log_magnitude_entry_indices": (
                self.targeted_log_magnitude_entry_indices
            ),
            "deep_null_log_magnitude_weight": (self.deep_null_log_magnitude_weight),
            "deep_null_threshold_magnitude": (self.deep_null_threshold_magnitude),
            "deep_null_weight": self.deep_null_weight,
            "passivity_weight": self.passivity_weight,
        }


@keras.saving.register_keras_serializable(package="sparam_surrogate")
class ResidualInsertionLoss(keras.losses.Loss):
    """
    Combine ordinary and separately normalized deep-null residual Huber loss.
    """

    def __init__(
        self,
        deep_null_threshold_db: float,
        deep_null_weight: float = 0.2,
        huber_delta_db: float = 5.0,
        reduction: Any = "sum_over_batch_size",
        name: str = "residual_insertion_loss",
    ) -> None:
        """
        Store the train-defined null threshold and loss weights.
        """
        super().__init__(reduction=reduction, name=name)
        self.deep_null_threshold_db = float(deep_null_threshold_db)
        self.deep_null_weight = float(deep_null_weight)
        self.huber_delta_db = float(huber_delta_db)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """
        Return one residual-head loss value per design.
        """
        n_paths = y_pred.shape[-1]
        residual_target = y_true[..., :n_paths]
        true_insertion_loss = y_true[..., n_paths:]
        absolute_error = keras.ops.abs(y_pred - residual_target)
        delta = self.huber_delta_db
        huber = keras.ops.where(
            absolute_error <= delta,
            0.5 * keras.ops.square(absolute_error),
            delta * absolute_error - 0.5 * delta**2,
        )
        ordinary = keras.ops.mean(huber, axis=(1, 2))
        deep_mask = keras.ops.cast(
            true_insertion_loss >= self.deep_null_threshold_db,
            huber.dtype,
        )
        deep = keras.ops.sum(huber * deep_mask, axis=(1, 2)) / keras.ops.maximum(
            keras.ops.sum(deep_mask, axis=(1, 2)),
            1.0,
        )
        return ordinary + self.deep_null_weight * deep

    def get_config(self) -> dict[str, Any]:
        """
        Return the serializable residual-loss configuration.
        """
        return {
            **super().get_config(),
            "deep_null_threshold_db": self.deep_null_threshold_db,
            "deep_null_weight": self.deep_null_weight,
            "huber_delta_db": self.huber_delta_db,
        }


@keras.saving.register_keras_serializable(package="sparam_surrogate")
class ComplexResidualHeadLoss(keras.losses.Loss):
    """
    Combine scaled complex error with targeted magnitude and deep-null terms.
    """

    def __init__(
        self,
        entry_scale: list[float] | np.ndarray,
        deep_null_threshold_magnitude: float,
        log_magnitude_weight: float = 0.1,
        deep_null_weight: float = 0.02,
        log_magnitude_floor: float = 1e-14,
        reduction: Any = "sum_over_batch_size",
        name: str = "complex_residual_head_loss",
    ) -> None:
        """
        Store path scales and auxiliary magnitude-loss settings.
        """
        super().__init__(reduction=reduction, name=name)
        self.entry_scale = np.asarray(entry_scale, dtype=np.float32)
        self.deep_null_threshold_magnitude = float(deep_null_threshold_magnitude)
        self.log_magnitude_weight = float(log_magnitude_weight)
        self.deep_null_weight = float(deep_null_weight)
        self.log_magnitude_floor = float(log_magnitude_floor)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """
        Return one composite complex-path loss value per design.
        """
        complex_error = keras.ops.mean(
            keras.ops.square(y_pred - y_true),
            axis=(1, 2),
        )
        n_paths = len(self.entry_scale)
        scale = keras.ops.convert_to_tensor(self.entry_scale)
        true_real = y_true[..., :n_paths] * scale
        true_imag = y_true[..., n_paths:] * scale
        pred_real = y_pred[..., :n_paths] * scale
        pred_imag = y_pred[..., n_paths:] * scale
        floor_squared = self.log_magnitude_floor**2
        true_magnitude_squared = true_real**2 + true_imag**2
        pred_magnitude_squared = pred_real**2 + pred_imag**2
        true_log = 0.5 * keras.ops.log(true_magnitude_squared + floor_squared)
        pred_log = 0.5 * keras.ops.log(pred_magnitude_squared + floor_squared)
        absolute_error = keras.ops.abs(pred_log - true_log)
        huber = keras.ops.where(
            absolute_error <= 1.0,
            0.5 * keras.ops.square(absolute_error),
            absolute_error - 0.5,
        )
        magnitude_loss = keras.ops.mean(huber, axis=(1, 2))
        deep_mask = keras.ops.cast(
            true_magnitude_squared <= self.deep_null_threshold_magnitude**2,
            huber.dtype,
        )
        deep_loss = keras.ops.sum(huber * deep_mask, axis=(1, 2)) / keras.ops.maximum(
            keras.ops.sum(deep_mask, axis=(1, 2)),
            1.0,
        )
        return (
            complex_error
            + self.log_magnitude_weight * magnitude_loss
            + self.deep_null_weight * deep_loss
        )

    def get_config(self) -> dict[str, Any]:
        """
        Return the serializable complex residual loss configuration.
        """
        return {
            **super().get_config(),
            "entry_scale": self.entry_scale.tolist(),
            "deep_null_threshold_magnitude": self.deep_null_threshold_magnitude,
            "log_magnitude_weight": self.log_magnitude_weight,
            "deep_null_weight": self.deep_null_weight,
            "log_magnitude_floor": self.log_magnitude_floor,
        }


@keras.saving.register_keras_serializable(package="sparam_surrogate")
class ComplexNRMSE(keras.metrics.Metric):
    """
    Accumulate unscaled complex normalized root-mean-squared error.
    """

    def __init__(
        self,
        entry_scale: list[float] | np.ndarray,
        name: str = "complex_nrmse",
        **kwargs: Any,
    ) -> None:
        """
        Store output scales and initialize error-energy totals.
        """
        super().__init__(name=name, **kwargs)
        self.entry_scale = np.asarray(entry_scale, dtype=np.float32)
        self.squared_error = self.add_weight(name="squared_error", initializer="zeros")
        self.target_energy = self.add_weight(name="target_energy", initializer="zeros")

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Any = None,
    ) -> None:
        """
        Add unscaled complex error and target energy for one batch.
        """
        _ = sample_weight
        scale = keras.ops.convert_to_tensor(
            np.concatenate((self.entry_scale, self.entry_scale))
        )
        error = (y_pred - y_true) * scale
        target = y_true * scale
        self.squared_error.assign_add(keras.ops.sum(keras.ops.square(error)))
        self.target_energy.assign_add(keras.ops.sum(keras.ops.square(target)))

    def result(self) -> Any:
        """
        Return the aggregate complex NRMSE.
        """
        return keras.ops.sqrt(self.squared_error / self.target_energy)

    def get_config(self) -> dict[str, Any]:
        """
        Return the serializable metric configuration.
        """
        return {
            **super().get_config(),
            "entry_scale": self.entry_scale.tolist(),
        }


@keras.saving.register_keras_serializable(package="sparam_surrogate")
class SixPathInsertionLossMAE(keras.metrics.Metric):
    """
    Accumulate insertion-loss MAE for configured complex matrix entries.
    """

    def __init__(
        self,
        entry_scale: list[float] | np.ndarray,
        n_ports: int,
        entry_indices: tuple[int, ...],
        reciprocal: bool = False,
        name: str = "six_path_mae_db",
        **kwargs: Any,
    ) -> None:
        """
        Store scaling and configured full-matrix entry indices.
        """
        super().__init__(name=name, **kwargs)
        self.entry_scale = np.asarray(entry_scale, dtype=np.float32)
        self.n_ports = int(n_ports)
        self.entry_indices = tuple(int(index) for index in entry_indices)
        self.reciprocal = bool(reciprocal)
        self.absolute_error = self.add_weight(
            name="absolute_error",
            initializer="zeros",
        )
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Any = None,
    ) -> None:
        """
        Add uncapped dB errors for configured paths in one batch.
        """
        _ = sample_weight
        n_entries = len(self.entry_scale)
        scale = keras.ops.convert_to_tensor(self.entry_scale)
        positions = keras.ops.convert_to_tensor(
            _output_entry_positions(
                self.n_ports,
                self.reciprocal,
                self.entry_indices,
            )
        )
        true_real = keras.ops.take(
            y_true[..., :n_entries] * scale,
            positions,
            axis=-1,
        )
        true_imag = keras.ops.take(
            y_true[..., n_entries:] * scale,
            positions,
            axis=-1,
        )
        pred_real = keras.ops.take(
            y_pred[..., :n_entries] * scale,
            positions,
            axis=-1,
        )
        pred_imag = keras.ops.take(
            y_pred[..., n_entries:] * scale,
            positions,
            axis=-1,
        )
        coefficient = -10.0 / np.log(10.0)
        true_il = coefficient * keras.ops.log(
            keras.ops.maximum(true_real**2 + true_imag**2, 1e-30)
        )
        pred_il = coefficient * keras.ops.log(
            keras.ops.maximum(pred_real**2 + pred_imag**2, 1e-30)
        )
        error = keras.ops.abs(pred_il - true_il)
        self.absolute_error.assign_add(keras.ops.sum(error))
        self.count.assign_add(keras.ops.cast(keras.ops.size(error), self.dtype))

    def result(self) -> Any:
        """
        Return aggregate configured-path insertion-loss MAE in dB.
        """
        return self.absolute_error / self.count

    def get_config(self) -> dict[str, Any]:
        """
        Return the serializable metric configuration.
        """
        return {
            **super().get_config(),
            "entry_scale": self.entry_scale.tolist(),
            "n_ports": self.n_ports,
            "entry_indices": self.entry_indices,
            "reciprocal": self.reciprocal,
        }


def complex_regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    n_ports: int,
) -> dict[str, float]:
    """
    Return unscaled complex MAE and NRMSE for full external-channel arrays.
    """
    truth = real_imag_channels_to_smatrix(targets, n_ports)
    estimate = real_imag_channels_to_smatrix(predictions, n_ports)
    error = estimate - truth
    return {
        "ComplexMAE": float(np.mean(np.abs(error))),
        "ComplexNRMSE": float(
            np.sqrt(np.sum(np.abs(error) ** 2) / np.sum(np.abs(truth) ** 2))
        ),
    }


def configured_insertion_loss_db(
    channels: np.ndarray,
    n_ports: int,
    entry_indices: tuple[int, ...],
) -> np.ndarray:
    """
    Return uncapped insertion loss for configured complex matrix entries.
    """
    n_entries = n_ports**2
    selected = np.asarray(entry_indices, dtype=np.int32)
    values = np.asarray(channels)
    real = values[..., selected]
    imag = values[..., n_entries + selected]
    with np.errstate(divide="ignore"):
        return -10.0 * np.log10(real**2 + imag**2)


def apply_insertion_loss_correction(
    channels: np.ndarray,
    corrections_db: np.ndarray,
    n_ports: int,
    entry_indices: tuple[int, ...],
    *,
    reciprocal: bool,
) -> np.ndarray:
    """
    Apply magnitude-only dB corrections while retaining complex phase.
    """
    values = np.asarray(channels, dtype=np.float32)
    corrected = values.copy()
    n_entries = n_ports**2
    factors = np.power(10.0, -np.asarray(corrections_db) / 20.0)
    for path_index, entry_index in enumerate(entry_indices):
        factor = factors[..., path_index]
        corrected[..., entry_index] = values[..., entry_index] * factor
        corrected[..., n_entries + entry_index] = (
            values[..., n_entries + entry_index] * factor
        )
        if reciprocal:
            row, column = divmod(entry_index, n_ports)
            mirrored_index = column * n_ports + row
            corrected[..., mirrored_index] = values[..., mirrored_index] * factor
            corrected[..., n_entries + mirrored_index] = (
                values[..., n_entries + mirrored_index] * factor
            )
    return corrected


def apply_complex_path_correction(
    channels: np.ndarray,
    corrected_path_channels: np.ndarray,
    n_ports: int,
    entry_indices: tuple[int, ...],
    *,
    reciprocal: bool,
) -> np.ndarray:
    """
    Replace configured complex paths and mirror them when reciprocal.
    """
    values = np.asarray(channels, dtype=np.float32)
    corrected = values.copy()
    n_entries = n_ports**2
    n_paths = len(entry_indices)
    path_values = np.asarray(corrected_path_channels, dtype=np.float32)
    for path_index, entry_index in enumerate(entry_indices):
        real = path_values[..., path_index]
        imag = path_values[..., n_paths + path_index]
        corrected[..., entry_index] = real
        corrected[..., n_entries + entry_index] = imag
        if reciprocal:
            row, column = divmod(entry_index, n_ports)
            mirrored_index = column * n_ports + row
            corrected[..., mirrored_index] = real
            corrected[..., n_entries + mirrored_index] = imag
    return corrected


def configured_insertion_loss_mae(
    targets: np.ndarray,
    predictions: np.ndarray,
    n_ports: int,
    entry_indices: tuple[int, ...],
) -> float:
    """
    Return uncapped insertion-loss MAE for configured full-matrix entries.
    """
    true_insertion_loss = configured_insertion_loss_db(
        targets,
        n_ports,
        entry_indices,
    )
    predicted_insertion_loss = configured_insertion_loss_db(
        predictions,
        n_ports,
        entry_indices,
    )

    return float(np.mean(np.abs(predicted_insertion_loss - true_insertion_loss)))


class _GuardedValidationSelector(keras.callbacks.Callback):
    """
    Retain the lowest six-path-MAE epoch that passes a full-matrix NRMSE guard.
    """

    def __init__(
        self,
        validation_features: np.ndarray,
        validation_targets: np.ndarray,
        y_scaler: ComplexRMSScaler,
        n_ports: int,
        reciprocal: bool,
        entry_indices: tuple[int, ...],
        complex_nrmse_guard: float,
        prediction_batch_size: int,
    ) -> None:
        """
        Store raw validation targets and fitted output scaling.
        """
        super().__init__()
        self.validation_features = validation_features
        self.validation_targets = validation_targets
        self.y_scaler = y_scaler
        self.n_ports = n_ports
        self.reciprocal = reciprocal
        self.entry_indices = entry_indices
        self.complex_nrmse_guard = complex_nrmse_guard
        self.prediction_batch_size = prediction_batch_size
        self.initial_weights: list[np.ndarray] | None = None
        self.best_weights: list[np.ndarray] | None = None
        self.best_epoch: int | None = None
        self.best_six_path_mae = np.inf
        self.best_complex_nrmse = np.inf
        self.complex_nrmse_history: list[float] = []
        self.six_path_mae_history: list[float] = []

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """
        Snapshot the baseline weights before fine-tuning changes them.
        """
        _ = logs
        self.initial_weights = self.model.get_weights()

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        """
        Measure direct validation metrics and retain a qualifying epoch.
        """
        if logs is None:
            logs = {}
        prediction_scaled = self.model.predict(
            self.validation_features,
            batch_size=self.prediction_batch_size,
            verbose=0,
        )
        unique_entries = (
            upper_triangle_entry_indices(self.n_ports) if self.reciprocal else None
        )
        prediction = self.y_scaler.inverse_transform(
            prediction_scaled,
            unique_entries,
        )
        if self.reciprocal:
            prediction = _expand_reciprocal_channels(prediction, self.n_ports)
        complex_nrmse = complex_regression_metrics(
            self.validation_targets,
            prediction,
            self.n_ports,
        )["ComplexNRMSE"]
        six_path_mae = configured_insertion_loss_mae(
            self.validation_targets,
            prediction,
            self.n_ports,
            self.entry_indices,
        )
        self.complex_nrmse_history.append(complex_nrmse)
        self.six_path_mae_history.append(six_path_mae)
        logs["val_full_complex_nrmse"] = complex_nrmse
        logs["val_six_path_mae_db"] = six_path_mae
        if (
            complex_nrmse <= self.complex_nrmse_guard
            and six_path_mae < self.best_six_path_mae
        ):
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
            self.best_six_path_mae = six_path_mae
            self.best_complex_nrmse = complex_nrmse

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """
        Restore the best qualifying epoch or the untouched baseline weights.
        """
        _ = logs
        weights = self.best_weights or self.initial_weights
        if weights is not None:
            self.model.set_weights(weights)


class _ResidualHeadValidationSelector(keras.callbacks.Callback):
    """
    Select residual-head weights using guarded full-matrix validation metrics.
    """

    def __init__(
        self,
        validation_inputs: list[np.ndarray],
        validation_targets: np.ndarray,
        baseline_predictions: np.ndarray,
        n_ports: int,
        entry_indices: tuple[int, ...],
        reciprocal: bool,
        complex_nrmse_guard: float,
        prediction_batch_size: int,
    ) -> None:
        """
        Store validation inputs and the unmodified complex baseline.
        """
        super().__init__()
        self.validation_inputs = validation_inputs
        self.validation_targets = validation_targets
        self.baseline_predictions = baseline_predictions
        self.n_ports = n_ports
        self.entry_indices = entry_indices
        self.reciprocal = reciprocal
        self.complex_nrmse_guard = complex_nrmse_guard
        self.prediction_batch_size = prediction_batch_size
        self.initial_weights: list[np.ndarray] | None = None
        self.best_weights: list[np.ndarray] | None = None
        self.best_epoch: int | None = None
        self.best_six_path_mae = np.inf
        self.best_complex_nrmse = np.inf
        self.complex_nrmse_history: list[float] = []
        self.six_path_mae_history: list[float] = []

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """
        Snapshot the zero-correction head before optimization.
        """
        _ = logs
        self.initial_weights = self.model.get_weights()

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        """
        Score the corrected full matrix and retain a qualifying epoch.
        """
        if logs is None:
            logs = {}
        corrections = self.model.predict(
            self.validation_inputs,
            batch_size=self.prediction_batch_size,
            verbose=0,
        )
        prediction = apply_insertion_loss_correction(
            self.baseline_predictions,
            corrections,
            self.n_ports,
            self.entry_indices,
            reciprocal=self.reciprocal,
        )
        complex_nrmse = complex_regression_metrics(
            self.validation_targets,
            prediction,
            self.n_ports,
        )["ComplexNRMSE"]
        six_path_mae = configured_insertion_loss_mae(
            self.validation_targets,
            prediction,
            self.n_ports,
            self.entry_indices,
        )
        self.complex_nrmse_history.append(complex_nrmse)
        self.six_path_mae_history.append(six_path_mae)
        logs["val_full_complex_nrmse"] = complex_nrmse
        logs["val_six_path_mae_db"] = six_path_mae
        if (
            complex_nrmse <= self.complex_nrmse_guard
            and six_path_mae < self.best_six_path_mae
        ):
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
            self.best_six_path_mae = six_path_mae
            self.best_complex_nrmse = complex_nrmse

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """
        Restore the best qualifying head or the exact zero correction.
        """
        _ = logs
        weights = self.best_weights or self.initial_weights
        if weights is not None:
            self.model.set_weights(weights)


class _ComplexResidualValidationSelector(keras.callbacks.Callback):
    """
    Select complex residual-head weights under the full-matrix NRMSE guard.
    """

    def __init__(
        self,
        validation_inputs: list[np.ndarray],
        validation_targets: np.ndarray,
        baseline_predictions: np.ndarray,
        path_scaler: ComplexRMSScaler,
        n_ports: int,
        entry_indices: tuple[int, ...],
        reciprocal: bool,
        complex_nrmse_guard: float,
        prediction_batch_size: int,
    ) -> None:
        """
        Store validation data and scaling for complex reconstruction.
        """
        super().__init__()
        self.validation_inputs = validation_inputs
        self.validation_targets = validation_targets
        self.baseline_predictions = baseline_predictions
        self.path_scaler = path_scaler
        self.n_ports = n_ports
        self.entry_indices = entry_indices
        self.reciprocal = reciprocal
        self.complex_nrmse_guard = complex_nrmse_guard
        self.prediction_batch_size = prediction_batch_size
        self.initial_weights: list[np.ndarray] | None = None
        self.best_weights: list[np.ndarray] | None = None
        self.best_epoch: int | None = None
        self.best_six_path_mae = np.inf
        self.best_complex_nrmse = np.inf
        self.complex_nrmse_history: list[float] = []
        self.six_path_mae_history: list[float] = []

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """
        Snapshot the zero complex residual before optimization.
        """
        _ = logs
        self.initial_weights = self.model.get_weights()

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        """
        Reconstruct and score the corrected complete complex matrix.
        """
        if logs is None:
            logs = {}
        corrected_scaled = self.model.predict(
            self.validation_inputs,
            batch_size=self.prediction_batch_size,
            verbose=0,
        )
        corrected_paths = self.path_scaler.inverse_transform(corrected_scaled)
        prediction = apply_complex_path_correction(
            self.baseline_predictions,
            corrected_paths,
            self.n_ports,
            self.entry_indices,
            reciprocal=self.reciprocal,
        )
        complex_nrmse = complex_regression_metrics(
            self.validation_targets,
            prediction,
            self.n_ports,
        )["ComplexNRMSE"]
        six_path_mae = configured_insertion_loss_mae(
            self.validation_targets,
            prediction,
            self.n_ports,
            self.entry_indices,
        )
        self.complex_nrmse_history.append(complex_nrmse)
        self.six_path_mae_history.append(six_path_mae)
        logs["val_full_complex_nrmse"] = complex_nrmse
        logs["val_six_path_mae_db"] = six_path_mae
        if (
            complex_nrmse <= self.complex_nrmse_guard
            and six_path_mae < self.best_six_path_mae
        ):
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
            self.best_six_path_mae = six_path_mae
            self.best_complex_nrmse = complex_nrmse

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """
        Restore the best qualifying head or the exact baseline output.
        """
        _ = logs
        weights = self.best_weights or self.initial_weights
        if weights is not None:
            self.model.set_weights(weights)


@dataclass
class SixPathResidualHead:
    """
    Correct configured insertion-loss magnitudes around a frozen complex model.
    """

    frequencies_ghz: np.ndarray
    n_ports: int
    entry_indices: tuple[int, ...]
    hidden_width: int = 64
    fourier_order: int = 4
    deep_null_threshold_db: float = 60.0
    deep_null_weight: float = 0.2
    huber_delta_db: float = 5.0
    batch_size: int = 64
    epochs: int = 20
    prediction_batch_size: int = 64
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 0.5
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    reciprocal: bool = True
    random_state: int = 128
    x_scaler: StandardScaler = field(init=False, repr=False)
    baseline_il_scaler: StandardScaler = field(init=False, repr=False)
    model: keras.Model | None = field(default=None, init=False, repr=False)
    history: keras.callbacks.History | None = field(
        default=None,
        init=False,
        repr=False,
    )
    selected_epoch_: int | None = field(default=None, init=False)
    guard_passed_: bool | None = field(default=None, init=False)
    guarded_validation_nrmse_: float | None = field(default=None, init=False)
    guarded_validation_six_path_mae_db_: float | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        """
        Normalize array-like configuration and initialize train-only scalers.
        """
        self.frequencies_ghz = np.asarray(self.frequencies_ghz, dtype=np.float32)
        self.entry_indices = tuple(int(index) for index in self.entry_indices)
        self.x_scaler = StandardScaler()
        self.baseline_il_scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        baseline_train: np.ndarray,
        X_val: np.ndarray,  # pylint: disable=invalid-name
        y_val: np.ndarray,
        baseline_val: np.ndarray,
        *,
        complex_nrmse_guard: float,
        verbose: str | int = 2,
    ) -> SixPathResidualHead:
        """
        Fit dB residuals and restore the best guard-qualified epoch.
        """
        keras.utils.set_random_seed(self.random_state)
        X_train_scaled = self.x_scaler.fit_transform(  # noqa: N806
            np.asarray(X_train, dtype=np.float32)
        ).astype(np.float32)
        X_val_scaled = self.x_scaler.transform(  # noqa: N806
            np.asarray(X_val, dtype=np.float32)
        ).astype(np.float32)
        true_train_il = configured_insertion_loss_db(
            y_train,
            self.n_ports,
            self.entry_indices,
        ).astype(np.float32)
        baseline_train_il = configured_insertion_loss_db(
            baseline_train,
            self.n_ports,
            self.entry_indices,
        ).astype(np.float32)
        baseline_val_il = configured_insertion_loss_db(
            baseline_val,
            self.n_ports,
            self.entry_indices,
        ).astype(np.float32)
        n_paths = len(self.entry_indices)
        self.baseline_il_scaler.fit(baseline_train_il.reshape(-1, n_paths))
        baseline_train_scaled = self._scale_baseline_il(baseline_train_il)
        baseline_val_scaled = self._scale_baseline_il(baseline_val_il)
        residual_target = true_train_il - baseline_train_il
        encoded_target = np.concatenate(
            (residual_target, true_train_il),
            axis=-1,
        ).astype(np.float32)

        self.model = build_insertion_loss_residual_head(
            input_width=X_train_scaled.shape[1],
            frequencies_ghz=self.frequencies_ghz,
            n_paths=n_paths,
            hidden_width=self.hidden_width,
            fourier_order=self.fourier_order,
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=self.gradient_clip_norm,
            ),
            loss=ResidualInsertionLoss(
                self.deep_null_threshold_db,
                self.deep_null_weight,
                self.huber_delta_db,
            ),
        )
        selector = _ResidualHeadValidationSelector(
            [X_val_scaled, baseline_val_scaled],
            np.asarray(y_val, dtype=np.float32),
            np.asarray(baseline_val, dtype=np.float32),
            self.n_ports,
            self.entry_indices,
            self.reciprocal,
            complex_nrmse_guard,
            self.prediction_batch_size,
        )
        self.history = self.model.fit(
            [X_train_scaled, baseline_train_scaled],
            encoded_target,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                selector,
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_six_path_mae_db",
                    factor=self.reduce_lr_factor,
                    patience=self.reduce_lr_patience,
                    min_delta=CALLBACK_MIN_DELTA,
                    mode="min",
                    min_lr=self.min_learning_rate,
                    verbose=1,
                ),
            ],
            shuffle=True,
            verbose=verbose,  # type: ignore[assignment]
        )
        self.history.history["val_full_complex_nrmse"] = selector.complex_nrmse_history
        self.history.history["val_six_path_mae_db"] = selector.six_path_mae_history
        self.guard_passed_ = selector.best_epoch is not None
        self.selected_epoch_ = selector.best_epoch
        if self.guard_passed_:
            self.guarded_validation_nrmse_ = selector.best_complex_nrmse
            self.guarded_validation_six_path_mae_db_ = selector.best_six_path_mae
        else:
            self.guarded_validation_nrmse_ = None
            self.guarded_validation_six_path_mae_db_ = None
        return self

    def predict_delta(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
        baseline_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Return six insertion-loss corrections in dB.
        """
        features = self.x_scaler.transform(np.asarray(X, dtype=np.float32)).astype(
            np.float32
        )
        baseline_il = configured_insertion_loss_db(
            baseline_predictions,
            self.n_ports,
            self.entry_indices,
        )
        baseline_scaled = self._scale_baseline_il(baseline_il)
        return np.asarray(
            self.keras_model.predict(
                [features, baseline_scaled],
                batch_size=self.prediction_batch_size,
                verbose=0,
            ),
            dtype=np.float32,
        )

    def predict(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
        baseline_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Return full channels with configured reciprocal magnitudes corrected.
        """
        corrections = self.predict_delta(X, baseline_predictions)
        return apply_insertion_loss_correction(
            baseline_predictions,
            corrections,
            self.n_ports,
            self.entry_indices,
            reciprocal=self.reciprocal,
        )

    def _scale_baseline_il(self, insertion_loss: np.ndarray) -> np.ndarray:
        """
        Apply train-only path-wise scaling to baseline insertion loss.
        """
        values = np.asarray(insertion_loss, dtype=np.float32)
        shape = values.shape
        scaled = self.baseline_il_scaler.transform(
            values.reshape(-1, len(self.entry_indices))
        )
        return scaled.reshape(shape).astype(np.float32)

    def plot_training_history(self) -> Figure:
        """
        Plot residual loss and guarded validation six-path MAE.
        """
        if self.history is None:
            raise RuntimeError("SixPathResidualHead has no training history.")
        history = self.history.history
        epochs = np.arange(1, len(history["loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs, history["loss"], label="training")
        axes[0].set(xlabel="Epoch", ylabel="Residual Huber loss")
        axes[1].plot(
            epochs,
            history["val_six_path_mae_db"],
            label="validation",
        )
        axes[1].set(xlabel="Epoch", ylabel="Six-path MAE (dB)")
        if self.selected_epoch_ is not None:
            for axis in axes:
                axis.axvline(
                    self.selected_epoch_,
                    color="black",
                    linestyle="--",
                    alpha=0.6,
                    label=f"restored epoch {self.selected_epoch_}",
                )
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend()
        fig.suptitle("Six-Path Residual Head Training History")
        fig.tight_layout()
        return fig

    @property
    def keras_model(self) -> keras.Model:
        """
        Return the fitted residual-head Keras model.
        """
        if self.model is None:
            raise RuntimeError("SixPathResidualHead must be fitted before prediction.")
        return self.model


@dataclass
class SixPathComplexResidualHead:
    """
    Correct configured real and imaginary paths around a frozen complex model.
    """

    frequencies_ghz: np.ndarray
    n_ports: int
    entry_indices: tuple[int, ...]
    hidden_width: int = 64
    fourier_order: int = 4
    frequency_rbf_count: int = 0
    deep_null_threshold_magnitude: float = 1e-3
    log_magnitude_weight: float = 0.1
    deep_null_weight: float = 0.02
    log_magnitude_floor: float = 1e-14
    batch_size: int = 64
    epochs: int = 20
    prediction_batch_size: int = 64
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 0.5
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    reciprocal: bool = True
    random_state: int = 128
    x_scaler: StandardScaler = field(init=False, repr=False)
    path_scaler: ComplexRMSScaler = field(init=False, repr=False)
    model: keras.Model | None = field(default=None, init=False, repr=False)
    history: keras.callbacks.History | None = field(
        default=None,
        init=False,
        repr=False,
    )
    selected_epoch_: int | None = field(default=None, init=False)
    guard_passed_: bool | None = field(default=None, init=False)
    guarded_validation_nrmse_: float | None = field(default=None, init=False)
    guarded_validation_six_path_mae_db_: float | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        """
        Normalize configuration and initialize train-only input/path scalers.
        """
        self.frequencies_ghz = np.asarray(self.frequencies_ghz, dtype=np.float32)
        self.entry_indices = tuple(int(index) for index in self.entry_indices)
        self.x_scaler = StandardScaler()
        self.path_scaler = ComplexRMSScaler()

    def fit(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        baseline_train: np.ndarray,
        X_val: np.ndarray,  # pylint: disable=invalid-name
        y_val: np.ndarray,
        baseline_val: np.ndarray,
        *,
        complex_nrmse_guard: float,
        verbose: str | int = 2,
    ) -> SixPathComplexResidualHead:
        """
        Fit complex path corrections and restore a guard-qualified epoch.
        """
        keras.utils.set_random_seed(self.random_state)
        X_train_scaled = self.x_scaler.fit_transform(  # noqa: N806
            np.asarray(X_train, dtype=np.float32)
        ).astype(np.float32)
        X_val_scaled = self.x_scaler.transform(  # noqa: N806
            np.asarray(X_val, dtype=np.float32)
        ).astype(np.float32)
        selected = np.asarray(self.entry_indices, dtype=np.int32)
        true_train_paths = _select_complex_entries(y_train, selected)
        baseline_train_paths = _select_complex_entries(baseline_train, selected)
        baseline_val_paths = _select_complex_entries(baseline_val, selected)
        self.path_scaler.fit(true_train_paths)
        true_train_scaled = self.path_scaler.transform(true_train_paths)
        baseline_train_scaled = self.path_scaler.transform(baseline_train_paths)
        baseline_val_scaled = self.path_scaler.transform(baseline_val_paths)
        entry_scale = self.path_scaler.entry_scale()

        self.model = build_complex_residual_head(
            input_width=X_train_scaled.shape[1],
            frequencies_ghz=self.frequencies_ghz,
            n_paths=len(self.entry_indices),
            hidden_width=self.hidden_width,
            fourier_order=self.fourier_order,
            frequency_rbf_count=self.frequency_rbf_count,
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=self.gradient_clip_norm,
            ),
            loss=ComplexResidualHeadLoss(
                entry_scale,
                self.deep_null_threshold_magnitude,
                self.log_magnitude_weight,
                self.deep_null_weight,
                self.log_magnitude_floor,
            ),
        )
        selector = _ComplexResidualValidationSelector(
            [X_val_scaled, baseline_val_scaled],
            np.asarray(y_val, dtype=np.float32),
            np.asarray(baseline_val, dtype=np.float32),
            self.path_scaler,
            self.n_ports,
            self.entry_indices,
            self.reciprocal,
            complex_nrmse_guard,
            self.prediction_batch_size,
        )
        self.history = self.model.fit(
            [X_train_scaled, baseline_train_scaled],
            true_train_scaled,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                selector,
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_six_path_mae_db",
                    factor=self.reduce_lr_factor,
                    patience=self.reduce_lr_patience,
                    min_delta=CALLBACK_MIN_DELTA,
                    mode="min",
                    min_lr=self.min_learning_rate,
                    verbose=1,
                ),
            ],
            shuffle=True,
            verbose=verbose,  # type: ignore[assignment]
        )
        self.history.history["val_full_complex_nrmse"] = selector.complex_nrmse_history
        self.history.history["val_six_path_mae_db"] = selector.six_path_mae_history
        self.guard_passed_ = selector.best_epoch is not None
        self.selected_epoch_ = selector.best_epoch
        if self.guard_passed_:
            self.guarded_validation_nrmse_ = selector.best_complex_nrmse
            self.guarded_validation_six_path_mae_db_ = selector.best_six_path_mae
        else:
            self.guarded_validation_nrmse_ = None
            self.guarded_validation_six_path_mae_db_ = None
        return self

    def predict_paths(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
        baseline_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Return corrected unscaled complex path channels.
        """
        features = self.x_scaler.transform(np.asarray(X, dtype=np.float32)).astype(
            np.float32
        )
        baseline_paths = _select_complex_entries(
            baseline_predictions,
            np.asarray(self.entry_indices, dtype=np.int32),
        )
        baseline_scaled = self.path_scaler.transform(baseline_paths)
        corrected_scaled = self.keras_model.predict(
            [features, baseline_scaled],
            batch_size=self.prediction_batch_size,
            verbose=0,
        )
        return np.asarray(
            self.path_scaler.inverse_transform(corrected_scaled),
            dtype=np.float32,
        )

    def predict(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
        baseline_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        Return the full matrix with corrected reciprocal complex paths.
        """
        corrected_paths = self.predict_paths(X, baseline_predictions)
        return apply_complex_path_correction(
            baseline_predictions,
            corrected_paths,
            self.n_ports,
            self.entry_indices,
            reciprocal=self.reciprocal,
        )

    def plot_training_history(self) -> Figure:
        """
        Plot complex residual loss and guarded validation path MAE.
        """
        if self.history is None:
            raise RuntimeError("SixPathComplexResidualHead has no training history.")
        history = self.history.history
        epochs = np.arange(1, len(history["loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs, history["loss"], label="training")
        axes[0].set(xlabel="Epoch", ylabel="Complex residual loss")
        axes[1].plot(
            epochs,
            history["val_six_path_mae_db"],
            label="validation",
        )
        axes[1].set(xlabel="Epoch", ylabel="Six-path MAE (dB)")
        if self.selected_epoch_ is not None:
            for axis in axes:
                axis.axvline(
                    self.selected_epoch_,
                    color="black",
                    linestyle="--",
                    alpha=0.6,
                    label=f"restored epoch {self.selected_epoch_}",
                )
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend()
        fig.suptitle("Six-Path Complex Residual Head Training History")
        fig.tight_layout()
        return fig

    @property
    def keras_model(self) -> keras.Model:
        """
        Return the fitted complex residual-head Keras model.
        """
        if self.model is None:
            raise RuntimeError(
                "SixPathComplexResidualHead must be fitted before prediction."
            )
        return self.model


@dataclass
class FullSMatrixModel(NeuralModel):
    """
    Wrapper for frequency-conditioned complete complex S-matrix prediction.
    """

    frequencies_ghz: np.ndarray
    n_ports: int
    hidden_width: int = 128
    residual_blocks: int = 3
    fourier_order: int = 4
    frequency_rbf_count: int = 0
    reciprocal: bool = False
    log_magnitude_weight: float = 0.0
    log_magnitude_entry_indices: tuple[int, ...] = ()
    log_magnitude_floor: float = 1e-6
    targeted_log_magnitude_weight: float = 0.0
    targeted_log_magnitude_entry_indices: tuple[int, ...] = ()
    deep_null_log_magnitude_weight: float = 0.0
    deep_null_threshold_magnitude: float = 0.0
    deep_null_weight: float = 0.0
    passivity_weight: float = 0.0
    batch_size: int = 64
    epochs: int = 100
    prediction_batch_size: int = 64
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 0.5
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    random_state: int = 128
    x_scaler: StandardScaler = field(init=False, repr=False)
    y_scaler: ComplexRMSScaler = field(init=False, repr=False)
    model: keras.Model | None = field(default=None, init=False, repr=False)
    history: keras.callbacks.History | None = field(
        default=None,
        init=False,
        repr=False,
    )
    selected_epoch_: int | None = field(default=None, init=False)
    fine_tune_selected_epoch_: int | None = field(default=None, init=False)
    guard_passed_: bool | None = field(default=None, init=False)
    guarded_validation_nrmse_: float | None = field(default=None, init=False)
    guarded_validation_six_path_mae_db_: float | None = field(
        default=None,
        init=False,
    )

    name: ClassVar[str] = "full_smatrix_neural"

    def __post_init__(self) -> None:
        """
        Normalize the frequency grid and initialize train-only scalers.
        """
        self.frequencies_ghz = np.asarray(self.frequencies_ghz, dtype=np.float32)
        self.log_magnitude_entry_indices = tuple(
            int(index) for index in self.log_magnitude_entry_indices
        )
        self.targeted_log_magnitude_entry_indices = tuple(
            int(index) for index in self.targeted_log_magnitude_entry_indices
        )
        self.x_scaler = StandardScaler()
        self.y_scaler = ComplexRMSScaler()

    def model_name(self) -> str:
        """
        Return the human-readable full-S-matrix model label.
        """
        return "Frequency-Conditioned Full S-Matrix Neural"

    def fit(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,  # pylint: disable=invalid-name
        y_val: np.ndarray | None = None,
        verbose: str | int = 2,
    ) -> FullSMatrixModel:
        """
        Fit using configured-path validation MAE when those paths are supplied.
        """
        if X_val is None or y_val is None:
            raise ValueError("FullSMatrixModel requires validation data.")
        X_train_array = np.asarray(X_train, dtype=np.float32)  # noqa: N806
        X_val_array = np.asarray(X_val, dtype=np.float32)  # noqa: N806
        y_train_array = np.asarray(y_train, dtype=np.float32)
        y_val_array = np.asarray(y_val, dtype=np.float32)

        keras.utils.set_random_seed(self.random_state)
        X_train_scaled = self.x_scaler.fit_transform(X_train_array).astype(  # noqa: N806
            np.float32
        )
        X_val_scaled = self.x_scaler.transform(X_val_array).astype(  # noqa: N806
            np.float32
        )
        self.y_scaler.fit(y_train_array)
        entry_indices = (
            upper_triangle_entry_indices(self.n_ports) if self.reciprocal else None
        )
        y_train_scaled = self.y_scaler.transform(y_train_array, entry_indices)
        y_val_scaled = self.y_scaler.transform(y_val_array, entry_indices)
        entry_scale = self.y_scaler.entry_scale(entry_indices)

        self.model = build_frequency_residual_model(
            input_width=X_train_scaled.shape[1],
            frequencies_ghz=self.frequencies_ghz,
            output_width=2 * len(entry_scale),
            hidden_width=self.hidden_width,
            residual_blocks=self.residual_blocks,
            fourier_order=self.fourier_order,
            frequency_rbf_count=self.frequency_rbf_count,
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=self.gradient_clip_norm,
            ),
            loss=ComplexPhysicsLoss(
                entry_scale,
                self.n_ports,
                reciprocal=self.reciprocal,
                log_magnitude_weight=self.log_magnitude_weight,
                log_magnitude_entry_indices=self.log_magnitude_entry_indices,
                log_magnitude_floor=self.log_magnitude_floor,
                targeted_log_magnitude_weight=(self.targeted_log_magnitude_weight),
                targeted_log_magnitude_entry_indices=(
                    self.targeted_log_magnitude_entry_indices
                ),
                deep_null_log_magnitude_weight=(self.deep_null_log_magnitude_weight),
                deep_null_threshold_magnitude=(self.deep_null_threshold_magnitude),
                deep_null_weight=self.deep_null_weight,
                passivity_weight=self.passivity_weight,
            ),
            metrics=[
                ComplexNRMSE(entry_scale),
                *(
                    [
                        SixPathInsertionLossMAE(
                            entry_scale,
                            self.n_ports,
                            self._selection_entry_indices(),
                            reciprocal=self.reciprocal,
                        )
                    ]
                    if self._selection_entry_indices()
                    else []
                ),
            ],
        )
        selection_metric = (
            "six_path_mae_db" if self._selection_entry_indices() else "complex_nrmse"
        )
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=f"val_{selection_metric}",
            patience=self.early_stopping_patience,
            min_delta=CALLBACK_MIN_DELTA,
            mode="min",
            restore_best_weights=True,
            verbose=1,
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
                    monitor=f"val_{selection_metric}",
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

    def fine_tune(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        X_val: np.ndarray,  # pylint: disable=invalid-name
        y_val: np.ndarray,
        *,
        complex_nrmse_guard: float,
        verbose: str | int = 2,
    ) -> FullSMatrixModel:
        """
        Fine-tune fitted weights and restore only a guard-qualified epoch.
        """
        baseline_selected_epoch = self.selected_epoch_
        X_train_scaled = self.x_scaler.transform(  # noqa: N806
            np.asarray(X_train, dtype=np.float32)
        ).astype(np.float32)
        X_val_scaled = self.x_scaler.transform(  # noqa: N806
            np.asarray(X_val, dtype=np.float32)
        ).astype(np.float32)
        y_train_array = np.asarray(y_train, dtype=np.float32)
        y_val_array = np.asarray(y_val, dtype=np.float32)
        unique_entries = (
            upper_triangle_entry_indices(self.n_ports) if self.reciprocal else None
        )
        y_train_scaled = self.y_scaler.transform(y_train_array, unique_entries)
        entry_scale = self.y_scaler.entry_scale(unique_entries)
        selection_entries = self._selection_entry_indices()
        if not selection_entries:
            raise ValueError("Guarded fine-tuning requires configured path entries.")

        self.keras_model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=self.gradient_clip_norm,
            ),
            loss=ComplexPhysicsLoss(
                entry_scale,
                self.n_ports,
                reciprocal=self.reciprocal,
                log_magnitude_weight=self.log_magnitude_weight,
                log_magnitude_entry_indices=self.log_magnitude_entry_indices,
                log_magnitude_floor=self.log_magnitude_floor,
                targeted_log_magnitude_weight=(self.targeted_log_magnitude_weight),
                targeted_log_magnitude_entry_indices=(
                    self.targeted_log_magnitude_entry_indices
                ),
                deep_null_log_magnitude_weight=(self.deep_null_log_magnitude_weight),
                deep_null_threshold_magnitude=(self.deep_null_threshold_magnitude),
                deep_null_weight=self.deep_null_weight,
                passivity_weight=self.passivity_weight,
            ),
        )
        selector = _GuardedValidationSelector(
            X_val_scaled,
            y_val_array,
            self.y_scaler,
            self.n_ports,
            self.reciprocal,
            selection_entries,
            complex_nrmse_guard,
            self.prediction_batch_size,
        )
        self.history = self.keras_model.fit(
            X_train_scaled,
            y_train_scaled,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                selector,
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_six_path_mae_db",
                    factor=self.reduce_lr_factor,
                    patience=self.reduce_lr_patience,
                    min_delta=CALLBACK_MIN_DELTA,
                    mode="min",
                    min_lr=self.min_learning_rate,
                    verbose=1,
                ),
            ],
            shuffle=True,
            verbose=verbose,  # type: ignore[assignment]
        )
        self.history.history["val_full_complex_nrmse"] = selector.complex_nrmse_history
        self.history.history["val_six_path_mae_db"] = selector.six_path_mae_history
        self.guard_passed_ = selector.best_epoch is not None
        self.fine_tune_selected_epoch_ = selector.best_epoch
        if self.guard_passed_:
            self.selected_epoch_ = selector.best_epoch
            self.guarded_validation_nrmse_ = selector.best_complex_nrmse
            self.guarded_validation_six_path_mae_db_ = selector.best_six_path_mae
        else:
            self.selected_epoch_ = baseline_selected_epoch
            self.guarded_validation_nrmse_ = None
            self.guarded_validation_six_path_mae_db_ = None
        return self

    def _selection_entry_indices(self) -> tuple[int, ...]:
        """
        Return configured paths used for validation model selection.
        """
        return (
            self.targeted_log_magnitude_entry_indices
            or self.log_magnitude_entry_indices
        )

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Return unscaled full real/imaginary S-matrix curves.
        """
        features = self.x_scaler.transform(np.asarray(X, dtype=np.float32)).astype(
            np.float32
        )
        prediction_scaled = self.keras_model.predict(
            features,
            batch_size=self.prediction_batch_size,
            verbose=0,  # type: ignore[assignment]
        )
        entry_indices = (
            upper_triangle_entry_indices(self.n_ports) if self.reciprocal else None
        )
        prediction = self.y_scaler.inverse_transform(
            prediction_scaled,
            entry_indices,
        )
        if self.reciprocal:
            prediction = _expand_reciprocal_channels(prediction, self.n_ports)
        return np.asarray(prediction, dtype=np.float32)

    def evaluate(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
        y: np.ndarray,
    ) -> dict[str, float]:
        """
        Return unscaled full-matrix complex metrics.
        """
        return complex_regression_metrics(y, self.predict(X), self.n_ports)

    def plot_training_history(self) -> Figure:
        """
        Plot the composite loss and validation-selection metric histories.
        """
        if self.history is None or self.selected_epoch_ is None:
            raise RuntimeError(f"{self.name} has no recorded training history.")
        history = self.history.history
        epochs = np.arange(1, len(history["loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        if "val_loss" not in history:
            axes[0].plot(epochs, history["loss"], label="training")
            axes[0].set(xlabel="Epoch", ylabel="Composite loss")
            axes[1].plot(
                epochs,
                history["val_six_path_mae_db"],
                label="validation",
            )
            axes[1].set(xlabel="Epoch", ylabel="Six-path MAE (dB)")
            if self.fine_tune_selected_epoch_ is not None:
                for axis in axes:
                    axis.axvline(
                        self.fine_tune_selected_epoch_,
                        color="black",
                        linestyle="--",
                        alpha=0.6,
                        label=(f"restored epoch {self.fine_tune_selected_epoch_}"),
                    )
            for axis in axes:
                axis.grid(True, alpha=0.3)
                axis.legend()
            fig.suptitle(f"{self.model_name()} Fine-Tuning History")
            fig.tight_layout()
            return fig
        for axis, train_key, val_key, ylabel in (
            (axes[0], "loss", "val_loss", "Composite loss"),
            (
                axes[1],
                (
                    "six_path_mae_db"
                    if "six_path_mae_db" in history
                    else "complex_nrmse"
                ),
                (
                    "val_six_path_mae_db"
                    if "val_six_path_mae_db" in history
                    else "val_complex_nrmse"
                ),
                (
                    "Six-path MAE (dB)"
                    if "six_path_mae_db" in history
                    else "Complex NRMSE"
                ),
            ),
        ):
            axis.plot(epochs, history[train_key], label="training")
            axis.plot(epochs, history[val_key], label="validation")
            axis.axvline(
                self.selected_epoch_,
                color="black",
                linestyle="--",
                alpha=0.6,
                label=f"restored epoch {self.selected_epoch_}",
            )
            axis.set(xlabel="Epoch", ylabel=ylabel)
            axis.grid(True, alpha=0.3)
            axis.legend()
        fig.suptitle(f"{self.model_name()} Training History")
        fig.tight_layout()
        return fig

    @property
    def keras_model(self) -> keras.Model:
        """
        Return the fitted Keras residual model.
        """
        if self.model is None:
            raise RuntimeError("FullSMatrixModel must be fitted before prediction.")
        return self.model


def physics_diagnostics(
    matrices: np.ndarray,
    *,
    batch_size: int = 128,
) -> dict[str, float]:
    """
    Return reciprocity, passivity, and magnitude diagnostics for S-matrices.
    """
    values = np.asarray(matrices)
    reciprocal_numerator = float(
        np.sum(np.abs(values - np.swapaxes(values, -1, -2)) ** 2)
    )
    reciprocal_denominator = float(np.sum(np.abs(values) ** 2))
    largest_singular_values: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        singular_values = np.linalg.svd(batch, compute_uv=False)
        largest_singular_values.append(singular_values[..., 0].reshape(-1))
    sigma_max = np.concatenate(largest_singular_values)
    excess = np.maximum(sigma_max - 1.0, 0.0)
    return {
        "MagnitudeMin": float(np.min(np.abs(values))),
        "MagnitudeMax": float(np.max(np.abs(values))),
        "ReciprocityResidual": float(
            np.sqrt(reciprocal_numerator / reciprocal_denominator)
        ),
        "PassivityViolationFraction": float(np.mean(excess > 0.0)),
        "MeanPassivityExcess": float(np.mean(excess)),
        "PassivityPenalty": float(np.mean(excess**2)),
        "MaximumSingularValue": float(np.max(sigma_max)),
    }


def band_limited_causality_residual(matrices: np.ndarray) -> float:
    """
    Return an indicative finite-band Hilbert-transform residual.
    """
    values = np.asarray(matrices)
    analytic_real = hilbert(values.real, axis=1).imag
    positive_sign = np.mean(np.abs(values.imag - analytic_real) ** 2)
    negative_sign = np.mean(np.abs(values.imag + analytic_real) ** 2)
    energy = np.mean(np.abs(values) ** 2)
    return float(np.sqrt(min(positive_sign, negative_sign) / energy))
