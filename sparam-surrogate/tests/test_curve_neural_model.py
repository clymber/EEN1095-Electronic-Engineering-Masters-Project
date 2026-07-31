"""
Tests for the S-TCNN-style whole-curve neural model.
"""

from pathlib import Path

import keras
import numpy as np

from sparam_surrogate.config.surrogate_config import (
    DatasetConfig,
    PathsConfig,
    PreprocessingConfig,
    ProjectConfig,
    SurrogateConfig,
)
from sparam_surrogate.models.curve_neural import (
    CurveAwareMSE,
    CurveNeuralModel,
    UnscaledMeanAbsoluteError,
    build_curve_decoder,
    frequency_features,
)
from sparam_surrogate.models.neural import NeuralModel
from sparam_surrogate.outputs.runner import ModelRunRunner
from sparam_surrogate.utils.non_neural_modelling_utils import regression_metrics


def _curve_arrays() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Return small design arrays, curve targets, and a shared frequency grid.
    """
    frequencies_ghz = np.linspace(1.0, 8.0, 8)
    X_train = np.asarray(  # pylint: disable=invalid-name
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )
    X_val = np.asarray(  # pylint: disable=invalid-name
        [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5]],
        dtype=float,
    )

    def targets(designs: np.ndarray) -> np.ndarray:
        """
        Build two smooth target channels from design and frequency values.
        """
        frequency = frequencies_ghz[np.newaxis, :]
        first = designs[:, 0, np.newaxis] + 0.1 * frequency
        second = designs[:, 1, np.newaxis] - 0.05 * frequency
        return np.stack((first, second), axis=-1)

    return X_train, targets(X_train), X_val, targets(X_val), frequencies_ghz


def _curve_model(frequencies_ghz: np.ndarray) -> CurveNeuralModel:
    """
    Return a tiny curve model suitable for focused unit tests.
    """
    return CurveNeuralModel(
        frequencies_ghz=frequencies_ghz,
        latent_dim=8,
        decoder_channels=(8, 4, 2),
        kernel_size=3,
        frequency_encoding="fourier",
        fourier_order=2,
        batch_size=4,
        epochs=1,
        prediction_batch_size=8,
        learning_rate=1e-3,
        random_state=3,
    )


def _runner_config(tmp_path: Path) -> SurrogateConfig:
    """
    Return a resolved configuration rooted in a temporary directory.
    """
    return SurrogateConfig(
        project=ProjectConfig(name="curve-test", seed=3),
        paths=PathsConfig(
            raw_data=tmp_path / "data" / "raw",
            processed_data=tmp_path / "data" / "processed",
            outputs=tmp_path / "outputs",
            benchmarks=tmp_path / "outputs" / "benchmarks",
            logs=tmp_path / "outputs" / "logs",
            figures=tmp_path / "outputs" / "figures",
            models=tmp_path / "outputs" / "models",
            reports=tmp_path / "outputs" / "reports",
            runs=tmp_path / "outputs" / "runs",
        ),
        dataset=DatasetConfig(
            name="curve-test",
            path=tmp_path / "data" / "raw" / "curve-test",
            parameter_csv=tmp_path / "data" / "raw" / "curve-test" / "params.csv",
            nports=2,
            ports=((1, 2),),
        ),
        preprocessing=PreprocessingConfig(
            cleaned_splits_csv=tmp_path / "data" / "processed" / "splits.csv",
            freq_expanded_csv=tmp_path / "data" / "processed" / "expanded.csv",
            val_fraction=0.2,
            test_fraction=0.2,
        ),
    )


def test_curve_neural_defaults_use_selected_configuration() -> None:
    """
    Wrapper defaults expose the compact Fourier model and loss retained by NB05.
    """
    model = CurveNeuralModel(frequencies_ghz=np.linspace(1.0, 8.0, 8))

    assert model.latent_dim == 32
    assert model.decoder_channels == (32, 16, 8)
    assert model.kernel_size == 5
    assert model.frequency_encoding == "fourier"
    assert model.derivative_loss_weight == 11.626038


def test_build_curve_decoder_returns_expected_baseline_shape() -> None:
    """
    The baseline decoder maps each design vector to one complete target curve.
    """
    model = build_curve_decoder(
        input_width=10,
        n_frequencies=200,
        n_targets=6,
        latent_dim=16,
        decoder_channels=(16, 8, 4),
        kernel_size=3,
    )

    assert model.input_shape == (None, 10)
    assert model.output_shape == (None, 200, 6)
    assert sum(
        isinstance(layer, keras.layers.Conv1DTranspose) for layer in model.layers
    ) == 3

    predictions = model(np.zeros((2, 10), dtype=np.float32), training=False)
    assert predictions.shape == (2, 200, 6)


def test_build_curve_decoder_accepts_explicit_frequency_features() -> None:
    """
    Frequency-aware variants concatenate an aligned per-frequency input.
    """
    model = build_curve_decoder(
        input_width=10,
        n_frequencies=200,
        n_targets=6,
        latent_dim=16,
        decoder_channels=(16, 8, 4),
        kernel_size=3,
        frequency_feature_width=3,
    )

    assert model.input_shape == [(None, 10), (None, 200, 3)]
    predictions = model(
        [
            np.zeros((2, 10), dtype=np.float32),
            np.zeros((2, 200, 3), dtype=np.float32),
        ],
        training=False,
    )
    assert predictions.shape == (2, 200, 6)


def test_frequency_features_build_linear_and_fourier_encodings() -> None:
    """
    Frequency encodings use only the supplied grid and have stable dimensions.
    """
    frequencies_ghz = np.linspace(1.0, 10.0, 5)

    linear = frequency_features(frequencies_ghz, "linear")
    fourier = frequency_features(frequencies_ghz, "fourier", fourier_order=2)

    assert linear.shape == (5, 1)
    np.testing.assert_allclose(linear[[0, -1], 0], [-1.0, 1.0])
    assert fourier.shape == (5, 5)
    np.testing.assert_allclose(fourier[:, 0], linear[:, 0])
    assert linear.dtype == np.float32
    assert fourier.dtype == np.float32


def test_frequency_features_reject_invalid_inputs() -> None:
    """
    Invalid encodings and constant grids fail with useful errors.
    """
    with np.testing.assert_raises_regex(ValueError, "at least two"):
        frequency_features(np.array([1.0]), "linear")
    with np.testing.assert_raises_regex(ValueError, "constant"):
        frequency_features(np.ones(3), "linear")
    with np.testing.assert_raises_regex(ValueError, "encoding"):
        frequency_features(np.arange(3.0), "unknown")  # type: ignore[arg-type]


def test_unscaled_mae_matches_full_inverse_transform() -> None:
    """
    The selection metric equals MAE after fully undoing target scaling.
    """
    target_scale = np.asarray([2.0, 0.5], dtype=np.float32)
    target_mean = np.asarray([10.0, -3.0], dtype=np.float32)
    y_true_scaled = np.asarray(
        [[[0.0, 1.0], [2.0, -1.0]]],
        dtype=np.float32,
    )
    y_pred_scaled = np.asarray(
        [[[0.5, 0.0], [1.0, 1.0]]],
        dtype=np.float32,
    )
    metric = UnscaledMeanAbsoluteError(target_scale)

    metric.update_state(y_true_scaled, y_pred_scaled)

    y_true = y_true_scaled * target_scale + target_mean
    y_pred = y_pred_scaled * target_scale + target_mean
    expected_mae = np.mean(np.abs(y_pred - y_true))
    np.testing.assert_allclose(metric.result(), expected_mae)


def test_curve_aware_mse_adds_serializable_first_difference_term() -> None:
    """
    Curve-aware loss combines point and first-difference errors as configured.
    """
    y_true = np.zeros((1, 3, 1), dtype=np.float32)
    y_pred = np.asarray([[[0.0], [1.0], [3.0]]], dtype=np.float32)
    loss = CurveAwareMSE(derivative_weight=2.0)

    result = loss(y_true, y_pred)
    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(loss)
    )

    expected_point_mse = (0.0**2 + 1.0**2 + 3.0**2) / 3.0
    expected_derivative_mse = (1.0**2 + 2.0**2) / 2.0
    np.testing.assert_allclose(
        result,
        expected_point_mse + 2.0 * expected_derivative_mse,
    )
    assert isinstance(restored, CurveAwareMSE)
    assert restored.derivative_weight == 2.0


def test_curve_neural_model_owns_scaling_and_curve_shape() -> None:
    """
    The common wrapper owns preprocessing and returns unscaled curve predictions.
    """
    X_train, y_train, X_val, y_val, frequencies_ghz = _curve_arrays()
    model = _curve_model(frequencies_ghz)

    fitted = model.fit(X_train, y_train, X_val, y_val, verbose=0)
    predictions = model.predict(X_val)

    assert fitted is model
    assert isinstance(model, NeuralModel)
    assert model.name == "curve_neural"
    assert predictions.shape == y_val.shape
    assert model.keras_model.input_shape == [(None, 2), (None, 8, 5)]
    assert model.selected_epoch_ == 1
    assert {"mae_db", "val_mae_db"}.issubset(model.history.history)
    np.testing.assert_allclose(model.x_scaler.mean_, X_train.mean(axis=0))
    np.testing.assert_allclose(
        model.y_scaler.mean_,
        y_train.reshape(-1, y_train.shape[-1]).mean(axis=0),
    )

    expected_metrics = regression_metrics(
        y_val.reshape(-1, y_val.shape[-1]),
        predictions.reshape(-1, predictions.shape[-1]),
    )
    np.testing.assert_allclose(
        model.history.history["val_mae_db"][model.selected_epoch_ - 1],
        expected_metrics["MAE"],
        rtol=1e-5,
        atol=1e-5,
    )
    assert model.evaluate(X_val, y_val) == expected_metrics


def test_curve_training_history_marks_mae_selected_epoch() -> None:
    """
    Curve history plots mark the epoch restored by validation MAE in dB.
    """
    model = CurveNeuralModel(frequencies_ghz=np.linspace(1.0, 8.0, 8))
    history = keras.callbacks.History()
    history.history = {
        "loss": [0.8, 0.6, 0.5],
        "val_loss": [0.7, 0.5, 0.4],
        "mae_db": [4.0, 3.0, 2.5],
        "val_mae_db": [3.8, 2.7, 2.9],
    }
    model.history = history
    model.selected_epoch_ = 2

    figure = model.plot_training_history()

    assert len(figure.axes) == 2
    assert figure.axes[0].get_ylabel() == "Composite loss (scaled target units)"
    assert figure.axes[1].get_ylabel() == "MAE (dB)"
    for axis in figure.axes:
        selection_lines = [
            line
            for line in axis.lines
            if line.get_label() == "restored epoch 2"
        ]
        assert len(selection_lines) == 1
        np.testing.assert_allclose(selection_lines[0].get_xdata(), [2, 2])


def test_curve_model_runner_and_artifact_round_trip(tmp_path: Path) -> None:
    """
    Runner persistence reloads a curve wrapper with identical predictions and state.
    """
    X_train, y_train, X_val, y_val, frequencies_ghz = _curve_arrays()
    runner = ModelRunRunner(
        _runner_config(tmp_path),
        _curve_model(frequencies_ghz),
        timestamp="20260729T120000Z",
    )

    model = runner.train(X_train, y_train, X_val, y_val)
    validation_metrics = runner.validate(X_val, y_val)
    expected = model.predict(X_val)
    paths = runner.persist(
        data_interface={
            "dataset_name": "curve-test",
            "input_features": ["x0", "x1"],
            "target_names": ["y0", "y1"],
            "target_scope": "curve",
            "target_units": "dB",
            "target_representation": "insertion_loss_db",
            "input_shape": [2],
            "output_shape": [8, 2],
            "frequency_units": "GHz",
            "frequencies_ghz": frequencies_ghz,
        },
        metric_units={"MAE": "dB", "RMSE": "dB"},
        refresh_benchmarks=False,
    )
    loaded = runner.manager.load_model()

    assert validation_metrics.keys() == {"MAE", "RMSE"}
    assert paths["model"].name == "model.keras"
    assert isinstance(loaded, CurveNeuralModel)
    np.testing.assert_allclose(loaded.predict(X_val), expected, atol=1e-5)
    np.testing.assert_allclose(loaded.frequencies_ghz, frequencies_ghz)
    np.testing.assert_allclose(loaded.x_scaler.mean_, model.x_scaler.mean_)
    np.testing.assert_allclose(loaded.y_scaler.mean_, model.y_scaler.mean_)
    assert loaded.decoder_channels == (8, 4, 2)
    assert loaded.frequency_encoding == "fourier"
    assert loaded.fourier_order == 2
