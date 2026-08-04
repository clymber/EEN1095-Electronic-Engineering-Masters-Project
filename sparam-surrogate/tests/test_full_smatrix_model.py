"""
Tests for the frequency-conditioned full S-matrix model.
"""

from pathlib import Path

import keras
import numpy as np

from sparam_surrogate.models.full_smatrix import (
    ComplexPhysicsLoss,
    ComplexResidualHeadLoss,
    ComplexRMSScaler,
    FixedFrequencyFeatures,
    FullSMatrixModel,
    ResidualInsertionLoss,
    SixPathComplexResidualHead,
    SixPathInsertionLossMAE,
    SixPathResidualHead,
    apply_complex_path_correction,
    apply_insertion_loss_correction,
    build_complex_residual_head,
    build_frequency_residual_model,
    build_insertion_loss_residual_head,
    complex_smatrix_to_channels,
    physics_diagnostics,
    real_imag_channels_to_smatrix,
)
from sparam_surrogate.outputs.runs import ModelRunArtifactManager


def _smatrix_arrays() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Return tiny reciprocal full-S-matrix train and validation curves.
    """
    frequencies_ghz = np.linspace(1.0, 8.0, 8, dtype=np.float32)
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
        dtype=np.float32,
    )
    X_val = np.asarray(  # pylint: disable=invalid-name
        [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5]],
        dtype=np.float32,
    )

    def targets(designs: np.ndarray) -> np.ndarray:
        """
        Build passive reciprocal two-port curves from design values.
        """
        frequency = frequencies_ghz[np.newaxis, :]
        matrices = np.zeros(
            (len(designs), len(frequencies_ghz), 2, 2),
            dtype=np.complex64,
        )
        matrices[:, :, 0, 0] = (
            0.1 + 0.005 * designs[:, 0, np.newaxis] + 0.001j * frequency
        )
        matrices[:, :, 1, 1] = (
            0.2 + 0.005 * designs[:, 1, np.newaxis] - 0.001j * frequency
        )
        transmission = (
            0.02 + 0.002 * designs.sum(axis=1, keepdims=True) + 0.0005j * frequency
        )
        matrices[:, :, 0, 1] = transmission
        matrices[:, :, 1, 0] = transmission
        return complex_smatrix_to_channels(matrices)

    return X_train, targets(X_train), X_val, targets(X_val), frequencies_ghz


def _tiny_model(frequencies_ghz: np.ndarray) -> FullSMatrixModel:
    """
    Return a small reciprocal model suitable for focused unit tests.
    """
    return FullSMatrixModel(
        frequencies_ghz=frequencies_ghz,
        n_ports=2,
        hidden_width=8,
        residual_blocks=1,
        fourier_order=2,
        reciprocal=True,
        log_magnitude_weight=0.1,
        log_magnitude_entry_indices=(1,),
        log_magnitude_floor=1e-14,
        deep_null_threshold_magnitude=0.03,
        deep_null_weight=9.0,
        batch_size=4,
        epochs=1,
        prediction_batch_size=8,
        random_state=3,
    )


def test_channel_conversion_preserves_external_real_then_imag_order() -> None:
    """
    Channel conversion round-trips row-major real entries followed by imaginary.
    """
    matrices = np.asarray(
        [[[[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]]]],
        dtype=np.complex64,
    )

    channels = complex_smatrix_to_channels(matrices)
    restored = real_imag_channels_to_smatrix(channels, n_ports=2)

    np.testing.assert_array_equal(
        channels,
        [[[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]]],
    )
    np.testing.assert_array_equal(restored, matrices)


def test_complex_rms_scaler_shares_scale_between_real_and_imaginary() -> None:
    """
    Each complex entry uses its train-only RMS magnitude for both components.
    """
    targets = np.asarray(
        [
            [[3.0, 0.0, 4.0, 2.0]],
            [[0.0, 4.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    scaler = ComplexRMSScaler().fit(targets)

    scaled = scaler.transform(targets)
    restored = scaler.inverse_transform(scaled)

    np.testing.assert_allclose(scaler.scale_, [np.sqrt(12.5), np.sqrt(10.0)])
    np.testing.assert_allclose(restored, targets)
    np.testing.assert_allclose(
        np.mean(
            scaled[..., :2] ** 2 + scaled[..., 2:] ** 2,
            axis=(0, 1),
        ),
        np.ones(2),
    )


def test_frequency_residual_model_maps_designs_to_full_curves() -> None:
    """
    The model expands frequency internally and contains residual dense blocks.
    """
    model = build_frequency_residual_model(
        input_width=10,
        frequencies_ghz=np.linspace(0.5, 100.0, 8),
        output_width=288,
        hidden_width=16,
        residual_blocks=2,
        fourier_order=2,
    )

    prediction = model(np.zeros((2, 10), dtype=np.float32), training=False)

    assert model.input_shape == (None, 10)
    assert model.output_shape == (None, 8, 288)
    assert prediction.shape == (2, 8, 288)
    assert not any(
        isinstance(layer, keras.layers.Conv1DTranspose) for layer in model.layers
    )
    assert sum(isinstance(layer, keras.layers.Add) for layer in model.layers) == 2


def test_frequency_features_add_localized_gaussian_basis() -> None:
    """
    Optional RBF features peak at their matching evenly spaced centres.
    """
    frequencies_ghz = np.linspace(1.0, 5.0, 5, dtype=np.float32)
    layer = FixedFrequencyFeatures(
        frequencies_ghz,
        fourier_order=2,
        rbf_count=5,
    )

    features = np.asarray(layer(np.zeros((1, 2), dtype=np.float32)))[0]
    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(layer)
    )

    assert features.shape == (5, 10)
    np.testing.assert_array_equal(np.argmax(features[:, 5:], axis=1), np.arange(5))
    assert restored.rbf_count == 5


def test_rbf_frequency_model_only_expands_the_input_projection() -> None:
    """
    Adding RBFs preserves outputs and adds one projection weight per RBF channel.
    """
    model = build_frequency_residual_model(
        input_width=10,
        frequencies_ghz=np.linspace(0.5, 100.0, 8),
        output_width=12,
        hidden_width=16,
        residual_blocks=2,
        fourier_order=2,
    )
    localized_model = build_frequency_residual_model(
        input_width=10,
        frequencies_ghz=np.linspace(0.5, 100.0, 8),
        output_width=12,
        hidden_width=16,
        residual_blocks=2,
        fourier_order=2,
        frequency_rbf_count=6,
    )

    assert localized_model.output_shape == model.output_shape
    assert localized_model.count_params() - model.count_params() == 6 * 16


def test_complex_physics_loss_combines_complex_and_passivity_terms() -> None:
    """
    The optional passivity term penalizes singular values above one.
    """
    channels = complex_smatrix_to_channels(
        1.5 * np.eye(2, dtype=np.complex64)[np.newaxis, np.newaxis, :, :]
    )
    loss = ComplexPhysicsLoss(
        entry_scale=np.ones(4),
        n_ports=2,
        passivity_weight=1.0,
    )

    result = loss(channels, channels)
    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(loss)
    )

    np.testing.assert_allclose(result, 0.25, rtol=1e-5)
    assert isinstance(restored, ComplexPhysicsLoss)
    assert restored.passivity_weight == 1.0


def test_complex_physics_loss_targets_configured_paths_and_deep_nulls() -> None:
    """
    Targeted log loss ignores other entries and upweights train-defined nulls.
    """
    truth = np.zeros((1, 2, 8), dtype=np.float32)
    prediction = np.zeros_like(truth)
    truth[0, :, 1] = [1e-3, 1e-1]
    prediction[0, :, 1] = [1e-1, 2e-1]
    prediction[0, :, 2] = 0.5
    baseline = ComplexPhysicsLoss(np.ones(4), n_ports=2)
    targeted = ComplexPhysicsLoss(
        np.ones(4),
        n_ports=2,
        log_magnitude_weight=1.0,
        log_magnitude_entry_indices=(1,),
        log_magnitude_floor=1e-14,
    )
    deep_weighted = ComplexPhysicsLoss(
        np.ones(4),
        n_ports=2,
        log_magnitude_weight=1.0,
        log_magnitude_entry_indices=(1,),
        log_magnitude_floor=1e-14,
        deep_null_threshold_magnitude=2e-3,
        deep_null_weight=9.0,
    )

    targeted_term = targeted(truth, prediction) - baseline(truth, prediction)
    weighted_term = deep_weighted(truth, prediction) - baseline(truth, prediction)
    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(deep_weighted)
    )

    assert float(weighted_term) > float(targeted_term)
    assert restored.log_magnitude_entry_indices == (1,)
    assert restored.log_magnitude_floor == 1e-14
    assert restored.deep_null_weight == 9.0


def test_complex_physics_loss_adds_broad_targeted_and_deep_null_terms() -> None:
    """
    Fine-tuning can retain broad coverage while emphasizing six-path deep nulls.
    """
    truth = np.zeros((1, 1, 8), dtype=np.float32)
    prediction = np.zeros_like(truth)
    truth[..., 1] = 1e-3
    truth[..., 2] = 1e-1
    prediction[..., 1] = 1e-1
    prediction[..., 2] = 2e-1
    broad = ComplexPhysicsLoss(
        np.ones(4),
        n_ports=2,
        log_magnitude_weight=0.1,
        log_magnitude_floor=1e-14,
    )
    combined = ComplexPhysicsLoss(
        np.ones(4),
        n_ports=2,
        log_magnitude_weight=0.1,
        log_magnitude_floor=1e-14,
        targeted_log_magnitude_weight=0.02,
        targeted_log_magnitude_entry_indices=(1,),
        deep_null_log_magnitude_weight=0.02,
        deep_null_threshold_magnitude=2e-3,
    )

    broad_value = broad(truth, prediction)
    combined_value = combined(truth, prediction)
    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(combined)
    )

    assert float(combined_value) > float(broad_value)
    assert restored.targeted_log_magnitude_weight == 0.02
    assert restored.targeted_log_magnitude_entry_indices == (1,)
    assert restored.deep_null_log_magnitude_weight == 0.02


def test_six_path_metric_reports_unscaled_insertion_loss_mae() -> None:
    """
    The selection metric measures only configured unscaled complex entries.
    """
    truth = np.zeros((1, 1, 8), dtype=np.float32)
    prediction = np.zeros_like(truth)
    truth[..., 1] = 0.1
    prediction[..., 1] = 0.01
    prediction[..., 2] = 1.0
    metric = SixPathInsertionLossMAE(
        entry_scale=np.ones(4),
        n_ports=2,
        entry_indices=(1,),
    )

    metric.update_state(truth, prediction)

    np.testing.assert_allclose(metric.result(), 20.0, rtol=1e-5)


def test_residual_head_starts_with_zero_insertion_loss_correction() -> None:
    """
    The residual head initially reproduces the frozen baseline exactly.
    """
    model = build_insertion_loss_residual_head(
        input_width=2,
        frequencies_ghz=np.linspace(1.0, 8.0, 8),
        n_paths=1,
        hidden_width=8,
        fourier_order=2,
    )

    prediction = model(
        [
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 8, 1), dtype=np.float32),
        ],
        training=False,
    )

    assert model.input_shape == [(None, 2), (None, 8, 1)]
    assert model.output_shape == (None, 8, 1)
    np.testing.assert_array_equal(prediction, 0.0)


def test_complex_residual_head_starts_at_baseline_complex_paths() -> None:
    """
    Zero-initialized complex residuals reproduce the supplied baseline paths.
    """
    model = build_complex_residual_head(
        input_width=2,
        frequencies_ghz=np.linspace(1.0, 8.0, 8),
        n_paths=1,
        hidden_width=8,
        fourier_order=2,
    )
    baseline_paths = np.linspace(0.1, 0.8, 16, dtype=np.float32).reshape(1, 8, 2)

    prediction = model(
        [
            np.zeros((1, 2), dtype=np.float32),
            baseline_paths,
        ],
        training=False,
    )

    assert model.input_shape == [(None, 2), (None, 8, 2)]
    assert model.output_shape == (None, 8, 2)
    np.testing.assert_array_equal(prediction, baseline_paths)


def test_complex_residual_head_accepts_localized_frequency_features() -> None:
    """
    RBF channels expand only the complex head's first hidden projection.
    """
    frequencies_ghz = np.linspace(1.0, 8.0, 8)
    head = build_complex_residual_head(
        input_width=2,
        frequencies_ghz=frequencies_ghz,
        n_paths=1,
        hidden_width=8,
        fourier_order=2,
    )
    localized_head = build_complex_residual_head(
        input_width=2,
        frequencies_ghz=frequencies_ghz,
        n_paths=1,
        hidden_width=8,
        fourier_order=2,
        frequency_rbf_count=5,
    )

    assert localized_head.output_shape == head.output_shape
    assert localized_head.count_params() - head.count_params() == 5 * 8


def test_residual_loss_adds_separately_normalized_deep_null_term() -> None:
    """
    Deep-null residual error contributes independently of ordinary points.
    """
    encoded_truth = np.asarray([[[20.0, 80.0], [0.0, 20.0]]], dtype=np.float32)
    prediction = np.zeros((1, 2, 1), dtype=np.float32)
    ordinary = ResidualInsertionLoss(
        deep_null_threshold_db=60.0,
        deep_null_weight=0.0,
    )
    deep_weighted = ResidualInsertionLoss(
        deep_null_threshold_db=60.0,
        deep_null_weight=0.2,
    )

    ordinary_value = ordinary(encoded_truth, prediction)
    weighted_value = deep_weighted(encoded_truth, prediction)
    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(deep_weighted)
    )

    assert float(weighted_value) > float(ordinary_value)
    assert restored.deep_null_threshold_db == 60.0
    assert restored.deep_null_weight == 0.2


def test_complex_residual_loss_emphasizes_deep_magnitude_error() -> None:
    """
    Complex residual training adds targeted magnitude and deep-null terms.
    """
    truth = np.asarray([[[1e-3, 0.0], [1e-1, 0.0]]], dtype=np.float32)
    prediction = np.asarray([[[1e-1, 0.0], [2e-1, 0.0]]], dtype=np.float32)
    ordinary = ComplexResidualHeadLoss(
        entry_scale=np.ones(1),
        deep_null_threshold_magnitude=2e-3,
        log_magnitude_weight=0.1,
        deep_null_weight=0.0,
    )
    deep_weighted = ComplexResidualHeadLoss(
        entry_scale=np.ones(1),
        deep_null_threshold_magnitude=2e-3,
        log_magnitude_weight=0.1,
        deep_null_weight=0.02,
    )

    ordinary_value = ordinary(truth, prediction)
    weighted_value = deep_weighted(truth, prediction)
    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(deep_weighted)
    )

    assert float(weighted_value) > float(ordinary_value)
    assert restored.log_magnitude_weight == 0.1
    assert restored.deep_null_weight == 0.02


def test_insertion_loss_correction_preserves_phase_and_reciprocity() -> None:
    """
    A dB correction scales both complex components and their reciprocal entry.
    """
    matrices = np.asarray(
        [[[[0.5, 0.1 + 0.1j], [0.1 + 0.1j, 0.4]]]],
        dtype=np.complex64,
    )
    channels = complex_smatrix_to_channels(matrices)

    corrected = apply_insertion_loss_correction(
        channels,
        np.asarray([[[20.0]]], dtype=np.float32),
        n_ports=2,
        entry_indices=(1,),
        reciprocal=True,
    )
    corrected_matrices = real_imag_channels_to_smatrix(corrected, n_ports=2)

    np.testing.assert_allclose(corrected_matrices[..., 0, 0], 0.5)
    np.testing.assert_allclose(corrected_matrices[..., 0, 1], 0.01 + 0.01j)
    np.testing.assert_allclose(corrected_matrices[..., 1, 0], 0.01 + 0.01j)
    np.testing.assert_allclose(
        np.angle(corrected_matrices[..., 0, 1]),
        np.angle(matrices[..., 0, 1]),
    )


def test_complex_path_correction_replaces_phase_and_preserves_reciprocity() -> None:
    """
    Corrected complex path values are mirrored without changing other entries.
    """
    matrices = np.asarray(
        [[[[0.5, 0.1 + 0.1j], [0.1 + 0.1j, 0.4]]]],
        dtype=np.complex64,
    )
    channels = complex_smatrix_to_channels(matrices)
    corrected_paths = np.asarray([[[0.02, -0.03]]], dtype=np.float32)

    corrected = apply_complex_path_correction(
        channels,
        corrected_paths,
        n_ports=2,
        entry_indices=(1,),
        reciprocal=True,
    )
    corrected_matrices = real_imag_channels_to_smatrix(corrected, n_ports=2)

    np.testing.assert_allclose(corrected_matrices[..., 0, 0], 0.5)
    np.testing.assert_allclose(corrected_matrices[..., 1, 1], 0.4)
    np.testing.assert_allclose(corrected_matrices[..., 0, 1], 0.02 - 0.03j)
    np.testing.assert_allclose(corrected_matrices[..., 1, 0], 0.02 - 0.03j)


def test_residual_head_restores_baseline_when_no_epoch_passes_guard() -> None:
    """
    Guarded residual training falls back to the exact baseline prediction.
    """
    X_train, y_train, X_val, y_val, frequencies_ghz = _smatrix_arrays()
    baseline_train = y_train.copy()
    baseline_val = y_val.copy()
    baseline_train[..., 1] *= 2.0
    baseline_train[..., 2] *= 2.0
    baseline_val[..., 1] *= 2.0
    baseline_val[..., 2] *= 2.0
    head = SixPathResidualHead(
        frequencies_ghz=frequencies_ghz,
        n_ports=2,
        entry_indices=(1,),
        hidden_width=8,
        fourier_order=2,
        batch_size=4,
        epochs=2,
        prediction_batch_size=8,
        random_state=3,
    )

    head.fit(
        X_train,
        y_train,
        baseline_train,
        X_val,
        y_val,
        baseline_val,
        complex_nrmse_guard=0.0,
        verbose=0,
    )

    np.testing.assert_allclose(head.predict(X_val, baseline_val), baseline_val)
    assert head.guard_passed_ is False
    assert head.selected_epoch_ is None
    assert {
        "val_full_complex_nrmse",
        "val_six_path_mae_db",
    }.issubset(head.history.history)

    head.epochs = 1
    head.fit(
        X_train,
        y_train,
        baseline_train,
        X_val,
        y_val,
        baseline_val,
        complex_nrmse_guard=10.0,
        verbose=0,
    )

    assert head.guard_passed_ is True
    assert head.selected_epoch_ == 1
    assert len(head.plot_training_history().axes) == 2


def test_complex_residual_head_uses_guarded_full_matrix_selection() -> None:
    """
    Complex residual training restores baseline unless a corrected epoch qualifies.
    """
    X_train, y_train, X_val, y_val, frequencies_ghz = _smatrix_arrays()
    baseline_train = y_train.copy()
    baseline_val = y_val.copy()
    baseline_train[..., 1] *= 2.0
    baseline_train[..., 2] *= 2.0
    baseline_val[..., 1] *= 2.0
    baseline_val[..., 2] *= 2.0
    head = SixPathComplexResidualHead(
        frequencies_ghz=frequencies_ghz,
        n_ports=2,
        entry_indices=(1,),
        hidden_width=8,
        fourier_order=2,
        batch_size=4,
        epochs=2,
        prediction_batch_size=8,
        random_state=3,
    )

    head.fit(
        X_train,
        y_train,
        baseline_train,
        X_val,
        y_val,
        baseline_val,
        complex_nrmse_guard=0.0,
        verbose=0,
    )

    np.testing.assert_allclose(head.predict(X_val, baseline_val), baseline_val)
    assert head.guard_passed_ is False
    assert head.selected_epoch_ is None

    head.epochs = 1
    head.fit(
        X_train,
        y_train,
        baseline_train,
        X_val,
        y_val,
        baseline_val,
        complex_nrmse_guard=10.0,
        verbose=0,
    )

    assert head.guard_passed_ is True
    assert head.selected_epoch_ == 1
    assert {
        "val_full_complex_nrmse",
        "val_six_path_mae_db",
    }.issubset(head.history.history)
    assert len(head.plot_training_history().axes) == 2


def test_reciprocal_wrapper_reconstructs_and_persists_full_matrix(
    tmp_path: Path,
) -> None:
    """
    A reciprocal model returns the full external contract after artifact reload.
    """
    X_train, y_train, X_val, y_val, frequencies_ghz = _smatrix_arrays()
    model = _tiny_model(frequencies_ghz)

    model.fit(X_train, y_train, X_val, y_val, verbose=0)
    prediction = model.predict(X_val)
    predicted_matrices = real_imag_channels_to_smatrix(prediction, n_ports=2)
    manager = ModelRunArtifactManager.create(
        tmp_path / "runs",
        model.name,
        timestamp="20260731T120000Z",
    )
    manager.save_model(model)
    loaded = manager.load_model()

    assert prediction.shape == y_val.shape
    np.testing.assert_allclose(
        predicted_matrices,
        np.swapaxes(predicted_matrices, -1, -2),
    )
    np.testing.assert_allclose(loaded.predict(X_val), prediction, atol=1e-5)
    assert model.selected_epoch_ == 1
    assert {
        "six_path_mae_db",
        "val_six_path_mae_db",
    }.issubset(model.history.history)
    assert model.evaluate(X_val, y_val).keys() == {
        "ComplexMAE",
        "ComplexNRMSE",
    }


def test_fine_tune_restores_only_an_epoch_that_passes_complex_guard() -> None:
    """
    Guarded fine-tuning keeps baseline weights when no epoch qualifies.
    """
    X_train, y_train, X_val, y_val, frequencies_ghz = _smatrix_arrays()
    model = _tiny_model(frequencies_ghz)
    model.fit(X_train, y_train, X_val, y_val, verbose=0)
    baseline_prediction = model.predict(X_val)
    baseline_x_scale = model.x_scaler.scale_.copy()
    baseline_y_scale = model.y_scaler.scale_.copy()
    model.log_magnitude_entry_indices = ()
    model.targeted_log_magnitude_weight = 0.02
    model.targeted_log_magnitude_entry_indices = (1,)
    model.deep_null_log_magnitude_weight = 0.02
    model.epochs = 2
    model.learning_rate = 1e-4

    model.fine_tune(
        X_train,
        y_train,
        X_val,
        y_val,
        complex_nrmse_guard=0.0,
        verbose=0,
    )

    np.testing.assert_allclose(model.predict(X_val), baseline_prediction, atol=1e-7)
    np.testing.assert_array_equal(model.x_scaler.scale_, baseline_x_scale)
    np.testing.assert_array_equal(model.y_scaler.scale_, baseline_y_scale)
    assert model.guard_passed_ is False
    assert model.fine_tune_selected_epoch_ is None
    assert {
        "val_full_complex_nrmse",
        "val_six_path_mae_db",
    }.issubset(model.history.history)

    model.epochs = 1
    model.fine_tune(
        X_train,
        y_train,
        X_val,
        y_val,
        complex_nrmse_guard=10.0,
        verbose=0,
    )

    assert model.guard_passed_ is True
    assert model.fine_tune_selected_epoch_ == 1
    assert len(model.plot_training_history().axes) == 2


def test_physics_diagnostics_report_reciprocity_and_passivity() -> None:
    """
    Physics diagnostics identify reciprocal passive and non-passive matrices.
    """
    matrices = np.stack(
        [
            0.5 * np.eye(2, dtype=np.complex64),
            1.2 * np.eye(2, dtype=np.complex64),
        ]
    )[np.newaxis, ...]

    metrics = physics_diagnostics(matrices)

    np.testing.assert_allclose(metrics["ReciprocityResidual"], 0.0)
    np.testing.assert_allclose(metrics["PassivityViolationFraction"], 0.5)
    np.testing.assert_allclose(metrics["MeanPassivityExcess"], 0.1, atol=1e-7)
