#!/usr/bin/env python3
"""
Tests for model-run artifact save/load helpers.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from sparam_surrogate.models import ScalarRidgeModel
from sparam_surrogate.outputs.runs import (
    KerasWrapperState,
    ModelRunArtifactManager,
    get_run_id,
)
from sparam_surrogate.utils.json_io import read_json


def _features() -> tuple[np.ndarray, np.ndarray]:
    """
    Return small train and validation feature matrices.
    """
    X_train = np.asarray(  # pylint: disable=invalid-name
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=float,
    )
    X_val = np.asarray(  # pylint: disable=invalid-name
        [
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
        ],
        dtype=float,
    )
    return X_train, X_val


def _scalar_target(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """
    Return a small scalar regression target.
    """
    return 2.0 * X[:, 0] - 0.5 * X[:, 1] + 1.0


def _vector_target(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """
    Return two-output targets for a small vector-regression problem.
    """
    return np.column_stack(
        (
            _scalar_target(X),
            -X[:, 0] + 1.5 * X[:, 1] - 2.0,
        )
    )


class TestModelRunArtifactManager:
    """
    Unit tests for model-run artifact persistence.
    """

    def test_get_run_id_formats_timestamped_model_slug(self) -> None:
        """
        Run IDs combine UTC timestamps with stable model-name slugs.
        """
        run_id = get_run_id(
            "Scalar Ridge",
            timestamp=datetime(2026, 7, 5, 15, 30, tzinfo=timezone.utc),
        )

        assert run_id == "20260705T153000Z_scalar_ridge"

    def test_get_run_id_rejects_invalid_timestamp_string(self) -> None:
        """
        Explicit timestamp strings must use the planned run ID timestamp form.
        """
        with pytest.raises(ValueError, match="YYYYMMDDTHHMMSSZ"):
            get_run_id("scalar_ridge", timestamp="2026-07-05")

    def test_create_uses_timestamped_model_run_directory(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Manager creation creates one deterministic model-run directory.
        """
        manager = ModelRunArtifactManager.create(
            tmp_path / "runs",
            "Scalar Ridge",
            timestamp=datetime(2026, 7, 5, 15, 30, tzinfo=timezone.utc),
        )

        assert manager.run_id == "20260705T153000Z_scalar_ridge"
        assert manager.run_dir == (
            tmp_path / "runs" / "20260705T153000Z_scalar_ridge"
        )
        assert manager.run_dir.is_dir()

    def test_create_rejects_existing_run_directory(self, tmp_path: Path) -> None:
        """
        Existing run directories are not overwritten.
        """
        ModelRunArtifactManager.create(
            tmp_path / "runs",
            "scalar_ridge",
            timestamp="20260705T153000Z",
        )

        with pytest.raises(FileExistsError, match="already exists"):
            ModelRunArtifactManager.create(
                tmp_path / "runs",
                "scalar_ridge",
                timestamp="20260705T153000Z",
            )

    def test_sklearn_wrapper_round_trip_preserves_predictions(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Fitted scikit-learn-style wrappers round-trip through ``model.joblib``.
        """
        X_train, X_val = _features()
        y_train = _scalar_target(X_train)
        y_val = _scalar_target(X_val)
        model = ScalarRidgeModel(alphas=(0.001,))
        model.fit(X_train, y_train, X_val, y_val)
        expected = model.predict(X_val)
        manager = ModelRunArtifactManager.create(
            tmp_path / "runs",
            model.name,
            timestamp="20260705T153000Z",
        )
        data_interface = {
            "frequency_hz": np.asarray([1.0e9, 2.0e9]),
            "input_dim": 2,
            "input_features": ["x0", "x1"],
            "output_dim": 1,
            "target_names": ["S7_1_DB"],
            "target_scope": "scalar",
            "target_units": "dB",
        }

        artifact_paths = manager.save_model(model, data_interface=data_interface)
        loaded = manager.load_model()
        metadata = read_json(artifact_paths["metadata"])

        assert artifact_paths == {
            "metadata": manager.run_dir / "metadata.json",
            "model": manager.run_dir / "model.joblib",
        }
        assert not (manager.run_dir / "model.keras").exists()
        assert not (manager.run_dir / "preprocessors.joblib").exists()
        assert metadata["schema_version"] == 1
        assert metadata["run_id"] == manager.run_id
        assert metadata["model"] == {
            "artifact_type": "joblib_wrapper",
            "class_path": "sparam_surrogate.models.ridge.ScalarRidgeModel",
            "family": "sklearn",
            "label": "Scalar Ridge",
            "name": "scalar_ridge",
        }
        assert metadata["artifacts"] == {"model": "model.joblib"}
        assert metadata["data_interface"] == {
            **data_interface,
            "frequency_hz": [1.0e9, 2.0e9],
        }
        assert metadata["selected_hyperparameters"] == {"best_alpha": 0.001}
        assert "metrics" not in metadata
        assert loaded.model_name() == model.model_name()
        np.testing.assert_allclose(loaded.predict(X_val), expected)

    def test_model_save_without_optional_context_writes_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Missing optional target context does not prevent metadata persistence.
        """
        X_train, X_val = _features()
        y_train = _scalar_target(X_train)
        y_val = _scalar_target(X_val)
        model = ScalarRidgeModel(alphas=(0.001,))
        model.fit(X_train, y_train, X_val, y_val)
        manager = ModelRunArtifactManager.create(
            tmp_path / "runs",
            model.name,
            timestamp="20260705T153000Z",
        )

        artifact_paths = manager.save_model(model)
        metadata = read_json(artifact_paths["metadata"])

        assert "data_interface" not in metadata
        assert metadata["selected_hyperparameters"] == {"best_alpha": 0.001}

    def test_unfitted_model_save_raises_clear_error(self, tmp_path: Path) -> None:
        """
        Unfitted models fail before any artifact is written.
        """
        manager = ModelRunArtifactManager.create(
            tmp_path / "runs",
            "scalar_ridge",
            timestamp="20260705T153000Z",
        )

        with pytest.raises(RuntimeError, match="fitted"):
            manager.save_model(ScalarRidgeModel())

        assert not (manager.run_dir / "model.joblib").exists()
        assert not (manager.run_dir / "metadata.json").exists()

    def test_vector_mlp_round_trip_preserves_predictions(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Raw-feature neural wrappers round-trip with Keras and scaler state.
        """
        pytest.importorskip("keras")
        from sparam_surrogate.models.neural_mlp import VectorMLP

        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)
        model = VectorMLP(epochs=1, batch_size=4, random_state=3)
        model.fit(X_train, y_train, X_val, y_val, verbose=0)
        expected = model.predict(X_val)
        manager = ModelRunArtifactManager.create(
            tmp_path / "runs",
            model.name,
            timestamp="20260705T153000Z",
        )

        artifact_paths = manager.save_model(model)
        loaded = manager.load_model()

        assert artifact_paths == {
            "metadata": manager.run_dir / "metadata.json",
            "model": manager.run_dir / "model.keras",
            "preprocessors": manager.run_dir / "preprocessors.joblib",
        }
        metadata = read_json(artifact_paths["metadata"])
        state = KerasWrapperState.load(artifact_paths["preprocessors"])
        assert not (manager.run_dir / "model.joblib").exists()
        assert metadata["model"] == {
            "artifact_type": "keras_with_wrapper_state",
            "class_path": "sparam_surrogate.models.neural_mlp.VectorMLP",
            "family": "keras",
            "label": "Neural MLP",
            "name": "neural_mlp",
        }
        assert metadata["artifacts"] == {
            "model": "model.keras",
            "preprocessors": "preprocessors.joblib",
        }
        assert metadata["training_controls"]["batch_size"] == model.batch_size
        assert metadata["training_controls"]["epochs"] == model.epochs
        assert metadata["training_controls"]["learning_rate"] == model.learning_rate
        assert metadata["training_controls"]["random_state"] == model.random_state
        assert state.class_path.endswith(".VectorMLP")
        assert state.constructor_params["batch_size"] == model.batch_size
        assert "x_scaler" in state.state_attrs
        assert "y_scaler" in state.state_attrs
        assert loaded.model_name() == model.model_name()
        np.testing.assert_allclose(loaded.predict(X_val), expected, atol=1e-5)

    def test_polynomial_mlp_round_trip_preserves_predictions(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Polynomial neural wrappers restore fitted polynomial preprocessing state.
        """
        pytest.importorskip("keras")
        from sparam_surrogate.models.neural_mlp import PolynomialVectorMLP

        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)
        model = PolynomialVectorMLP(
            polynomial_degree=2,
            epochs=1,
            batch_size=4,
            random_state=3,
        )
        model.fit(X_train, y_train, X_val, y_val, verbose=0)
        expected = model.predict(X_val)
        manager = ModelRunArtifactManager.create(
            tmp_path / "runs",
            model.name,
            timestamp="20260705T153000Z",
        )

        manager.save_model(model)
        loaded = manager.load_model()

        assert loaded.expanded_feature_count_ == model.expanded_feature_count_
        np.testing.assert_allclose(loaded.predict(X_val), expected, atol=1e-5)
