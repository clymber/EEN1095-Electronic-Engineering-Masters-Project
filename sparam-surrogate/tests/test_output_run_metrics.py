#!/usr/bin/env python3
"""
Tests for run-local metrics, validation, and history artifacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparam_surrogate.models import (
    PolynomialModel,
    RandomForestModel,
    ScalarRidgeModel,
)
from sparam_surrogate.outputs.runs import ModelRunArtifactManager, save_run_metrics
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


def _manager(tmp_path: Path, model_name: str) -> ModelRunArtifactManager:
    """
    Return a deterministic run artifact manager for tests.
    """
    return ModelRunArtifactManager.create(
        tmp_path / "runs",
        model_name,
        timestamp="20260705T153000Z",
    )


class TestRunMetrics:
    """
    Tests for metrics.json persistence.
    """

    def test_save_metrics_writes_json_schema_and_units(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Metrics JSON preserves MAE/RMSE keys and optional metric units.
        """
        manager = _manager(tmp_path, "scalar_ridge")

        path = manager.save_metrics(
            {
                "validation": {"MAE": np.float64(0.12), "RMSE": 0.34},
                "test": {"MAE": 0.23, "RMSE": np.float64(0.45)},
            },
            metric_units={"MAE": "dB", "RMSE": "dB"},
        )

        assert path == manager.run_dir / "metrics.json"
        assert read_json(path) == {
            "metric_units": {"MAE": "dB", "RMSE": "dB"},
            "metrics": {
                "test": {"MAE": 0.23, "RMSE": 0.45},
                "validation": {"MAE": 0.12, "RMSE": 0.34},
            },
            "schema_version": 1,
        }

    def test_save_metrics_rejects_existing_file(self, tmp_path: Path) -> None:
        """
        Existing metrics files are not overwritten accidentally.
        """
        manager = _manager(tmp_path, "scalar_ridge")

        manager.save_metrics({"test": {"MAE": 0.1, "RMSE": 0.2}})

        with pytest.raises(FileExistsError, match="metrics.json"):
            manager.save_metrics({"test": {"MAE": 0.3, "RMSE": 0.4}})

    def test_save_run_metrics_accepts_run_dir_first(self, tmp_path: Path) -> None:
        """
        Module-level helpers name the run directory before artifact data.
        """
        run_dir = tmp_path / "runs" / "20260705T153000Z_scalar_ridge"

        path = save_run_metrics(run_dir, {"test": {"MAE": 0.1, "RMSE": 0.2}})

        assert path == run_dir / "metrics.json"
        assert read_json(path)["metrics"] == {"test": {"MAE": 0.1, "RMSE": 0.2}}


class TestValidationResults:
    """
    Tests for validation_results.csv persistence.
    """

    @pytest.mark.parametrize(
        "model",
        [
            ScalarRidgeModel(alphas=(0.001, 0.01)),
            PolynomialModel(degrees=(1, 2), alphas=(0.001,)),
            RandomForestModel(
                n_estimators=2,
                max_depths=(None,),
                min_samples_leafs=(1,),
                n_jobs=1,
            ),
        ],
    )
    def test_save_validation_results_from_model(
        self,
        tmp_path: Path,
        model,
    ) -> None:
        """
        Validation sweeps stored on fitted wrappers save as CSV.
        """
        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)
        model.fit(X_train, y_train, X_val, y_val)
        manager = _manager(tmp_path, model.name)

        path = manager.save_validation_results(model=model)
        table = pd.read_csv(path)

        assert path == manager.run_dir / "validation_results.csv"
        assert {"MAE", "RMSE"}.issubset(table.columns)
        assert len(table) == len(model.validation_results)

    def test_save_validation_results_accepts_explicit_table(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Validation tables can be supplied directly by notebooks.
        """
        manager = _manager(tmp_path, "manual_model")

        path = manager.save_validation_results(
            [{"alpha": 0.001, "MAE": 0.1, "RMSE": 0.2}]
        )

        assert pd.read_csv(path).to_dict("records") == [
            {"alpha": 0.001, "MAE": 0.1, "RMSE": 0.2}
        ]


class TestTrainingHistory:
    """
    Tests for training_history.csv persistence.
    """

    def test_save_training_history_accepts_history_mapping(
        self,
        tmp_path: Path,
    ) -> None:
        """
        History dictionaries save as epoch-indexed CSV files.
        """
        manager = _manager(tmp_path, "neural_mlp")

        path = manager.save_training_history(
            {"loss": [0.3, 0.2], "val_loss": [0.4, 0.25]}
        )

        assert pd.read_csv(path).to_dict("records") == [
            {"epoch": 1, "loss": 0.3, "val_loss": 0.4},
            {"epoch": 2, "loss": 0.2, "val_loss": 0.25},
        ]

    def test_save_training_history_from_keras_model(self, tmp_path: Path) -> None:
        """
        Keras History objects stored on neural wrappers save as CSV.
        """
        pytest.importorskip("keras")
        from sparam_surrogate.models.neural_mlp import VectorMLP

        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)
        model = VectorMLP(epochs=1, batch_size=4, random_state=3)
        model.fit(X_train, y_train, X_val, y_val, verbose=0)
        manager = _manager(tmp_path, model.name)

        path = manager.save_training_history(model=model)
        table = pd.read_csv(path)

        assert path == manager.run_dir / "training_history.csv"
        assert {"epoch", "loss", "val_loss"}.issubset(table.columns)
        assert table["epoch"].tolist() == [1]
