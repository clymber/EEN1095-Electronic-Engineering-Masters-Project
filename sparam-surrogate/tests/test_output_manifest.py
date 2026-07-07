#!/usr/bin/env python3
"""
Tests for minimal run manifest persistence.
"""

from pathlib import Path

import numpy as np

from sparam_surrogate.models import ScalarRidgeModel
from sparam_surrogate.outputs.runs import (
    ModelRunArtifactManager,
    RunManifest,
    save_run_manifest,
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
        ],
        dtype=float,
    )
    X_val = np.asarray(  # pylint: disable=invalid-name
        [
            [0.5, 0.5],
            [1.5, 0.5],
        ],
        dtype=float,
    )
    return X_train, X_val


def _target(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """
    Return a small scalar regression target.
    """
    return 2.0 * X[:, 0] - 0.5 * X[:, 1] + 1.0


def _fitted_scalar_ridge() -> ScalarRidgeModel:
    """
    Return a fitted scalar Ridge wrapper.
    """
    X_train, X_val = _features()
    model = ScalarRidgeModel(alphas=(0.001,))
    model.fit(X_train, _target(X_train), X_val, _target(X_val))
    return model


def test_save_manifest_indexes_existing_run_artifacts(tmp_path: Path) -> None:
    """
    Manifest JSON indexes existing files and model identity.
    """
    model = _fitted_scalar_ridge()
    manager = ModelRunArtifactManager.create(
        tmp_path / "runs",
        model.name,
        timestamp="20260705T153000Z",
    )
    manager.save_model(model)
    manager.save_metrics({"test": {"MAE": 0.1, "RMSE": 0.2}})
    manager.save_validation_results([{"alpha": 0.001, "MAE": 0.1, "RMSE": 0.2}])

    path = manager.save_manifest(
        completed_steps=("train", "validate", "test", "persist")
    )
    manifest = read_json(path)

    assert path == manager.run_dir / "manifest.json"
    assert manifest == {
        "artifacts": {
            "metadata": "metadata.json",
            "metrics": "metrics.json",
            "model": "model.joblib",
            "validation_results": "validation_results.csv",
        },
        "completed_steps": ["train", "validate", "test", "persist"],
        "model": {
            "artifact_type": "joblib_wrapper",
            "class_path": "sparam_surrogate.models.ridge.ScalarRidgeModel",
            "family": "sklearn",
            "label": "Scalar Ridge",
            "name": "scalar_ridge",
        },
        "run_id": manager.run_id,
        "schema_version": 1,
    }

    for artifact_path in manifest["artifacts"].values():
        assert (manager.run_dir / artifact_path).is_file()


def test_run_manifest_from_run_dir_builds_artifact_record(
    tmp_path: Path,
) -> None:
    """
    RunManifest owns the manifest schema and artifact discovery.
    """
    model = _fitted_scalar_ridge()
    manager = ModelRunArtifactManager.create(
        tmp_path / "runs",
        model.name,
        timestamp="20260705T153000Z",
    )
    manager.save_model(model)

    manifest = RunManifest.from_run_dir(
        manager.run_dir,
        completed_steps=("persist",),
    )

    assert manifest.run_id == manager.run_id
    assert manifest.artifacts == {
        "metadata": "metadata.json",
        "model": "model.joblib",
    }
    assert manifest.completed_steps == ("persist",)
    assert manifest.model is not None
    assert manifest.model["name"] == "scalar_ridge"
    assert manifest.to_dict()["schema_version"] == 1


def test_save_manifest_omits_missing_optional_files(tmp_path: Path) -> None:
    """
    Missing optional files are omitted from the manifest.
    """
    model = _fitted_scalar_ridge()
    manager = ModelRunArtifactManager.create(
        tmp_path / "runs",
        model.name,
        timestamp="20260705T153000Z",
    )
    manager.save_model(model)

    path = save_run_manifest(manager.run_dir, completed_steps=("persist",))
    manifest = read_json(path)

    assert manifest["artifacts"] == {
        "metadata": "metadata.json",
        "model": "model.joblib",
    }
    assert manifest["completed_steps"] == ["persist"]


def test_save_manifest_can_describe_run_without_metadata(
    tmp_path: Path,
) -> None:
    """
    Manifest creation still works before model metadata exists.
    """
    run_dir = tmp_path / "runs" / "20260705T153000Z_manual_model"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")

    path = save_run_manifest(run_dir)
    manifest = read_json(path)

    assert manifest == {
        "artifacts": {"metrics": "metrics.json"},
        "completed_steps": [],
        "run_id": "20260705T153000Z_manual_model",
        "schema_version": 1,
    }
