#!/usr/bin/env python3
"""
Tests for model registry pointer files.
"""

from pathlib import Path

import numpy as np
import pytest

from sparam_surrogate.models import ScalarRidgeModel
from sparam_surrogate.outputs.models import ModelRegistry, ModelRegistryEntry
from sparam_surrogate.outputs.runs import ModelRunArtifactManager
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


def _scalar_target(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """
    Return a small scalar regression target.
    """
    return 2.0 * X[:, 0] - 0.5 * X[:, 1] + 1.0


def _save_scalar_run(
    runs_root: Path,
    timestamp: str,
    *,
    alpha: float,
) -> tuple[ModelRunArtifactManager, ScalarRidgeModel, np.ndarray]:
    """
    Save one fitted scalar ridge run for registry tests.
    """
    X_train, X_val = _features()
    y_train = _scalar_target(X_train)
    y_val = _scalar_target(X_val)
    model = ScalarRidgeModel(alphas=(alpha,))
    model.fit(X_train, y_train, X_val, y_val)
    manager = ModelRunArtifactManager.create(
        runs_root,
        model.name,
        timestamp=timestamp,
    )
    manager.save_model(
        model,
        data_interface={
            "dataset_name": "tiny_fixture",
            "target_names": ["S7_1_DB"],
            "target_scope": "scalar",
            "target_units": "dB",
        },
    )
    return manager, model, X_val


class TestModelRegistry:
    """
    Unit tests for model registry JSON pointer management.
    """

    def test_register_run_writes_registry_pointer_files(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Registering a saved run writes latest, selected, and history JSON files.
        """
        outputs_root = tmp_path / "outputs"
        manager, _, _ = _save_scalar_run(
            outputs_root / "runs",
            "20260705T153000Z",
            alpha=0.001,
        )
        registry = ModelRegistry(outputs_root / "models", project_root=tmp_path)

        entry = registry.register_run(manager.run_dir)

        expected = {
            "artifact_path": (
                f"outputs/runs/{manager.run_id}/model.joblib"
            ),
            "artifact_type": "joblib_wrapper",
            "created_at": "2026-07-05T15:30:00Z",
            "dataset_name": "tiny_fixture",
            "metadata_path": (
                f"outputs/runs/{manager.run_id}/metadata.json"
            ),
            "metrics_path": f"outputs/runs/{manager.run_id}/metrics.json",
            "model_family": "sklearn",
            "model_label": "Scalar Ridge",
            "model_name": "scalar_ridge",
            "run_id": manager.run_id,
            "run_path": f"outputs/runs/{manager.run_id}",
        }
        assert entry == ModelRegistryEntry.from_dict(expected)
        assert read_json(outputs_root / "models" / "latest.json") == {
            "models": {"scalar_ridge": expected},
            "schema_version": 1,
        }
        assert read_json(outputs_root / "models" / "selected.json") == {
            "models": {"scalar_ridge": expected},
            "schema_version": 1,
        }
        assert read_json(outputs_root / "models" / "registry.json") == {
            "runs": [expected],
            "schema_version": 1,
        }

    def test_latest_updates_but_selected_remains_stable(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Newer registrations update latest without replacing selected.
        """
        outputs_root = tmp_path / "outputs"
        registry = ModelRegistry(outputs_root / "models", project_root=tmp_path)
        first, _, _ = _save_scalar_run(
            outputs_root / "runs",
            "20260705T153000Z",
            alpha=0.001,
        )
        second, _, _ = _save_scalar_run(
            outputs_root / "runs",
            "20260705T163000Z",
            alpha=0.01,
        )

        registry.register_run(first.run_dir)
        registry.register_run(second.run_dir)

        assert registry.latest("scalar_ridge").run_id == second.run_id
        assert registry.selected("scalar_ridge").run_id == first.run_id

    def test_promote_updates_selected_pointer(self, tmp_path: Path) -> None:
        """
        Explicit promotion replaces the selected run for a model.
        """
        outputs_root = tmp_path / "outputs"
        registry = ModelRegistry(outputs_root / "models", project_root=tmp_path)
        first, _, _ = _save_scalar_run(
            outputs_root / "runs",
            "20260705T153000Z",
            alpha=0.001,
        )
        second, _, _ = _save_scalar_run(
            outputs_root / "runs",
            "20260705T163000Z",
            alpha=0.01,
        )
        registry.register_run(first.run_dir)
        registry.register_run(second.run_dir)

        selected = registry.promote("scalar_ridge", second.run_id)

        assert selected.run_id == second.run_id
        assert registry.selected("scalar_ridge").run_id == second.run_id

    def test_register_run_upserts_history(self, tmp_path: Path) -> None:
        """
        Registering the same run twice updates history without duplicating it.
        """
        outputs_root = tmp_path / "outputs"
        manager, _, _ = _save_scalar_run(
            outputs_root / "runs",
            "20260705T153000Z",
            alpha=0.001,
        )
        registry = ModelRegistry(outputs_root / "models", project_root=tmp_path)

        registry.register_run(manager.run_dir)
        registry.register_run(manager.run_dir)

        history = read_json(outputs_root / "models" / "registry.json")
        assert len(history["runs"]) == 1
        assert history["runs"][0]["run_id"] == manager.run_id

    def test_load_loads_pointed_model_artifact(self, tmp_path: Path) -> None:
        """
        Registry entries resolve back to loadable model artifacts.
        """
        outputs_root = tmp_path / "outputs"
        manager, model, X_val = _save_scalar_run(  # pylint: disable=invalid-name
            outputs_root / "runs",
            "20260705T153000Z",
            alpha=0.001,
        )
        registry = ModelRegistry(outputs_root / "models", project_root=tmp_path)
        registry.register_run(manager.run_dir)

        loaded = registry.load(registry.selected("scalar_ridge"))

        np.testing.assert_allclose(loaded.predict(X_val), model.predict(X_val))

    def test_register_run_rejects_runs_outside_project_root(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Registry pointers must stay project-relative and portable.
        """
        project_root = tmp_path / "project"
        external_root = tmp_path / "external"
        manager, _, _ = _save_scalar_run(
            external_root / "outputs" / "runs",
            "20260705T153000Z",
            alpha=0.001,
        )
        registry = ModelRegistry(
            project_root / "outputs" / "models",
            project_root=project_root,
        )

        with pytest.raises(ValueError, match="not inside project root"):
            registry.register_run(manager.run_dir)
