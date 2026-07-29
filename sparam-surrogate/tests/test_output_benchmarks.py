#!/usr/bin/env python3
"""
Tests for benchmark summary CSV refresh helpers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sparam_surrogate.models import ScalarRidgeModel
from sparam_surrogate.outputs.benchmarks import (
    refresh_benchmarks,
    regenerate_benchmarks,
)
from sparam_surrogate.outputs.models import ModelRegistry
from sparam_surrogate.outputs.runs import ModelRunArtifactManager

METRICS_A = {
    "validation": {"MAE": 0.12, "RMSE": 0.34},
    "test": {"MAE": 0.23, "RMSE": 0.45},
}

METRICS_B = {
    "validation": {"MAE": 0.5, "RMSE": 0.6},
    "test": {"MAE": 0.7, "RMSE": 0.8},
}


def _features() -> tuple[np.ndarray, np.ndarray]:
    """
    Return small train and validation feature matrices.
    """
    X_train = np.asarray(  # pylint: disable=invalid-name
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    X_val = np.asarray(  # pylint: disable=invalid-name
        [[0.5, 0.5], [1.5, 0.5]],
        dtype=float,
    )
    return X_train, X_val


def _target(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """
    Return a small scalar regression target.
    """
    return 2.0 * X[:, 0] - 0.5 * X[:, 1] + 1.0


def _registry(outputs_root: Path, project_root: Path) -> ModelRegistry:
    """
    Return a model registry rooted under an outputs directory.
    """
    return ModelRegistry(outputs_root / "models", project_root=project_root)


def _save_run(
    outputs_root: Path,
    timestamp: str,
    *,
    metrics: dict | None = METRICS_A,
    target_scope: str = "vector",
    target_names: list[str] | None = None,
    target_representation: str | None = None,
) -> ModelRunArtifactManager:
    """
    Save and optionally score one lightweight model run.
    """
    X_train, X_val = _features()
    model = ScalarRidgeModel(alphas=(0.001,))
    model.fit(X_train, _target(X_train), X_val, _target(X_val))
    manager = ModelRunArtifactManager.create(
        outputs_root / "runs",
        model.name,
        timestamp=timestamp,
    )
    data_interface = {
        "target_names": target_names or ["S7_1_DB", "S8_2_DB"],
        "target_scope": target_scope,
        "target_units": "dB",
    }
    if target_representation is not None:
        data_interface["target_representation"] = target_representation
    manager.save_model(model, data_interface=data_interface)
    if metrics is not None:
        manager.save_metrics(metrics, metric_units={"MAE": "dB", "RMSE": "dB"})
    return manager


def _register_two_runs(
    outputs_root: Path,
    project_root: Path,
) -> tuple[ModelRegistry, ModelRunArtifactManager, ModelRunArtifactManager]:
    """
    Register two scored vector runs for latest/selected tests.
    """
    registry = _registry(outputs_root, project_root)
    first = _save_run(outputs_root, "20260705T153000Z", metrics=METRICS_A)
    second = _save_run(outputs_root, "20260705T163000Z", metrics=METRICS_B)
    registry.register_run(first.run_dir)
    registry.register_run(second.run_dir)
    return registry, first, second


def _rows(path: Path) -> list[dict]:
    """
    Return CSV rows as dictionaries.
    """
    return pd.read_csv(path).to_dict("records")


def _vector_path(outputs_root: Path, selection: str) -> Path:
    """
    Return the vector benchmark path for a selection.
    """
    return outputs_root / "benchmarks" / f"vector_magnitude_db_{selection}.csv"


def _s7_path(outputs_root: Path, selection: str) -> Path:
    """
    Return the S7_1 benchmark path for a selection.
    """
    return outputs_root / "benchmarks" / f"s7_1_magnitude_db_{selection}.csv"


def _il_vector_path(outputs_root: Path, selection: str) -> Path:
    """
    Return the vector insertion-loss benchmark path for a selection.
    """
    return (
        outputs_root
        / "benchmarks"
        / f"vector_insertion_loss_db_{selection}.csv"
    )


def _il_s7_path(outputs_root: Path, selection: str) -> Path:
    """
    Return the S7_1 insertion-loss benchmark path for a selection.
    """
    return (
        outputs_root
        / "benchmarks"
        / f"s7_1_insertion_loss_db_{selection}.csv"
    )


def test_refresh_latest_benchmarks_writes_and_replaces_vector_row(
    tmp_path: Path,
) -> None:
    """
    Latest refresh writes one vector row and replaces it for newer runs.
    """
    outputs_root = tmp_path / "outputs"
    registry = _registry(outputs_root, tmp_path)
    first = _save_run(outputs_root, "20260705T153000Z", metrics=METRICS_A)
    registry.register_run(first.run_dir)

    paths = refresh_benchmarks(
        outputs_root / "benchmarks",
        registry,
        "scalar_ridge",
        selection="latest",
    )

    path = _vector_path(outputs_root, "latest")
    assert paths == [path]
    assert _rows(path)[0]["run_id"] == first.run_id

    second = _save_run(outputs_root, "20260705T163000Z", metrics=METRICS_B)
    registry.register_run(second.run_dir)
    refresh_benchmarks(
        outputs_root / "benchmarks",
        registry,
        "scalar_ridge",
        selection="latest",
    )

    rows = _rows(path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == second.run_id
    assert rows[0]["test_mae_db"] == 0.7


def test_refresh_selected_benchmarks_uses_selected_pointer(
    tmp_path: Path,
) -> None:
    """
    Selected benchmark summaries remain stable until promotion.
    """
    outputs_root = tmp_path / "outputs"
    registry, first, _ = _register_two_runs(outputs_root, tmp_path)

    refresh_benchmarks(
        outputs_root / "benchmarks",
        registry,
        "scalar_ridge",
        selection="selected",
    )

    rows = _rows(_vector_path(outputs_root, "selected"))
    assert rows[0]["run_id"] == first.run_id
    assert rows[0]["test_mae_db"] == 0.23


def test_refresh_benchmarks_writes_s7_and_per_target_summaries(
    tmp_path: Path,
) -> None:
    """
    S7_1 and per-target summaries are written when compatible metrics exist.
    """
    outputs_root = tmp_path / "outputs"
    metrics = {
        **METRICS_A,
        "per_target": {
            "S7_1_DB": {
                "validation": {"MAE": 0.11, "RMSE": 0.33},
                "test": {"MAE": 0.22, "RMSE": 0.44},
            }
        },
    }
    manager = _save_run(
        outputs_root,
        "20260705T153000Z",
        metrics=metrics,
        target_scope="scalar",
        target_names=["S7_1_DB"],
    )
    registry = _registry(outputs_root, tmp_path)
    registry.register_run(manager.run_dir)

    paths = refresh_benchmarks(
        outputs_root / "benchmarks",
        registry,
        "scalar_ridge",
        selection="latest",
    )

    assert [path.name for path in paths] == [
        "s7_1_magnitude_db_latest.csv",
        "per_target_magnitude_db_latest.csv",
    ]
    assert _rows(paths[0])[0]["run_id"] == manager.run_id
    assert _rows(paths[1]) == [
        {
            "model_name": "scalar_ridge",
            "target_name": "S7_1_DB",
            "val_mae_db": 0.11,
            "val_rmse_db": 0.33,
            "test_mae_db": 0.22,
            "test_rmse_db": 0.44,
            "run_id": manager.run_id,
        }
    ]


def test_refresh_benchmarks_writes_s7_row_for_vector_per_target_metrics(
    tmp_path: Path,
) -> None:
    """
    Vector runs contribute S7_1 summaries from per-target metrics.
    """
    outputs_root = tmp_path / "outputs"
    metrics = {
        **METRICS_A,
        "per_target": {
            "S7_1_DB": {
                "validation": {"MAE": 0.11, "RMSE": 0.33},
                "test": {"MAE": 0.22, "RMSE": 0.44},
            },
            "S8_2_DB": {
                "validation": {"MAE": 0.15, "RMSE": 0.35},
                "test": {"MAE": 0.25, "RMSE": 0.45},
            },
        },
    }
    manager = _save_run(
        outputs_root,
        "20260705T153000Z",
        metrics=metrics,
        target_scope="vector",
        target_names=["S7_1_DB", "S8_2_DB"],
    )
    registry = _registry(outputs_root, tmp_path)
    registry.register_run(manager.run_dir)

    paths = refresh_benchmarks(
        outputs_root / "benchmarks",
        registry,
        "scalar_ridge",
        selection="latest",
    )

    assert [path.name for path in paths] == [
        "vector_magnitude_db_latest.csv",
        "s7_1_magnitude_db_latest.csv",
        "per_target_magnitude_db_latest.csv",
    ]
    assert _rows(_s7_path(outputs_root, "latest")) == [
        {
            "model_name": "scalar_ridge",
            "val_mae_db": 0.11,
            "val_rmse_db": 0.33,
            "test_mae_db": 0.22,
            "test_rmse_db": 0.44,
            "run_id": manager.run_id,
        }
    ]


def test_refresh_benchmarks_separates_insertion_loss_from_magnitude(
    tmp_path: Path,
) -> None:
    """
    Insertion-loss runs write IL tables without changing magnitude tables.
    """
    outputs_root = tmp_path / "outputs"
    metrics = {
        **METRICS_A,
        "per_target": {
            "IL_S7_1_DB": {
                "validation": {"MAE": 0.11, "RMSE": 0.33},
                "test": {"MAE": 0.22, "RMSE": 0.44},
            },
            "IL_S8_2_DB": {
                "validation": {"MAE": 0.15, "RMSE": 0.35},
                "test": {"MAE": 0.25, "RMSE": 0.45},
            },
        },
    }
    manager = _save_run(
        outputs_root,
        "20260705T153000Z",
        metrics=metrics,
        target_names=["IL_S7_1_DB", "IL_S8_2_DB"],
        target_representation="insertion_loss_db",
    )
    registry = _registry(outputs_root, tmp_path)
    registry.register_run(manager.run_dir)

    paths = refresh_benchmarks(
        outputs_root / "benchmarks",
        registry,
        "scalar_ridge",
        selection="latest",
    )

    assert [path.name for path in paths] == [
        "vector_insertion_loss_db_latest.csv",
        "s7_1_insertion_loss_db_latest.csv",
        "per_target_insertion_loss_db_latest.csv",
    ]
    assert _rows(_il_vector_path(outputs_root, "latest"))[0]["run_id"] == (
        manager.run_id
    )
    assert _rows(_il_s7_path(outputs_root, "latest"))[0]["test_mae_db"] == 0.22
    assert not _vector_path(outputs_root, "latest").exists()
    assert not _s7_path(outputs_root, "latest").exists()


def test_refresh_benchmarks_skips_runs_without_metrics(tmp_path: Path) -> None:
    """
    Missing metrics.json leaves benchmark summaries unchanged.
    """
    outputs_root = tmp_path / "outputs"
    manager = _save_run(outputs_root, "20260705T153000Z", metrics=None)
    registry = _registry(outputs_root, tmp_path)
    registry.register_run(manager.run_dir)

    paths = refresh_benchmarks(
        outputs_root / "benchmarks",
        registry,
        "scalar_ridge",
        selection="latest",
    )

    assert paths == []
    assert not (outputs_root / "benchmarks").exists()


def test_regenerate_benchmarks_rebuilds_and_replaces(
    tmp_path: Path,
) -> None:
    """
    Regeneration rebuilds latest and selected summaries, removing stale rows.
    """
    outputs_root = tmp_path / "outputs"
    registry, first, second = _register_two_runs(outputs_root, tmp_path)
    stale_path = _vector_path(outputs_root, "latest")
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text(
        "model_name,val_mae_db,val_rmse_db,test_mae_db,test_rmse_db,run_id\n"
        "stale_model,9.0,9.0,9.0,9.0,old_run\n",
        encoding="utf-8",
    )

    paths = regenerate_benchmarks(outputs_root / "benchmarks", registry)

    assert paths == [stale_path, _vector_path(outputs_root, "selected")]
    assert _rows(paths[0])[0]["run_id"] == second.run_id
    assert _rows(paths[1])[0]["run_id"] == first.run_id

    refresh_benchmarks(
        outputs_root / "benchmarks",
        registry,
        "scalar_ridge",
        selection="latest",
        regenerate=True,
    )
    assert _rows(stale_path)[0]["model_name"] == "scalar_ridge"
