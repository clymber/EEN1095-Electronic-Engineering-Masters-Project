#!/usr/bin/env python3
"""
Tests for notebook-friendly model run orchestration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sparam_surrogate.config.surrogate_config import (
    DatasetConfig,
    PathsConfig,
    PreprocessingConfig,
    ProjectConfig,
    SurrogateConfig,
)
from sparam_surrogate.models import ScalarRidgeModel
from sparam_surrogate.outputs.runner import ModelRunRunner
from sparam_surrogate.utils.json_io import read_json


def _config(tmp_path: Path) -> SurrogateConfig:
    """
    Return a small resolved surrogate configuration.
    """
    return SurrogateConfig(
        project=ProjectConfig(name="demo-project", seed=123),
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
            name="demo-dataset",
            path=tmp_path / "data" / "raw" / "demo-dataset",
            parameter_csv=tmp_path / "data" / "raw" / "demo-dataset" / "params.csv",
            nports=2,
            ports=((1, 2),),
        ),
        preprocessing=PreprocessingConfig(
            processed_csv=tmp_path / "data" / "processed" / "cleaned.csv",
            val_fraction=0.2,
            test_fraction=0.1,
        ),
    )


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    Return a tiny feature and target pair.
    """
    X = np.asarray([[0.0], [1.0], [2.0]], dtype=float)  # pylint: disable=invalid-name
    y = np.asarray([1.0, 1.5, 2.0], dtype=float)
    return X, y


def test_runner_persists_full_workflow_and_refreshes_benchmarks(
    tmp_path: Path,
) -> None:
    """
    Full runner workflow persists artifacts, registry pointers, and benchmarks.
    """
    cfg = _config(tmp_path)
    X, y = _arrays()  # pylint: disable=invalid-name
    model = ScalarRidgeModel(alphas=(0.001,))
    runner = ModelRunRunner(cfg, model, timestamp="20260705T153000Z")

    runner.train(X, y, X, y)
    runner.validate(X, y)
    runner.test(X, y)
    paths = runner.persist(
        data_interface={
            "target_names": ["S7_1_DB", "S8_2_DB"],
            "target_scope": "vector",
            "target_units": "dB",
        },
        metric_units={"MAE": "dB", "RMSE": "dB"},
    )

    run_dir = cfg.paths.runs / "20260705T153000Z_scalar_ridge"
    assert paths["model"] == run_dir / "model.joblib"
    assert (run_dir / "config_resolved.json").is_file()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "validation_results.csv").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (cfg.paths.models / "latest.json").is_file()
    assert (cfg.paths.models / "selected.json").is_file()
    assert (
        cfg.paths.benchmarks / "vector_magnitude_db_latest.csv"
    ).is_file()
    assert (
        cfg.paths.benchmarks / "vector_magnitude_db_selected.csv"
    ).is_file()
    assert read_json(run_dir / "manifest.json")["completed_steps"] == [
        "train",
        "validate",
        "test",
        "persist",
    ]


def test_runner_supports_training_only_persistence(tmp_path: Path) -> None:
    """
    Training-only persistence skips metrics and benchmark summaries.
    """
    cfg = _config(tmp_path)
    X, y = _arrays()  # pylint: disable=invalid-name
    model = ScalarRidgeModel(alphas=(0.001,))
    runner = ModelRunRunner(cfg, model, timestamp="20260705T153000Z")

    runner.train(X, y, X, y)
    runner.persist(refresh_benchmarks=False)

    run_dir = cfg.paths.runs / "20260705T153000Z_scalar_ridge"
    assert (run_dir / "model.joblib").is_file()
    assert not (run_dir / "metrics.json").exists()
    assert not cfg.paths.benchmarks.exists()
    assert read_json(run_dir / "manifest.json")["completed_steps"] == [
        "train",
        "persist",
    ]
