#!/usr/bin/env python3
"""
Tests for notebook-friendly model run orchestration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from sparam_surrogate.config.surrogate_config import (
    DatasetConfig,
    PathsConfig,
    PreprocessingConfig,
    ProjectConfig,
    SurrogateConfig,
)
from sparam_surrogate.models import ScalarRidgeModel
from sparam_surrogate.models.base import SparamModel
from sparam_surrogate.outputs.runner import ModelRunRunner
from sparam_surrogate.utils.json_io import read_json


class _History:
    """
    Tiny Keras-like history object for runner tests.
    """

    history: dict[str, list[float]] = {
        "loss": [0.4, 0.2],
        "val_loss": [0.5, 0.25],
    }


class HistoryPlotModel(SparamModel):
    """
    Tiny fitted model with training history and a training-history plot.
    """

    name = "history_plot_model"

    def __init__(self) -> None:
        """
        Create an unfitted model with no recorded training history.
        """
        self.history: _History | None = None

    def fit(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,  # pylint: disable=invalid-name
        y_val: np.ndarray | None = None,
    ) -> SparamModel:
        """
        Record a tiny training history.
        """
        self.history = _History()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Return constant predictions with the right row count.
        """
        return np.zeros(len(X), dtype=float)

    def plot_training_history(self) -> Figure:
        """
        Return a real matplotlib training-history figure.
        """
        if self.history is None:
            raise RuntimeError("Training history is not available.")
        fig, ax = plt.subplots()
        ax.plot(self.history.history["loss"], label="loss")
        ax.plot(self.history.history["val_loss"], label="val_loss")
        ax.legend()
        return fig


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
            cleaned_splits_csv=(
                tmp_path / "data" / "processed" / "cleaned_splits_parameter.csv"
            ),
            freq_expanded_csv=(
                tmp_path / "data" / "processed" / "frequency_expanded_dataset.csv"
            ),
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
            "target_names": ["IL_S7_1_DB", "IL_S8_2_DB"],
            "target_scope": "vector",
            "target_units": "dB",
            "target_representation": "insertion_loss_db",
        },
        extra_metrics={
            "per_target": {
                "IL_S7_1_DB": {
                    "validation": {"MAE": 0.01, "RMSE": 0.02},
                    "test": {"MAE": 0.03, "RMSE": 0.04},
                }
            }
        },
        metric_units={"MAE": "dB", "RMSE": "dB"},
    )

    run_dir = cfg.paths.runs / "20260705T153000Z_scalar_ridge"
    saved_metrics = read_json(run_dir / "metrics.json")["metrics"]
    assert paths["model"] == run_dir / "model.joblib"
    assert saved_metrics["per_target"]["IL_S7_1_DB"]["test"]["MAE"] == 0.03
    assert (run_dir / "config_resolved.json").is_file()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "validation_results.csv").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (cfg.paths.models / "latest.json").is_file()
    assert (cfg.paths.models / "selected.json").is_file()
    assert (
        cfg.paths.benchmarks / "vector_insertion_loss_db_latest.csv"
    ).is_file()
    assert (
        cfg.paths.benchmarks / "vector_insertion_loss_db_selected.csv"
    ).is_file()
    assert (
        cfg.paths.benchmarks / "s7_1_insertion_loss_db_latest.csv"
    ).is_file()
    assert (
        cfg.paths.benchmarks / "s7_1_insertion_loss_db_selected.csv"
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


def test_runner_saves_training_history_figure_when_available(
    tmp_path: Path,
) -> None:
    """
    Runner persistence saves a training-history plot when the model provides one.
    """
    cfg = _config(tmp_path)
    X, y = _arrays()  # pylint: disable=invalid-name
    runner = ModelRunRunner(
        cfg,
        HistoryPlotModel(),
        timestamp="20260705T153000Z",
    )

    runner.train(X, y, X, y)
    paths = runner.persist(refresh_benchmarks=False)

    run_dir = cfg.paths.runs / "20260705T153000Z_history_plot_model"
    figure_path = run_dir / "figures" / "training_history.png"
    manifest = read_json(run_dir / "manifest.json")

    assert paths["training_history"] == run_dir / "training_history.csv"
    assert paths["training_history_figure"] == figure_path
    assert figure_path.is_file()
    assert figure_path.stat().st_size > 0
    assert manifest["figures"] == {
        "training_history": "figures/training_history.png"
    }
