"""
Notebook-friendly orchestration for one model run.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from sparam_surrogate.config.surrogate_config import SurrogateConfig
from sparam_surrogate.models.base import SparamModel
from sparam_surrogate.outputs.benchmarks import refresh_benchmarks as refresh_rows
from sparam_surrogate.outputs.models import ModelRegistry
from sparam_surrogate.outputs.runs import (
    ModelRunArtifactManager,
    create_run_artifact_dirs,
    save_run_config,
    save_run_environment,
)


class ModelRunRunner:
    """
    Coordinate train, validate, test, persist, and benchmark refresh steps.
    """

    def __init__(
        self,
        cfg: SurrogateConfig,
        model: SparamModel,
        *,
        timestamp: datetime | str | None = None,
        project_root: Path | str | None = None,
    ) -> None:
        """
        Create a runner with a fresh run directory.
        """
        self.cfg = cfg
        self.model = model
        self.manager = ModelRunArtifactManager.create(
            cfg.paths.runs,
            model.name,
            timestamp=timestamp,
        )
        if project_root is None:
            project_root = cfg.paths.outputs.parent
        self.registry = ModelRegistry(cfg.paths.models, project_root=project_root)
        self.validation_metrics = None
        self.test_metrics = None
        self.completed_steps = []

    def train(
        self,
        X_train: Any,  # pylint: disable=invalid-name
        y_train: Any,
        X_val: Any | None = None,  # pylint: disable=invalid-name
        y_val: Any | None = None,
    ) -> SparamModel:
        """
        Fit the model and record the train step.
        """
        self.model.fit(X_train, y_train, X_val, y_val)
        self._record_step("train")
        return self.model

    def validate(
        self,
        X_val: Any,  # pylint: disable=invalid-name
        y_val: Any,
    ) -> dict[str, float]:
        """
        Evaluate validation metrics and record the validate step.
        """
        self.validation_metrics = self.model.evaluate(X_val, y_val)
        self._record_step("validate")
        return self.validation_metrics

    def test(
        self,
        X_test: Any,  # pylint: disable=invalid-name
        y_test: Any,
    ) -> dict[str, float]:
        """
        Evaluate test metrics and record the test step.
        """
        self.test_metrics = self.model.evaluate(X_test, y_test)
        self._record_step("test")
        return self.test_metrics

    def persist(
        self,
        *,
        data_interface: Mapping[str, Any] | None = None,
        metric_units: Mapping[str, str] | None = None,
        refresh_benchmarks: bool = True,
    ) -> dict[str, Path]:
        """
        Persist available artifacts and return created artifact paths.
        """
        artifact_paths = self.manager.save_model(
            self.model,
            data_interface=data_interface,
        )
        artifact_paths["config"] = save_run_config(self.manager.run_dir, self.cfg)
        artifact_paths["environment"] = save_run_environment(self.manager.run_dir)

        if getattr(self.model, "validation_results", None) is not None:
            artifact_paths["validation_results"] = (
                self.manager.save_validation_results(model=self.model)
            )
        if getattr(self.model, "history", None) is not None:
            artifact_paths["training_history"] = (
                self.manager.save_training_history(model=self.model)
            )

        metrics: dict[str, dict[str, float]] = {}
        if self.validation_metrics is not None:
            metrics["validation"] = self.validation_metrics
        if self.test_metrics is not None:
            metrics["test"] = self.test_metrics
        if metrics:
            artifact_paths["metrics"] = self.manager.save_metrics(
                metrics,
                metric_units=metric_units,
            )

        create_run_artifact_dirs(self.manager.run_dir)
        self.registry.register_run(self.manager.run_dir)
        self._record_step("persist")

        if refresh_benchmarks and self.test_metrics is not None:
            for selection in ("latest", "selected"):
                try:
                    refresh_rows(
                        self.cfg.paths.benchmarks,
                        self.registry,
                        self.model.name,
                        selection=selection,
                    )
                except Exception:
                    pass

        artifact_paths["manifest"] = self.manager.save_manifest(
            completed_steps=self.completed_steps
        )
        return artifact_paths

    def _record_step(self, step: str) -> None:
        """
        Append a completed step once while preserving workflow order.
        """
        if step not in self.completed_steps:
            self.completed_steps.append(step)
