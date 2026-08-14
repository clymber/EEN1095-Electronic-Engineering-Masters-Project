"""
Notebook-friendly orchestration for one model run.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

from sparam_surrogate.config.surrogate_config import SurrogateConfig
from sparam_surrogate.models.base import SparamModel
from sparam_surrogate.outputs.benchmarks import (
    refresh_benchmarks as refresh_rows,
)
from sparam_surrogate.outputs.benchmarks import (
    regenerate_benchmarks as regenerate_rows,
)
from sparam_surrogate.outputs.models import ModelRegistry
from sparam_surrogate.outputs.runs import (
    ModelRunArtifactManager,
    create_run_artifact_dirs,
    save_run_config,
    save_run_environment,
)

ModelT = TypeVar("ModelT", bound=SparamModel)


class ModelRunRunner(Generic[ModelT]):
    """
    Coordinate train, validate, test, persist, and benchmark refresh steps.
    """

    def __init__(
        self,
        cfg: SurrogateConfig,
        model: ModelT,
        *,
        timestamp: datetime | str | None = None,
        project_root: Path | str | None = None,
    ) -> None:
        """
        Create a runner with a fresh run directory.
        """
        self.cfg = cfg
        self.model: ModelT = model
        self.manager = ModelRunArtifactManager.create(
            cfg.paths.runs,
            model.name,
            timestamp=timestamp,
        )
        if project_root is None:
            project_root = cfg.paths.outputs.parent
        self.registry = ModelRegistry(cfg.paths.models, project_root=project_root)
        self.validation_metrics: dict[str, float] | None = None
        self.test_metrics: dict[str, float] | None = None
        self.completed_steps: list[str] = []

    def train(
        self,
        X_train: Any,  # pylint: disable=invalid-name
        y_train: Any,
        X_val: Any | None = None,  # pylint: disable=invalid-name
        y_val: Any | None = None,
    ) -> ModelT:
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
        metrics = self.model.evaluate(X_val, y_val)
        self.validation_metrics = metrics
        self._record_step("validate")
        return metrics

    def test(
        self,
        X_test: Any,  # pylint: disable=invalid-name
        y_test: Any,
    ) -> dict[str, float]:
        """
        Evaluate test metrics and record the test step.
        """
        metrics = self.model.evaluate(X_test, y_test)
        self.test_metrics = metrics
        self._record_step("test")
        return metrics

    def persist(
        self,
        *,
        data_interface: Mapping[str, Any] | None = None,
        extra_metrics: Mapping[str, Any] | None = None,
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
            figure_path = self._save_training_history_figure()
            if figure_path is not None:
                artifact_paths["training_history_figure"] = figure_path

        metrics: dict[str, Any] = {}
        if self.validation_metrics is not None:
            metrics["validation"] = self.validation_metrics
        if self.test_metrics is not None:
            metrics["test"] = self.test_metrics
        if extra_metrics is not None:
            duplicate_keys = set(metrics).intersection(extra_metrics)
            if duplicate_keys:
                names = ", ".join(sorted(duplicate_keys))
                raise ValueError(f"extra_metrics duplicates run metrics: {names}")
            metrics.update(extra_metrics)
        if metrics:
            artifact_paths["metrics"] = self.manager.save_metrics(
                metrics,
                metric_units=metric_units,
            )

        create_run_artifact_dirs(self.manager.run_dir)
        try:
            previous_selected_run_id = self.registry.selected(
                self.model.name
            ).run_id
        except KeyError:
            previous_selected_run_id = None
        self.registry.register_run(self.manager.run_dir)
        selected_pointer_changed = (
            self.registry.selected(self.model.name).run_id
            != previous_selected_run_id
        )
        self._record_step("persist")

        if refresh_benchmarks and self.test_metrics is not None:
            refresh_rows(
                self.cfg.paths.benchmarks,
                self.registry,
                self.model.name,
                selection="latest",
            )
            if selected_pointer_changed:
                regenerate_rows(
                    self.cfg.paths.benchmarks,
                    self.registry,
                    selections=("selected",),
                )

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

    def _save_training_history_figure(self) -> Path | None:
        """
        Save a training-history figure when the model exposes a plot method.
        """
        plot_training_history = getattr(self.model, "plot_training_history", None)
        if not callable(plot_training_history):
            return None

        from matplotlib import pyplot as plt
        from matplotlib.figure import Figure

        fig = cast(Figure, plot_training_history())
        try:
            return self.manager.save_figure(fig, "training_history.png")
        finally:
            plt.close(fig)
