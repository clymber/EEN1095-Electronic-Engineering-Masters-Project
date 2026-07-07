"""
Run-directory artifact helpers for fitted surrogate models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, cast

import pandas as pd
from joblib import dump, load

from sparam_surrogate.models.base import SparamModel
from sparam_surrogate.outputs.naming import get_run_id
from sparam_surrogate.utils.filesystem import ensure_dir
from sparam_surrogate.utils.json_io import json_ready, write_json

# Filename for full scikit-learn-style wrapper artifacts.
MODEL_JOBLIB = "model.joblib"

# Filename for saved Keras network artifacts.
MODEL_KERAS = "model.keras"

# Filename for non-Keras neural wrapper state.
PREPROCESSORS_JOBLIB = "preprocessors.joblib"

# Filename for human-readable model artifact metadata.
METADATA_JSON = "metadata.json"

# Filename for final model-run metrics.
METRICS_JSON = "metrics.json"

# Filename for validation-search or validation-sweep tables.
VALIDATION_RESULTS_CSV = "validation_results.csv"

# Filename for epoch-by-epoch neural training history.
TRAINING_HISTORY_CSV = "training_history.csv"


@dataclass(frozen=True)
class ModelRunArtifactManager:
    """
    Manage model artifact persistence for one immutable run directory.
    """

    runs_root: Path
    run_id: str
    run_dir: Path

    @classmethod
    def create(
        cls,
        runs_root: Path | str,
        model_name: str,
        *,
        timestamp: datetime | str | None = None,
    ) -> ModelRunArtifactManager:
        """
        Create a new manager with a fresh timestamped run directory.
        """
        run_dir = create_run_dir(runs_root, model_name, timestamp=timestamp)
        return cls(
            runs_root=Path(runs_root),
            run_id=run_dir.name,
            run_dir=run_dir,
        )

    def save_model(
        self,
        model: SparamModel,
        *,
        data_interface: Mapping[str, Any] | None = None,
    ) -> dict[str, Path]:
        """
        Save one fitted model artifact family into this run directory.
        """
        return save_model_artifact(
            model,
            self.run_dir,
            data_interface=data_interface,
        )

    def load_model(self) -> SparamModel:
        """
        Load the model artifact stored in this run directory.
        """
        return load_model_artifact(self.run_dir)

    def save_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        metric_units: Mapping[str, str] | None = None,
    ) -> Path:
        """
        Save model-run metrics into this run directory.
        """
        return save_run_metrics(
            self.run_dir,
            metrics,
            metric_units=metric_units,
        )

    def save_validation_results(
        self,
        results: Any | None = None,
        *,
        model: SparamModel | None = None,
    ) -> Path:
        """
        Save validation sweep results into this run directory.
        """
        return save_run_validation_results(self.run_dir, results, model=model)

    def save_training_history(
        self,
        history: Any | None = None,
        *,
        model: SparamModel | None = None,
    ) -> Path:
        """
        Save training history into this run directory.
        """
        return save_run_training_history(self.run_dir, history, model=model)


@dataclass(frozen=True)
class KerasWrapperState:
    """
    Persist non-Keras wrapper state around a saved Keras model.
    """

    class_path: str  # Import path for the wrapper class.
    constructor_params: dict[str, Any]  # Parameters used to rebuild the wrapper.
    state_attrs: dict[str, Any]  # Fitted attributes restored onto the wrapper.

    # Constructor attributes needed to rebuild neural wrappers before weights.
    CONSTRUCTOR_ATTRS: ClassVar[tuple[str, ...]] = (
        "polynomial_degree",
        "batch_size",
        "epochs",
        "prediction_batch_size",
        "learning_rate",
        "gradient_clip_norm",
        "early_stopping_patience",
        "reduce_lr_patience",
        "reduce_lr_factor",
        "min_learning_rate",
        "random_state",
    )

    # Fitted preprocessing attributes needed for neural wrapper prediction.
    STATE_ATTRS: ClassVar[tuple[str, ...]] = (
        "x_scaler",
        "input_scaler",
        "polynomial_features",
        "expanded_feature_scaler",
        "y_scaler",
        "expanded_feature_count_",
    )

    @classmethod
    def from_model(cls, model: SparamModel) -> KerasWrapperState:
        """
        Return wrapper state extracted from a fitted neural model.
        """
        return cls(
            class_path=ModelMetadata.model_class_path(model),
            constructor_params={
                name: getattr(model, name)
                for name in cls.CONSTRUCTOR_ATTRS
                if hasattr(model, name)
            },
            state_attrs={
                name: getattr(model, name)
                for name in cls.STATE_ATTRS
                if hasattr(model, name)
            },
        )

    def save(self, path: Path | str) -> None:
        """
        Save this wrapper state to a joblib artifact.
        """
        dump(self, Path(path))

    @classmethod
    def load(cls, path: Path | str) -> KerasWrapperState:
        """
        Load wrapper state from a joblib artifact.
        """
        state = load(Path(path))
        if not isinstance(state, cls):
            raise TypeError(f"Loaded artifact is not a {cls.__name__}: {path}")
        return state

    def restore(self, keras_model: Any) -> SparamModel:
        """
        Rebuild a wrapper around a loaded Keras model.
        """
        model_class = _import_object(self.class_path)
        wrapper = model_class(**self.constructor_params)
        for name, value in self.state_attrs.items():
            setattr(wrapper, name, value)
        cast(Any, wrapper).model = keras_model
        return cast(SparamModel, wrapper)


@dataclass(frozen=True)
class ModelMetadata:
    """
    Human-readable metadata for one persisted model artifact family.
    """

    run_id: str  # Directory name and stable identifier for this model run.
    model: dict[str, Any]  # Model identity, framework family, and artifact type.
    artifacts: dict[str, str]  # Run-relative artifact filenames by logical role.
    data_interface: Mapping[str, Any] | None = None  # Optional feature/target context.
    selected_hyperparameters: Mapping[str, Any] | None = None  # Fitted choices.
    training_controls: Mapping[str, Any] | None = None  # Neural training settings.

    # Metadata schema version written into every metadata.json file.
    SCHEMA_VERSION: ClassVar[int] = 1

    # Fitted hyperparameter attributes exposed by current model wrappers.
    SELECTED_HYPERPARAMETER_ATTRS: ClassVar[tuple[str, ...]] = (
        "best_alpha",
        "best_degree",
        "best_max_depth",
        "best_min_samples_leaf",
    )

    @classmethod
    def from_model(
        cls,
        model: SparamModel,
        run_dir: Path,
        artifact_paths: Mapping[str, Path],
        *,
        data_interface: Mapping[str, Any] | None,
    ) -> ModelMetadata:
        """
        Build metadata from a saved model and its artifact paths.
        """
        is_neural = _is_neural_wrapper(model)
        selected_hyperparameters = cls._selected_hyperparameters(model)
        training_controls = cls._training_controls(model) if is_neural else {}
        return cls(
            run_id=run_dir.name,
            model={
                "artifact_type": (
                    "keras_with_wrapper_state" if is_neural else "joblib_wrapper"
                ),
                "class_path": cls.model_class_path(model),
                "family": "keras" if is_neural else "sklearn",
                "label": model.model_name(),
                "name": model.name,
            },
            artifacts={
                name: path.relative_to(run_dir).as_posix()
                for name, path in artifact_paths.items()
            },
            data_interface=data_interface,
            selected_hyperparameters=selected_hyperparameters or None,
            training_controls=training_controls or None,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Return this metadata as a JSON-ready dictionary.
        """
        metadata: dict[str, Any] = {
            "artifacts": self.artifacts,
            "model": self.model,
            "run_id": self.run_id,
            "schema_version": self.SCHEMA_VERSION,
        }
        if self.data_interface is not None:
            metadata["data_interface"] = dict(self.data_interface)
        if self.selected_hyperparameters is not None:
            metadata["selected_hyperparameters"] = dict(
                self.selected_hyperparameters
            )
        if self.training_controls is not None:
            metadata["training_controls"] = dict(self.training_controls)
        return json_ready(metadata)

    def save(self, path: Path | str) -> None:
        """
        Save this metadata as stable, human-readable JSON.
        """
        write_json(path, self.to_dict())

    @staticmethod
    def model_class_path(model: SparamModel) -> str:
        """
        Return the import path for a model wrapper class.
        """
        model_type = type(model)
        return f"{model_type.__module__}.{model_type.__qualname__}"

    @classmethod
    def _selected_hyperparameters(cls, model: SparamModel) -> dict[str, Any]:
        """
        Return selected fitted hyperparameters when wrappers expose them.
        """
        return {
            name: getattr(model, name)
            for name in cls.SELECTED_HYPERPARAMETER_ATTRS
            if hasattr(model, name) and getattr(model, name) is not None
        }

    @staticmethod
    def _training_controls(model: SparamModel) -> dict[str, Any]:
        """
        Return neural training controls stored on the wrapper.
        """
        return {
            name: getattr(model, name)
            for name in KerasWrapperState.CONSTRUCTOR_ATTRS
            if hasattr(model, name)
        }


@dataclass(frozen=True)
class RunMetrics:
    """
    JSON metrics artifact for one persisted model run.
    """

    metrics: Mapping[str, Any]  # Split or aggregate metrics to persist.
    metric_units: Mapping[str, str] | None = None  # Optional units by metric key.

    # Metrics schema version written into every metrics.json file.
    SCHEMA_VERSION: ClassVar[int] = 1

    def to_dict(self) -> dict[str, Any]:
        """
        Return this metrics artifact as a JSON-ready dictionary.
        """
        data: dict[str, Any] = {
            "metrics": dict(self.metrics),
            "schema_version": self.SCHEMA_VERSION,
        }
        if self.metric_units is not None:
            data["metric_units"] = dict(self.metric_units)
        return json_ready(data)

    def save(self, path: Path | str) -> None:
        """
        Save metrics as stable, human-readable JSON.
        """
        write_json(path, self.to_dict())


@dataclass(frozen=True)
class ValidationResults:
    """
    Tabular validation sweep artifact for one persisted model run.
    """

    table: pd.DataFrame

    @classmethod
    def from_model(cls, model: SparamModel) -> ValidationResults:
        """
        Build validation results from a fitted model wrapper.
        """
        results = getattr(model, "validation_results", None)
        if results is None:
            raise ValueError(f"{model.name} has no validation_results to save.")
        return cls.from_value(results)

    @classmethod
    def from_value(cls, value: Any) -> ValidationResults:
        """
        Build validation results from a dataframe-like value.
        """
        table = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
        if table.empty:
            raise ValueError("validation_results must contain at least one row.")
        return cls(table=table)

    def save(self, path: Path | str) -> None:
        """
        Save validation results as CSV without the dataframe index.
        """
        self.table.to_csv(path, index=False)


@dataclass(frozen=True)
class TrainingHistory:
    """
    Tabular training history artifact for one persisted model run.
    """

    table: pd.DataFrame

    @classmethod
    def from_model(cls, model: SparamModel) -> TrainingHistory:
        """
        Build training history from a fitted model wrapper.
        """
        history = getattr(model, "history", None)
        if history is None:
            raise ValueError(f"{model.name} has no training history to save.")
        return cls.from_value(history)

    @classmethod
    def from_value(cls, value: Any) -> TrainingHistory:
        """
        Build training history from a Keras History, mapping, or dataframe.
        """
        if isinstance(value, pd.DataFrame):
            table = value.copy()
        elif isinstance(value, Mapping):
            table = pd.DataFrame(value)
        elif hasattr(value, "history"):
            table = pd.DataFrame(cast(Any, value).history)
        else:
            table = pd.DataFrame(value)

        if table.empty:
            raise ValueError("training_history must contain at least one row.")
        if "epoch" not in table.columns:
            table.insert(0, "epoch", range(1, len(table) + 1))
        return cls(table=table)

    def save(self, path: Path | str) -> None:
        """
        Save training history as CSV without the dataframe index.
        """
        self.table.to_csv(path, index=False)


def create_run_dir(
    runs_root: Path | str,
    model_name: str,
    *,
    timestamp: datetime | str | None = None,
) -> Path:
    """
    Create and return a timestamped model-run directory.
    """
    root = ensure_dir(runs_root)
    run_dir = root / get_run_id(model_name, timestamp=timestamp)
    run_dir.mkdir()
    return run_dir


def save_model_artifact(
    model: SparamModel,
    run_dir: Path | str,
    *,
    data_interface: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """
    Save a fitted model artifact family into an existing run directory.
    """
    destination = ensure_dir(run_dir)
    _ensure_no_existing_artifacts(destination)
    _ensure_fitted(model)

    if _is_neural_wrapper(model):
        artifact_paths = _save_neural_artifacts(model, destination)
    else:
        model_path = destination / MODEL_JOBLIB
        dump(model, model_path)
        artifact_paths = {"model": model_path}

    metadata_path = destination / METADATA_JSON
    metadata = ModelMetadata.from_model(
        model,
        destination,
        artifact_paths,
        data_interface=data_interface,
    )
    metadata.save(metadata_path)
    return {"metadata": metadata_path, **artifact_paths}


def save_run_metrics(
    run_dir: Path | str,
    metrics: Mapping[str, Any],
    *,
    metric_units: Mapping[str, str] | None = None,
) -> Path:
    """
    Save run metrics into an existing run directory.
    """
    destination = ensure_dir(run_dir)
    path = destination / METRICS_JSON
    _ensure_missing(path)
    RunMetrics(metrics=metrics, metric_units=metric_units).save(path)
    return path


def save_run_validation_results(
    run_dir: Path | str,
    results: Any | None = None,
    *,
    model: SparamModel | None = None,
) -> Path:
    """
    Save validation sweep results into an existing run directory.
    """
    destination = ensure_dir(run_dir)
    path = destination / VALIDATION_RESULTS_CSV
    _ensure_missing(path)
    artifact = (
        ValidationResults.from_model(model)
        if model is not None
        else ValidationResults.from_value(results)
    )
    artifact.save(path)
    return path


def save_run_training_history(
    run_dir: Path | str,
    history: Any | None = None,
    *,
    model: SparamModel | None = None,
) -> Path:
    """
    Save training history into an existing run directory.
    """
    destination = ensure_dir(run_dir)
    path = destination / TRAINING_HISTORY_CSV
    _ensure_missing(path)
    artifact = (
        TrainingHistory.from_model(model)
        if model is not None
        else TrainingHistory.from_value(history)
    )
    artifact.save(path)
    return path


def load_model_artifact(run_dir: Path | str) -> SparamModel:
    """
    Load the model artifact family stored in a run directory.
    """
    source = Path(run_dir)
    joblib_path = source / MODEL_JOBLIB
    keras_path = source / MODEL_KERAS
    preprocessors_path = source / PREPROCESSORS_JOBLIB

    has_joblib = joblib_path.is_file()
    has_keras_family = keras_path.is_file() or preprocessors_path.is_file()
    if has_joblib and has_keras_family:
        raise RuntimeError(f"Run directory has multiple model artifacts: {source}")
    if has_joblib:
        loaded = load(joblib_path)
        if not isinstance(loaded, SparamModel):
            raise TypeError(f"Loaded artifact is not an SparamModel: {joblib_path}")
        return loaded
    if keras_path.is_file() and preprocessors_path.is_file():
        return _load_neural_artifacts(keras_path, preprocessors_path)
    if has_keras_family:
        raise FileNotFoundError(
            f"Incomplete Keras artifact family in run directory: {source}"
        )
    raise FileNotFoundError(f"No model artifact found in run directory: {source}")


def _ensure_no_existing_artifacts(run_dir: Path) -> None:
    """
    Fail if model artifact files already exist in a run directory.
    """
    existing = [
        path
        for path in (
            run_dir / MODEL_JOBLIB,
            run_dir / MODEL_KERAS,
            run_dir / PREPROCESSORS_JOBLIB,
            run_dir / METADATA_JSON,
        )
        if path.exists()
    ]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Model artifact already exists: {names}")


def _ensure_missing(path: Path) -> None:
    """
    Fail clearly when an artifact path already exists.
    """
    if path.exists():
        raise FileExistsError(f"Run artifact already exists: {path.name}")


def _ensure_fitted(model: SparamModel) -> None:
    """
    Fail clearly when a wrapper has no fitted estimator/model attached.
    """
    if hasattr(model, "model") and cast(Any, model).model is None:
        raise RuntimeError(f"{model.name} must be fitted before saving.")


def _is_neural_wrapper(model: SparamModel) -> bool:
    """
    Return whether a model wrapper exposes a Keras model property.
    """
    return any(
        isinstance(cls.__dict__.get("keras_model"), property)
        for cls in type(model).mro()
    )


def _save_neural_artifacts(model: SparamModel, run_dir: Path) -> dict[str, Path]:
    """
    Save Keras model and non-Keras wrapper state artifacts.
    """
    keras_model = cast(Any, model).keras_model
    model_path = run_dir / MODEL_KERAS

    preprocessors_path = run_dir / PREPROCESSORS_JOBLIB
    keras_state = KerasWrapperState.from_model(model)

    keras_model.save(model_path)
    keras_state.save(preprocessors_path)
    return {"model": model_path, "preprocessors": preprocessors_path}


def _load_neural_artifacts(
    model_path: Path,
    preprocessors_path: Path,
) -> SparamModel:
    """
    Load a neural wrapper from Keras and joblib artifacts.
    """
    keras = import_module("keras")
    keras_model = keras.models.load_model(model_path)
    return KerasWrapperState.load(preprocessors_path).restore(keras_model)


def _import_object(import_path: str) -> type:
    """
    Import and return an object from a dotted import path.
    """
    module_path, _, object_name = import_path.rpartition(".")
    module = import_module(module_path)
    return getattr(module, object_name)
