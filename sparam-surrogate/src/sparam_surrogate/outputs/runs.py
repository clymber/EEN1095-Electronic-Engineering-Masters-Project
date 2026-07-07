"""
Run-directory artifact helpers for fitted surrogate models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, cast

from joblib import dump, load

from sparam_surrogate.models.base import SparamModel

# Filename for full scikit-learn-style wrapper artifacts.
MODEL_JOBLIB = "model.joblib"

# Filename for saved Keras network artifacts.
MODEL_KERAS = "model.keras"

# Filename for non-Keras neural wrapper state.
PREPROCESSORS_JOBLIB = "preprocessors.joblib"

# Required run timestamp form: YYYYMMDDTHHMMSSZ.
TIMESTAMP_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")


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

    def save_model(self, model: SparamModel) -> dict[str, Path]:
        """
        Save one fitted model artifact family into this run directory.
        """
        return save_model_artifact(model, self.run_dir)

    def load_model(self) -> SparamModel:
        """
        Load the model artifact stored in this run directory.
        """
        return load_model_artifact(self.run_dir)


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
            class_path=cls._object_path(type(model)),
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
        if not isinstance(wrapper, SparamModel):
            raise TypeError(
                f"Restored wrapper is not an SparamModel: {self.class_path}"
            )
        return wrapper

    @staticmethod
    def _object_path(obj: type) -> str:
        """
        Return the import path for a top-level class.
        """
        return f"{obj.__module__}.{obj.__qualname__}"


def create_run_dir(
    runs_root: Path | str,
    model_name: str,
    *,
    timestamp: datetime | str | None = None,
) -> Path:
    """
    Create and return a timestamped model-run directory.
    """
    root = Path(runs_root)
    run_id = get_run_id(model_name, timestamp=timestamp)
    run_dir = root / run_id
    root.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"Run directory already exists: {run_dir}") from exc
    return run_dir


def get_run_id(
    model_name: str,
    *,
    timestamp: datetime | str | None = None,
) -> str:
    """
    Return the timestamped run ID for a model name.
    """

    def _format_timestamp(timestamp: datetime | str | None) -> str:
        """
        Return a UTC timestamp formatted as ``YYYYMMDDTHHMMSSZ``.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        if isinstance(timestamp, str):
            if not TIMESTAMP_PATTERN.match(timestamp):
                raise ValueError("timestamp must match YYYYMMDDTHHMMSSZ.")
            return timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


    def _slugify_model_name(model_name: str) -> str:
        """
        Return a stable lowercase slug for a model name.
        """
        slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name.strip().lower()).strip("_")
        if not slug:
            raise ValueError("model_name must contain at least one alphanumeric value.")
        return slug

    return f"{_format_timestamp(timestamp)}_{_slugify_model_name(model_name)}"


def save_model_artifact(model: SparamModel, run_dir: Path | str) -> dict[str, Path]:
    """
    Save a fitted model artifact family into an existing run directory.
    """
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _ensure_no_existing_artifacts(destination)
    _ensure_fitted(model)

    if _is_neural_wrapper(model):
        return _save_neural_artifacts(model, destination)

    model_path = destination / MODEL_JOBLIB
    dump(model, model_path)
    return {"model": model_path}


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
        )
        if path.exists()
    ]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Model artifact already exists: {names}")


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
    if not module_path or not object_name:
        raise ValueError(f"Invalid import path: {import_path}")
    module = import_module(module_path)
    return getattr(module, object_name)
