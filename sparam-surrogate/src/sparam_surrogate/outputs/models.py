"""
Model registry helpers for latest, selected, and history pointer files.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

from sparam_surrogate.config.paths import relative_to_project_root
from sparam_surrogate.models.base import SparamModel
from sparam_surrogate.outputs.naming import (
    created_at_from_run_id,
    slugify_model_name,
)
from sparam_surrogate.outputs.runs import METADATA_JSON, load_model_artifact
from sparam_surrogate.utils.json_io import read_json, write_json

# Registry file containing newest known run pointers by model name.
LATEST_JSON = "latest.json"

# Registry file containing explicitly selected run pointers by model name.
SELECTED_JSON = "selected.json"

# Registry file containing known persisted run history.
REGISTRY_JSON = "registry.json"

# Future metrics artifact location recorded before metrics are implemented.
METRICS_JSON = "metrics.json"


@dataclass(frozen=True)
class ModelRegistryEntry:
    """
    JSON pointer entry for one persisted model run.
    """

    run_id: str  # Immutable run directory name.
    model_name: str  # Stable model slug, such as ``scalar_ridge``.
    model_family: str  # Framework family, such as ``sklearn`` or ``keras``.
    run_path: str  # Project-relative path to the run directory.
    artifact_path: str  # Project-relative path to the primary model artifact.
    metadata_path: str  # Project-relative path to ``metadata.json``.
    metrics_path: str  # Project-relative path reserved for ``metrics.json``.
    created_at: str | None  # ISO timestamp derived from the run ID.
    dataset_name: str | None = None  # Optional dataset name from metadata.
    artifact_type: str | None = None  # Model artifact storage strategy.
    model_label: str | None = None  # Human-readable model label.

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path | str,
        *,
        project_root: Path,
    ) -> ModelRegistryEntry:
        """
        Build a registry entry from a saved run directory.
        """
        run_path = Path(run_dir).resolve()
        if not run_path.is_dir():
            raise NotADirectoryError(f"Run directory does not exist: {run_path}")

        metadata_path = run_path / METADATA_JSON
        metadata = read_json(metadata_path)
        model_info = _require_mapping(metadata, "model", metadata_path)
        artifacts = _require_mapping(metadata, "artifacts", metadata_path)
        artifact_name = artifacts.get("model")
        if not isinstance(artifact_name, str):
            raise KeyError(f"metadata artifacts must include a model entry: {run_path}")

        artifact_path = run_path / artifact_name
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Model artifact does not exist: {artifact_path}")

        data_interface = metadata.get("data_interface")
        dataset_name = None
        if isinstance(data_interface, Mapping):
            dataset_name = data_interface.get("dataset_name")

        project_relative = partial(relative_to_project_root, project_root=project_root)

        return cls(
            run_id=run_path.name,
            model_name=str(model_info["name"]),
            model_family=str(model_info["family"]),
            run_path=project_relative(run_path),
            artifact_path=project_relative(artifact_path),
            metadata_path=project_relative(metadata_path),
            metrics_path=project_relative(run_path / METRICS_JSON),
            created_at=created_at_from_run_id(run_path.name),
            dataset_name=_optional_str(dataset_name),
            artifact_type=_optional_str(model_info.get("artifact_type")),
            model_label=_optional_str(model_info.get("label")),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelRegistryEntry:
        """
        Build a registry entry from a JSON dictionary.
        """

        def optional_str(key: str) -> str | None:
            """
            Return an optional JSON field as a string.
            """
            return _optional_str(data.get(key))

        return cls(
            run_id=str(data["run_id"]),
            model_name=str(data["model_name"]),
            model_family=str(data["model_family"]),
            run_path=str(data["run_path"]),
            artifact_path=str(data["artifact_path"]),
            metadata_path=str(data["metadata_path"]),
            metrics_path=str(data["metrics_path"]),
            created_at=optional_str("created_at"),
            dataset_name=optional_str("dataset_name"),
            artifact_type=optional_str("artifact_type"),
            model_label=optional_str("model_label"),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Return this entry as a JSON-ready dictionary.
        """
        return {
            "artifact_path": self.artifact_path,
            "artifact_type": self.artifact_type,
            "created_at": self.created_at,
            "dataset_name": self.dataset_name,
            "metadata_path": self.metadata_path,
            "metrics_path": self.metrics_path,
            "model_family": self.model_family,
            "model_label": self.model_label,
            "model_name": self.model_name,
            "run_id": self.run_id,
            "run_path": self.run_path,
        }


class ModelRegistry:
    """
    Manage mutable JSON pointers to immutable model-run directories.
    """

    # Schema version for the registry JSON files.
    SCHEMA_VERSION: ClassVar[int] = 1

    models_root: Path
    project_root: Path

    def __init__(
        self,
        models_root: Path | str,
        *,
        project_root: Path | str | None = None,
    ) -> None:
        """
        Create a registry rooted at an ``outputs/models`` directory.
        """
        models_root = Path(models_root).resolve()
        if project_root is None:
            project_root = _default_project_root(models_root)
        self.models_root = models_root
        self.project_root = Path(project_root).resolve()

    def register_run(self, run_dir: Path | str) -> ModelRegistryEntry:
        """
        Register a saved run and update latest, selected, and history files.
        """
        self.models_root.mkdir(parents=True, exist_ok=True)
        entry = ModelRegistryEntry.from_run_dir(
            run_dir,
            project_root=self.project_root,
        )
        self._validate_entry(entry)
        self._upsert_history(entry)
        self._update_latest(entry)
        self._initialize_selected(entry)
        return entry

    def latest(self, model_name: str) -> ModelRegistryEntry:
        """
        Return the latest registered run entry for a model.
        """
        return self._entry_from_model_index(self.latest_path, model_name)

    def selected(self, model_name: str) -> ModelRegistryEntry:
        """
        Return the selected run entry for a model.
        """
        return self._entry_from_model_index(self.selected_path, model_name)

    def promote(self, model_name: str, run_id: str) -> ModelRegistryEntry:
        """
        Promote a registered run to the selected pointer for a model.
        """
        model_key = slugify_model_name(model_name)
        history = self._read_history()
        for entry_data in history["runs"]:
            entry = ModelRegistryEntry.from_dict(entry_data)
            if entry.model_name == model_key and entry.run_id == run_id:
                self._set_model_index_entry(self.selected_path, entry)
                return entry
        raise KeyError(f"No registered run for {model_key}: {run_id}")

    def load(self, entry: ModelRegistryEntry) -> SparamModel:
        """
        Load the model artifact pointed to by a registry entry.
        """
        self._validate_entry(entry)
        return load_model_artifact(self.resolve_path(entry.run_path))

    def resolve_path(self, path: Path | str) -> Path:
        """
        Resolve a registry path against the project root.
        """
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.project_root / candidate

    @property
    def latest_path(self) -> Path:
        """
        Return the path to ``latest.json``.
        """
        return self.models_root / LATEST_JSON

    @property
    def selected_path(self) -> Path:
        """
        Return the path to ``selected.json``.
        """
        return self.models_root / SELECTED_JSON

    @property
    def registry_path(self) -> Path:
        """
        Return the path to ``registry.json``.
        """
        return self.models_root / REGISTRY_JSON

    def _entry_from_model_index(
        self,
        path: Path,
        model_name: str,
    ) -> ModelRegistryEntry:
        """
        Return one model entry from a latest or selected index file.
        """
        model_key = slugify_model_name(model_name)
        index = self._read_model_index(path)
        models = _require_mapping(index, "models", path)
        if model_key not in models:
            raise KeyError(f"No registry entry for model: {model_key}")
        entry_data = models[model_key]
        if not isinstance(entry_data, Mapping):
            raise TypeError(f"Registry entry is not an object: {model_key}")
        entry = ModelRegistryEntry.from_dict(entry_data)
        self._validate_entry(entry)
        return entry

    def _set_model_index_entry(
        self,
        path: Path,
        entry: ModelRegistryEntry,
    ) -> None:
        """
        Set one model entry in a latest or selected index file.
        """
        index = self._read_model_index(path)
        models = _require_mutable_mapping(index, "models", path)
        models[entry.model_name] = entry.to_dict()
        write_json(path, index)

    def _update_latest(self, entry: ModelRegistryEntry) -> None:
        """
        Update latest.json when an entry is newer than the current pointer.
        """
        index = self._read_model_index(self.latest_path)
        models = _require_mutable_mapping(index, "models", self.latest_path)
        current = models.get(entry.model_name)
        if not isinstance(current, Mapping):
            models[entry.model_name] = entry.to_dict()
        else:
            current_entry = ModelRegistryEntry.from_dict(current)
            if _is_entry_newer_or_equal(entry, current_entry):
                models[entry.model_name] = entry.to_dict()
        write_json(self.latest_path, index)

    def _initialize_selected(self, entry: ModelRegistryEntry) -> None:
        """
        Initialize selected.json for models that do not have a selected run.
        """
        index = self._read_model_index(self.selected_path)
        models = _require_mutable_mapping(index, "models", self.selected_path)
        if entry.model_name not in models:
            models[entry.model_name] = entry.to_dict()
        write_json(self.selected_path, index)

    def _upsert_history(self, entry: ModelRegistryEntry) -> None:
        """
        Update or insert one run entry in registry.json history.
        """
        history = self._read_history()
        runs = history["runs"]
        if not isinstance(runs, list):
            raise TypeError(f"registry runs must be a list: {self.registry_path}")

        replacement = entry.to_dict()
        for index, entry_data in enumerate(runs):
            if not isinstance(entry_data, Mapping):
                raise TypeError("registry run entries must be JSON objects.")
            existing = ModelRegistryEntry.from_dict(entry_data)
            same_model = existing.model_name == entry.model_name
            same_run = existing.run_id == entry.run_id
            if same_model and same_run:
                runs[index] = replacement
                break
        else:
            runs.append(replacement)
        write_json(self.registry_path, history)

    def _read_model_index(self, path: Path) -> dict[str, Any]:
        """
        Read a latest or selected index, returning an empty default if missing.
        """
        if not path.exists():
            return {"models": {}, "schema_version": self.SCHEMA_VERSION}
        return read_json(path)

    def _read_history(self) -> dict[str, Any]:
        """
        Read registry history, returning an empty default if missing.
        """
        if not self.registry_path.exists():
            return {"runs": [], "schema_version": self.SCHEMA_VERSION}
        return read_json(self.registry_path)

    def _validate_entry(self, entry: ModelRegistryEntry) -> None:
        """
        Fail if a registry entry points to missing required artifacts.
        """
        run_path = self.resolve_path(entry.run_path)
        artifact_path = self.resolve_path(entry.artifact_path)
        metadata_path = self.resolve_path(entry.metadata_path)
        if not run_path.is_dir():
            raise FileNotFoundError(f"Registry run path does not exist: {run_path}")
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Registry artifact path does not exist: {artifact_path}"
            )
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Registry metadata path does not exist: {metadata_path}"
            )


def _default_project_root(models_root: Path) -> Path:
    """
    Infer the project root from an ``outputs/models`` path when possible.
    """
    if models_root.name == "models" and models_root.parent.name == "outputs":
        return models_root.parent.parent
    return models_root.parent


def _optional_str(value: Any) -> str | None:
    """
    Return an optional value as a string without converting None.
    """
    return str(value) if value is not None else None


def _is_entry_newer_or_equal(
    candidate: ModelRegistryEntry,
    current: ModelRegistryEntry,
) -> bool:
    """
    Return whether a candidate entry should replace the current latest entry.
    """
    if candidate.created_at is None or current.created_at is None:
        return candidate.run_id >= current.run_id
    return candidate.created_at >= current.created_at


def _require_mapping(
    data: Mapping[str, Any],
    key: str,
    source: Path,
) -> Mapping[str, Any]:
    """
    Return a nested mapping from JSON data, or fail clearly.
    """
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected JSON object at {key!r} in {source}")
    return value


def _require_mutable_mapping(
    data: Mapping[str, Any],
    key: str,
    source: Path,
) -> MutableMapping[str, Any]:
    """
    Return a nested mutable JSON object, or fail clearly.
    """
    value = data.get(key)
    if not isinstance(value, MutableMapping):
        raise TypeError(f"Expected mutable JSON object at {key!r} in {source}")
    return value
