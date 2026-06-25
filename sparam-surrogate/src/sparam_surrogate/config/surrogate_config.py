"""
Typed, ready-to-use configuration for the project.
"""

from dataclasses import dataclass
from pathlib import Path

from .basic_cfg import load_config


@dataclass(frozen=True)
class ProjectConfig:
    """
    Typed configuration for project-level parameters.
    """

    name: str  #: Project name.
    seed: int  #: Random seed for reproducible runs.

    @classmethod
    def resolve(cls, project_cfg: dict) -> "ProjectConfig":
        """
        Resolve project-level configuration.
        """
        return cls(seed=project_cfg["seed"], name=project_cfg["name"])


@dataclass(frozen=True)
class PathsConfig:
    """
    Typed configuration for project paths.
    """

    raw_data: Path  #: Directory containing raw input data.
    processed_data: Path  #: Directory for generated processed data.

    @classmethod
    def resolve(cls, paths_cfg: dict) -> "PathsConfig":
        """
        Resolve relative paths against project root.
        """
        _raw_data = Path(paths_cfg["raw_data"])
        _processed_data = Path(paths_cfg["processed_data"])

        return cls(
            raw_data=_raw_data.resolve(),
            processed_data=_processed_data.resolve(),
        )


@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration for SI-PI dataset-specific parameters.
    """

    name: str  #: SI-PI Dataset name.
    path: Path  #: Resolved path to the dataset directory.
    parameter_csv: Path  #: Resolved path to the dataset parameter CSV.
    nports: int  #: Number of ports in each Touchstone sample.
    ports: tuple[tuple[int, int], ...]  #: One-based receiver/source port pairs.

    @classmethod
    def resolve(cls, dataset_cfg: dict, paths: PathsConfig) -> "DatasetConfig":
        """
        Resolve relative paths against project root.
        """
        _name = dataset_cfg["name"]
        _path = paths.raw_data / _name
        _nports = int(dataset_cfg["nports"])
        if _nports <= 0:
            raise ValueError("dataset.nports must be positive.")

        _ports = cls._resolve_ports(dataset_cfg["ports"])
        for pair in _ports:
            receiver, source = pair
            if receiver < 1 or source < 1:
                raise ValueError(
                    f"Configured port pair {pair} must use one-based port numbers."
                )
            if receiver > _nports or source > _nports:
                raise ValueError(
                    f"Configured port pair {pair} exceeds dataset.nports ({_nports})."
                )

        return cls(
            name=_name,
            path=_path,
            parameter_csv=_path / dataset_cfg["parameter_file"],
            nports=_nports,
            ports=_ports,
        )

    @staticmethod
    def _resolve_ports(ports: list[list[int]]) -> tuple[tuple[int, int], ...]:
        """
        Resolve configured port pairs as exactly two integer ports.
        """
        resolved_ports: list[tuple[int, int]] = []
        for pair in ports:
            if len(pair) != 2:
                raise ValueError("Each dataset port pair must contain two ports.")
            receiver, source = pair
            resolved_ports.append((int(receiver), int(source)))
        return tuple(resolved_ports)


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Typed configuration for data preprocessing.
    """

    processed_csv: Path  #: Resolved path to the cleaned dataset CSV.
    val_fraction: float  #: Fraction of data reserved for validation.
    test_fraction: float  #: Fraction of data reserved for testing.

    @classmethod
    def resolve(cls, preproc: dict, paths: PathsConfig) -> "PreprocessingConfig":
        """
        Resolve relative paths against project root.
        """
        _processed_csv = paths.processed_data / preproc["cleaned_csv"]

        return cls(
            processed_csv=_processed_csv,
            val_fraction=float(preproc["val_fraction"]),
            test_fraction=float(preproc["test_fraction"]),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """
    Typed configuration for model training.
    """

    batch_size: int = 32  #: Number of samples per training batch.
    epochs: int = 100  #: Number of training epochs.

    @classmethod
    def resolve(cls, training_cfg: dict) -> "TrainingConfig":
        """
        Resolve training configuration.
        """
        return cls(
            batch_size=int(training_cfg.get("batch_size", 32)),
            epochs=int(training_cfg.get("epochs", 100)),
        )


@dataclass(frozen=True)
class SurrogateConfig:
    """
    Typed, ready-to-use project configuration.
    """

    project: ProjectConfig  #: Project-level settings.
    paths: PathsConfig  #: Resolved filesystem paths.
    dataset: DatasetConfig  #: Dataset loading settings.
    preprocessing: PreprocessingConfig  #: Data preprocessing settings.
    training: TrainingConfig  #: Model training settings.

    @classmethod
    def from_csv(cls, cfg_csv: Path | str | None = None) -> "SurrogateConfig":
        """
        Create a SurrogateConfig instance from project configuration files.
        """
        _cfg = load_config(cfg_csv)
        _project = ProjectConfig.resolve(_cfg["project"])
        _paths = PathsConfig.resolve(_cfg["paths"])
        _dataset = DatasetConfig.resolve(_cfg["dataset"], _paths)
        _preproc = PreprocessingConfig.resolve(_cfg["preprocessing"], _paths)
        _training = TrainingConfig.resolve(_cfg["training"])

        return cls(
            project=_project,
            paths=_paths,
            dataset=_dataset,
            preprocessing=_preproc,
            training=_training,
        )
