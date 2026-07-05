"""
Typed, ready-to-use configuration for the project.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def _to_float_tuple(values: Sequence[Any]) -> tuple[float, ...]:
    """
    Resolve configured values as a tuple of floats.
    """
    return tuple(float(value) for value in values)


def _to_int_tuple(values: Sequence[Any]) -> tuple[int, ...]:
    """
    Resolve configured values as a tuple of integers.
    """
    return tuple(int(value) for value in values)


def _to_optional_int_tuple(values: Sequence[Any]) -> tuple[int | None, ...]:
    """
    Resolve configured values as a tuple of optional integers.
    """
    return tuple(None if value is None else int(value) for value in values)


@dataclass(frozen=True)
class RidgeModelConfig:
    """
    Typed configuration for Ridge-family models.
    """

    #: Candidate Ridge regularisation strengths.
    alphas: tuple[float, ...] = (
        0.00001,
        0.00005,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.1,
        1.0,
        10.0,
    )

    @classmethod
    def resolve(cls, model_cfg: Mapping[str, Any]) -> "RidgeModelConfig":
        """
        Resolve a Ridge model configuration block.
        """
        defaults = cls()
        return cls(alphas=_to_float_tuple(model_cfg.get("alphas", defaults.alphas)))


@dataclass(frozen=True)
class PolynomialRidgeModelConfig:
    """
    Typed configuration for the powers-only polynomial Ridge model.
    """

    #: Candidate polynomial degrees.
    degrees: tuple[int, ...] = (3, 4, 5)

    #: Candidate Ridge regularisation strengths.
    alphas: tuple[float, ...] = (
        50.0,
        100.0,
        200.0,
        500.0,
        1000.0,
    )

    @classmethod
    def resolve(
        cls,
        model_cfg: Mapping[str, Any],
    ) -> "PolynomialRidgeModelConfig":
        """
        Resolve a polynomial Ridge model configuration block.
        """
        defaults = cls()
        return cls(
            degrees=_to_int_tuple(model_cfg.get("degrees", defaults.degrees)),
            alphas=_to_float_tuple(model_cfg.get("alphas", defaults.alphas)),
        )


@dataclass(frozen=True)
class RandomForestModelConfig:
    """
    Typed configuration for the Random Forest model.
    """

    n_estimators: int = 256  #: Number of trees per forest candidate.
    max_depths: tuple[int | None, ...] = (None,)  #: Candidate maximum depths.
    min_samples_leafs: tuple[int, ...] = (2,)  #: Candidate leaf sizes.
    random_state: int = 42  #: Seed for reproducible forests.
    n_jobs: int = -1  #: Worker count passed to scikit-learn.

    @classmethod
    def resolve(cls, model_cfg: Mapping[str, Any]) -> "RandomForestModelConfig":
        """
        Resolve a Random Forest model configuration block.
        """
        defaults = cls()
        return cls(
            n_estimators=int(
                model_cfg.get("n_estimators", defaults.n_estimators)
            ),
            max_depths=_to_optional_int_tuple(
                model_cfg.get("max_depths", defaults.max_depths)
            ),
            min_samples_leafs=_to_int_tuple(
                model_cfg.get("min_samples_leafs", defaults.min_samples_leafs)
            ),
            random_state=int(
                model_cfg.get("random_state", defaults.random_state)
            ),
            n_jobs=int(model_cfg.get("n_jobs", defaults.n_jobs)),
        )


@dataclass(frozen=True)
class NeuralMLPModelConfig:
    """
    Typed configuration for the raw-feature neural MLP model.
    """

    batch_size: int = 512  #: Mini-batch size used during training.
    epochs: int = 100  #: Maximum number of training epochs.
    prediction_batch_size: int = 4096  #: Batch size used for prediction.
    learning_rate: float = 0.00003  #: Initial Adam learning rate.
    gradient_clip_norm: float = 0.5  #: Adam gradient clip norm.
    early_stopping_patience: int = 18  #: Epoch patience for early stopping.
    reduce_lr_patience: int = 6  #: Epoch patience before reducing learning rate.
    reduce_lr_factor: float = 0.5  #: Learning-rate reduction multiplier.
    min_learning_rate: float = 0.000001  #: Lower bound for reduced learning rate.
    random_state: int = 128  #: Seed for reproducible neural training.

    @classmethod
    def resolve(cls, model_cfg: Mapping[str, Any]) -> "NeuralMLPModelConfig":
        """
        Resolve a neural MLP model configuration block.
        """
        defaults = cls()
        return cls(
            batch_size=int(
                model_cfg.get("batch_size", defaults.batch_size)
            ),
            epochs=int(
                model_cfg.get("epochs", defaults.epochs)
            ),
            prediction_batch_size=int(
                model_cfg.get("prediction_batch_size", defaults.prediction_batch_size)
            ),
            learning_rate=float(
                model_cfg.get("learning_rate", defaults.learning_rate)
            ),
            gradient_clip_norm=float(
                model_cfg.get("gradient_clip_norm", defaults.gradient_clip_norm)
            ),
            early_stopping_patience=int(
                model_cfg.get(
                    "early_stopping_patience", defaults.early_stopping_patience
                )
            ),
            reduce_lr_patience=int(
                model_cfg.get("reduce_lr_patience", defaults.reduce_lr_patience)
            ),
            reduce_lr_factor=float(
                model_cfg.get("reduce_lr_factor", defaults.reduce_lr_factor)
            ),
            min_learning_rate=float(
                model_cfg.get("min_learning_rate", defaults.min_learning_rate)
            ),
            random_state=int(model_cfg.get("random_state", defaults.random_state)),
        )


@dataclass(frozen=True)
class PolynomialNeuralMLPModelConfig(NeuralMLPModelConfig):
    """
    Typed configuration for the polynomial-feature neural MLP model.
    """

    polynomial_degree: int = 5  #: Powers-only polynomial expansion degree.

    @classmethod
    def resolve(
        cls,
        model_cfg: Mapping[str, Any],
    ) -> "PolynomialNeuralMLPModelConfig":
        """
        Resolve a polynomial neural MLP model configuration block.
        """
        defaults = cls()
        neural_cfg = NeuralMLPModelConfig.resolve(model_cfg)
        return cls(
            polynomial_degree=int(
                model_cfg.get("polynomial_degree", defaults.polynomial_degree)
            ),
            batch_size=neural_cfg.batch_size,
            epochs=neural_cfg.epochs,
            prediction_batch_size=neural_cfg.prediction_batch_size,
            learning_rate=neural_cfg.learning_rate,
            gradient_clip_norm=neural_cfg.gradient_clip_norm,
            early_stopping_patience=neural_cfg.early_stopping_patience,
            reduce_lr_patience=neural_cfg.reduce_lr_patience,
            reduce_lr_factor=neural_cfg.reduce_lr_factor,
            min_learning_rate=neural_cfg.min_learning_rate,
            random_state=neural_cfg.random_state,
        )


@dataclass(frozen=True)
class ModelsConfig:
    """
    Typed configuration for all surrogate model defaults.
    """

    #: Scalar Ridge defaults.
    scalar_ridge: RidgeModelConfig = field(
        default_factory=RidgeModelConfig
    )

    #: Vector Ridge defaults.
    vector_ridge: RidgeModelConfig = field(
        default_factory=RidgeModelConfig
    )

    #: Polynomial Ridge defaults.
    polynomial_ridge: PolynomialRidgeModelConfig = field(
        default_factory=PolynomialRidgeModelConfig
    )

    #: Random Forest defaults.
    random_forest: RandomForestModelConfig = field(
        default_factory=RandomForestModelConfig
    )

    #: Raw-feature neural MLP defaults.
    neural_mlp: NeuralMLPModelConfig = field(
        default_factory=NeuralMLPModelConfig
    )

    #: Polynomial-feature neural MLP defaults.
    polynomial_neural_mlp: PolynomialNeuralMLPModelConfig = field(
        default_factory=PolynomialNeuralMLPModelConfig
    )

    @classmethod
    def resolve(cls, models_cfg: Mapping[str, Any]) -> "ModelsConfig":
        """
        Resolve all configured model blocks.
        """
        return cls(
            scalar_ridge=RidgeModelConfig.resolve(
                models_cfg.get("scalar_ridge", {})
            ),
            vector_ridge=RidgeModelConfig.resolve(
                models_cfg.get("vector_ridge", {})
            ),
            polynomial_ridge=PolynomialRidgeModelConfig.resolve(
                models_cfg.get("polynomial_ridge", {})
            ),
            random_forest=RandomForestModelConfig.resolve(
                models_cfg.get("random_forest", {})
            ),
            neural_mlp=NeuralMLPModelConfig.resolve(models_cfg.get("neural_mlp", {})),
            polynomial_neural_mlp=PolynomialNeuralMLPModelConfig.resolve(
                models_cfg.get("polynomial_neural_mlp", {})
            ),
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
    models: ModelsConfig = field(default_factory=ModelsConfig)  #: Model settings.

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
        _models = ModelsConfig.resolve(_cfg.get("models", {}))

        return cls(
            project=_project,
            paths=_paths,
            dataset=_dataset,
            preprocessing=_preproc,
            models=_models,
        )
