"""
Random Forest surrogate models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sparam_surrogate.models.base import SparamModel
from sparam_surrogate.utils.non_neural_modelling_utils import regression_metrics

if TYPE_CHECKING:
    from sparam_surrogate.config.surrogate_config import RandomForestModelConfig

# Number of trees fitted for each Random Forest candidate.
RANDOM_FOREST_N_ESTIMATORS = 256

# Candidate maximum tree depths; None lets trees grow until leaf constraints stop them.
RANDOM_FOREST_MAX_DEPTH_GRID: tuple[int | None, ...] = (None,)

# Candidate minimum sample counts required in each leaf node.
RANDOM_FOREST_MIN_SAMPLES_LEAF_GRID = (2,)

# Seed used to make forest training reproducible.
RANDOM_FOREST_RANDOM_STATE = 42

# Worker count passed to scikit-learn; -1 uses all available cores.
RANDOM_FOREST_N_JOBS = -1


def _build_random_forest_regressor(
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int,
    n_jobs: int,
) -> RandomForestRegressor:
    """
    Return one Random Forest candidate.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )


RandomForestModelT = TypeVar("RandomForestModelT", bound="RandomForestModel")

class RandomForestModel(SparamModel):
    """
    Validation-selected Random Forest baseline for scalar or vector targets.
    """

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = RANDOM_FOREST_N_ESTIMATORS,
        max_depths: Sequence[int | None] = RANDOM_FOREST_MAX_DEPTH_GRID,
        min_samples_leafs: Sequence[int] = RANDOM_FOREST_MIN_SAMPLES_LEAF_GRID,
        random_state: int = RANDOM_FOREST_RANDOM_STATE,
        n_jobs: int = RANDOM_FOREST_N_JOBS,
    ) -> None:
        """
        Store Random Forest candidate grids and runtime controls.
        """
        self.n_estimators = int(n_estimators)
        self.max_depths = tuple(max_depths)
        self.min_samples_leafs = tuple(int(value) for value in min_samples_leafs)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.model: RandomForestRegressor | None = None
        self.validation_results: pd.DataFrame | None = None
        self.best_max_depth: int | None = None
        self.best_min_samples_leaf: int | None = None

    @classmethod
    def from_config(
        cls: type[RandomForestModelT],
        cfg: RandomForestModelConfig,
    ) -> RandomForestModelT:
        """
        Return a Random Forest model initialized from typed configuration.
        """
        return cls(
            n_estimators=cfg.n_estimators,
            max_depths=cfg.max_depths,
            min_samples_leafs=cfg.min_samples_leafs,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )

    def fit(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,  # pylint: disable=invalid-name
        y_val: np.ndarray | None = None,
    ) -> RandomForestModel:
        """
        Fit forest candidates and keep the lowest-validation-MAE model.
        """
        if X_val is None or y_val is None:
            raise ValueError(
                "RandomForestModel requires validation data for hyperparameter "
                "selection."
            )
        if self.n_estimators < 1:
            raise ValueError("RandomForestModel requires at least one tree.")
        if not self.max_depths:
            raise ValueError("RandomForestModel requires at least one max depth.")
        if not self.min_samples_leafs:
            raise ValueError(
                "RandomForestModel requires at least one min_samples_leaf value."
            )
        if any(value < 1 for value in self.min_samples_leafs):
            raise ValueError("min_samples_leaf values must be positive integers.")

        rows: list[dict[str, Any]] = []
        best_model: RandomForestRegressor | None = None
        best_max_depth: int | None = None
        best_min_samples_leaf: int | None = None
        best_mae = np.inf

        for max_depth in self.max_depths:
            for min_samples_leaf in self.min_samples_leafs:
                model = _build_random_forest_regressor(
                    n_estimators=self.n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
                model.fit(X_train, y_train)
                y_val_pred = np.asarray(model.predict(X_val))
                metrics = regression_metrics(y_val, y_val_pred)
                rows.append(
                    {
                        "n_estimators": self.n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        **metrics,
                    }
                )

                if metrics["MAE"] < best_mae:
                    best_mae = metrics["MAE"]
                    best_model = model
                    best_max_depth = max_depth
                    best_min_samples_leaf = min_samples_leaf

        self.model = best_model
        self.best_max_depth = best_max_depth
        self.best_min_samples_leaf = best_min_samples_leaf
        self.validation_results = pd.DataFrame(rows)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Return predictions from the selected Random Forest regressor.
        """
        return np.asarray(self.regressor.predict(X))

    @property
    def regressor(self) -> RandomForestRegressor:
        """
        Return the selected fitted Random Forest regressor.
        """
        if self.model is None:
            raise RuntimeError(f"{self.name} must be fitted before prediction.")
        return self.model
