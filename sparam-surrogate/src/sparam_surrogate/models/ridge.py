"""
Ridge-regression surrogate models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sparam_surrogate.models.base import SparamModel
from sparam_surrogate.utils.non_neural_modelling_utils import regression_metrics

if TYPE_CHECKING:
    from sparam_surrogate.config.surrogate_config import RidgeModelConfig

RIDGE_ALPHA_GRID = (
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

RidgeModelT = TypeVar("RidgeModelT", bound="RidgeModel")


def _build_ridge_pipeline(alpha: float) -> Pipeline:
    """
    Return the scaler-plus-Ridge pipeline used by each alpha candidate.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


class RidgeModel(SparamModel):
    """
    Shared validation-sweep implementation for Ridge surrogate models.
    """

    name = "ridge"

    def __init__(self, alphas: Sequence[float] = RIDGE_ALPHA_GRID) -> None:
        self.alphas = tuple(alphas)
        self.model: Pipeline | None = None
        self.validation_results: pd.DataFrame | None = None
        self.best_alpha: float | None = None

    @classmethod
    def from_config(cls: type[RidgeModelT], cfg: RidgeModelConfig) -> RidgeModelT:
        """
        Return a Ridge model initialized from typed model configuration.
        """
        return cls(alphas=cfg.alphas)

    def fit(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,  # pylint: disable=invalid-name
        y_val: np.ndarray | None = None,
    ) -> RidgeModel:
        """
        Fit Ridge candidates and keep the lowest-validation-MAE model.
        """
        if X_val is None or y_val is None:
            raise ValueError("RidgeModel requires validation data for alpha selection.")
        if not self.alphas:
            raise ValueError("RidgeModel requires at least one alpha candidate.")

        rows: list[dict[str, float]] = []
        best_model: Pipeline | None = None
        best_alpha: float | None = None
        best_mae = np.inf

        for alpha in self.alphas:
            model = _build_ridge_pipeline(alpha)
            model.fit(X_train, y_train)
            y_val_pred = np.asarray(model.predict(X_val))
            metrics = regression_metrics(y_val, y_val_pred)
            rows.append({"alpha": float(alpha), **metrics})

            if metrics["MAE"] < best_mae:
                best_mae = metrics["MAE"]
                best_model = model
                best_alpha = float(alpha)

        if best_model is None or best_alpha is None:
            raise RuntimeError("No Ridge model was fitted.")

        self.model = best_model
        self.best_alpha = best_alpha
        self.validation_results = pd.DataFrame(rows)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Return predictions from the selected Ridge pipeline.
        """
        return np.asarray(self.pipeline.predict(X))

    @property
    def pipeline(self) -> Pipeline:
        """
        Return the selected fitted pipeline.
        """
        if self.model is None:
            raise RuntimeError(f"{self.name} must be fitted before prediction.")
        return self.model


class ScalarRidgeModel(RidgeModel):
    """
    Ridge baseline for one-dimensional insertion-loss targets.
    """

    name = "scalar_ridge"


class VectorRidgeModel(RidgeModel):
    """
    Ridge baseline for multi-output insertion-loss targets.
    """

    name = "vector_ridge"
