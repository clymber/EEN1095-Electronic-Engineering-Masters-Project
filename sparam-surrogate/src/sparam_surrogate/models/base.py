"""
Base interfaces for S-parameter surrogate models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sparam_surrogate.utils.non_neural_modelling_utils import regression_metrics


class SparamModel(ABC):
    """Common interface for S-parameter surrogate models."""

    name = "sparam_model"

    def model_name(self) -> str:
        """Return the human-readable model name used in reports and plots."""
        return str(self.name).replace("_", " ").title()

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,  # pylint: disable=invalid-name
        y_val: np.ndarray | None = None,
    ) -> SparamModel:
        """Fit the model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """Return model predictions."""

    def evaluate(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
        y: np.ndarray,
    ) -> dict[str, float]:
        """Return common regression metrics for this model."""
        return regression_metrics(y, self.predict(X))
