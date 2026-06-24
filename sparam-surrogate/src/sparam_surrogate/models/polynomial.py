"""
Powers-only polynomial surrogate models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from sparam_surrogate.models.base import SparamModel
from sparam_surrogate.utils.non_neural_modelling_utils import regression_metrics

POLYNOMIAL_DEGREE_GRID = (3, 4, 5)
POLYNOMIAL_ALPHA_GRID = (
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
)


class PowersOnlyPolynomialFeatures(BaseEstimator, TransformerMixin):
    """
    Expand each feature independently into powers up to ``degree``.
    """

    def __init__(self, degree: int = 3) -> None:
        self.degree = int(degree)

    def fit(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
        y: np.ndarray | None = None,
    ) -> PowersOnlyPolynomialFeatures:
        """
        Record feature widths for a later powers-only transform.
        """
        del y
        if self.degree < 1:
            raise ValueError("degree must be a positive integer.")
        X_checked = check_array(X)
        self.n_features_in_ = X_checked.shape[1]
        self.n_output_features_ = self.n_features_in_ * self.degree
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Return ``[X, X**2, ..., X**degree]`` with no cross terms.
        """
        check_is_fitted(self, ("n_features_in_", "n_output_features_"))
        X_checked = check_array(X)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features from the fitted data."
            )

        return np.concatenate(
            [X_checked**power for power in range(1, self.degree + 1)],
            axis=1,
        )


def _build_polynomial_pipeline(degree: int, alpha: float) -> Pipeline:
    """
    Return the powers-only polynomial Ridge pipeline.
    """
    return Pipeline(
        [
            ("input_scaler", StandardScaler()),
            ("polynomial", PowersOnlyPolynomialFeatures(degree=degree)),
            ("feature_scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


class PolynomialModel(SparamModel):
    """
    Vector-output powers-only polynomial Ridge model.
    """

    name = "polynomial_ridge"

    def __init__(
        self,
        degrees: Sequence[int] = POLYNOMIAL_DEGREE_GRID,
        alphas: Sequence[float] = POLYNOMIAL_ALPHA_GRID,
    ) -> None:
        self.degrees = tuple(degrees)
        self.alphas = tuple(alphas)
        self.model: Pipeline | None = None
        self.validation_results: pd.DataFrame | None = None
        self.best_degree: int | None = None
        self.best_alpha: float | None = None

    def fit(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,  # pylint: disable=invalid-name
        y_val: np.ndarray | None = None,
    ) -> PolynomialModel:
        """
        Fit polynomial candidates and keep the lowest-validation-MAE model.
        """
        if X_val is None or y_val is None:
            raise ValueError(
                "PolynomialModel requires validation data for degree/alpha selection."
            )
        if not self.degrees:
            raise ValueError("PolynomialModel requires at least one degree candidate.")
        if not self.alphas:
            raise ValueError("PolynomialModel requires at least one alpha candidate.")

        y_train_array = np.asarray(y_train)
        y_val_array = np.asarray(y_val)
        if y_train_array.ndim != 2 or y_val_array.ndim != 2:
            raise ValueError("PolynomialModel expects two-dimensional vector targets.")

        rows: list[dict[str, Any]] = []
        best_model: Pipeline | None = None
        best_degree: int | None = None
        best_alpha: float | None = None
        best_mae = np.inf

        for degree in self.degrees:
            for alpha in self.alphas:
                model = _build_polynomial_pipeline(degree, float(alpha))
                model.fit(X_train, y_train_array)
                y_val_pred = np.asarray(model.predict(X_val))
                metrics = regression_metrics(y_val_array, y_val_pred)
                rows.append({"degree": int(degree), "alpha": float(alpha), **metrics})

                if metrics["MAE"] < best_mae:
                    best_mae = metrics["MAE"]
                    best_model = model
                    best_degree = int(degree)
                    best_alpha = float(alpha)

        if best_model is None or best_degree is None or best_alpha is None:
            raise RuntimeError("No polynomial model was fitted.")

        self.model = best_model
        self.best_degree = best_degree
        self.best_alpha = best_alpha
        self.validation_results = pd.DataFrame(rows)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Return predictions from the selected polynomial pipeline.
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
