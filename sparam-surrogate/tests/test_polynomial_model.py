"""
Tests for powers-only polynomial surrogate models.
"""

import numpy as np
import pytest

from sparam_surrogate.config.surrogate_config import PolynomialRidgeModelConfig
from sparam_surrogate.models import (
    POLYNOMIAL_ALPHA_GRID,
    POLYNOMIAL_DEGREE_GRID,
    PolynomialModel,
    PowersOnlyPolynomialFeatures,
)


def _features() -> tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(  # pylint: disable=invalid-name
        [
            [-2.0, -1.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )
    X_val = np.asarray(  # pylint: disable=invalid-name
        [
            [-1.5, -0.5],
            [0.5, 1.5],
            [2.5, 3.5],
        ]
    )
    return X_train, X_val


def _vector_target(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    return np.column_stack(
        (
            X[:, 0] ** 2 + 0.5 * X[:, 1],
            -0.25 * X[:, 0] + X[:, 1] ** 3,
        )
    )


class TestPowersOnlyPolynomialFeatures:
    """
    Unit tests for powers-only feature expansion.
    """

    def test_transform_concatenates_each_power(self) -> None:
        """
        Degree three expansion returns X, X squared, then X cubed.
        """
        transformer = PowersOnlyPolynomialFeatures(degree=3)
        features = np.asarray([[2.0, -3.0]])

        expanded = transformer.fit_transform(features)

        np.testing.assert_allclose(
            expanded,
            np.asarray([[2.0, -3.0, 4.0, 9.0, 8.0, -27.0]]),
        )
        assert transformer.n_output_features_ == 6

    def test_rejects_non_positive_degree(self) -> None:
        """
        Degree must be a positive integer.
        """
        transformer = PowersOnlyPolynomialFeatures(degree=0)

        with pytest.raises(ValueError, match="positive integer"):
            transformer.fit(np.asarray([[1.0, 2.0]]))


class TestPolynomialModel:
    """
    Unit tests for vector polynomial surrogate models.
    """

    def test_default_grid_sweeps_degrees_and_regularisation(self) -> None:
        """
        The default polynomial baseline keeps a clear degree-3/4/5 narrative.
        """
        model = PolynomialModel()

        assert POLYNOMIAL_DEGREE_GRID == (3, 4, 5)
        assert POLYNOMIAL_ALPHA_GRID == (
            50.0,
            100.0,
            200.0,
            500.0,
            1000.0,
        )
        assert model.degrees == POLYNOMIAL_DEGREE_GRID
        assert model.alphas == POLYNOMIAL_ALPHA_GRID

    def test_from_config_uses_configured_candidate_grid(self) -> None:
        """
        Model config objects initialize polynomial Ridge hyperparameters.
        """
        cfg = PolynomialRidgeModelConfig(
            degrees=(2, 4),
            alphas=(0.01, 0.2),
        )

        model = PolynomialModel.from_config(cfg)

        assert model.degrees == cfg.degrees
        assert model.alphas == cfg.alphas

    def test_fit_records_validation_results_and_predicts_vector_values(self) -> None:
        """
        Fitting sweeps degree-alpha candidates and returns vector predictions.
        """
        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)
        degrees = (2, 3)
        alphas = (0.001, 0.1)

        model = PolynomialModel(degrees=degrees, alphas=alphas)
        result = model.fit(X_train, y_train, X_val, y_val)

        assert result is model
        assert model.best_degree in degrees
        assert model.best_alpha in alphas
        assert model.validation_results is not None
        assert len(model.validation_results) == len(degrees) * len(alphas)
        assert model.pipeline.named_steps["polynomial"].n_output_features_ == {
            2: 4,
            3: 6,
        }[model.best_degree]

        prediction = model.predict(X_val)

        assert prediction.shape == y_val.shape
        assert np.isfinite(prediction).all()

    def test_predict_before_fit_raises_clear_error(self) -> None:
        """
        Unfitted models fail before reaching scikit-learn internals.
        """
        model = PolynomialModel()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(np.asarray([[0.0, 0.0]]))

    def test_evaluate_returns_regression_metrics(self) -> None:
        """
        Evaluation delegates to the common regression metric helper.
        """
        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)

        model = PolynomialModel(degrees=(2,), alphas=(0.001,))
        model.fit(X_train, y_train, X_val, y_val)

        metrics = model.evaluate(X_val, y_val)

        assert set(metrics) == {"MAE", "RMSE"}
        assert metrics["MAE"] >= 0.0
        assert metrics["RMSE"] >= 0.0

    def test_rejects_empty_degree_grid(self) -> None:
        """
        Degree grid must contain at least one candidate.
        """
        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)

        model = PolynomialModel(degrees=())

        with pytest.raises(ValueError, match="degree"):
            model.fit(X_train, y_train, X_val, y_val)

    def test_rejects_empty_alpha_grid(self) -> None:
        """
        Alpha grid must contain at least one candidate.
        """
        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)

        model = PolynomialModel(alphas=())

        with pytest.raises(ValueError, match="alpha"):
            model.fit(X_train, y_train, X_val, y_val)
