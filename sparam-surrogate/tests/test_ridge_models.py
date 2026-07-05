"""
Tests for Ridge surrogate model classes.
"""

import numpy as np
import pytest

from sparam_surrogate.config.surrogate_config import RidgeModelConfig
from sparam_surrogate.models import ScalarRidgeModel, VectorRidgeModel


def _features() -> tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(  # pylint: disable=invalid-name
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    X_val = np.asarray(  # pylint: disable=invalid-name
        [
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
        ]
    )
    return X_train, X_val


def _scalar_target(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    return 2.0 * X[:, 0] - 0.5 * X[:, 1] + 1.0


def _vector_target(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    return np.column_stack(
        (
            _scalar_target(X),
            -X[:, 0] + 1.5 * X[:, 1] - 2.0,
        )
    )


class TestScalarRidgeModel:
    """
    Unit tests for scalar Ridge surrogate models.
    """

    def test_fit_records_validation_results_and_predicts_scalar_values(self) -> None:
        """
        Fitting chooses one alpha and returns one prediction per row.
        """
        X_train, X_val = _features()
        y_train = _scalar_target(X_train)
        y_val = _scalar_target(X_val)
        alphas = (0.001, 0.1, 1.0)

        model = ScalarRidgeModel(alphas=alphas)
        result = model.fit(X_train, y_train, X_val, y_val)

        assert result is model
        assert model.best_alpha in alphas
        assert model.validation_results is not None
        assert len(model.validation_results) == len(alphas)
        assert model.validation_results["alpha"].tolist() == list(alphas)

        prediction = model.predict(X_val)

        assert prediction.shape == y_val.shape
        assert np.isfinite(prediction).all()

    def test_evaluate_returns_regression_metrics(self) -> None:
        """
        Evaluation delegates to the common regression metric helper.
        """
        X_train, X_val = _features()
        y_train = _scalar_target(X_train)
        y_val = _scalar_target(X_val)

        model = ScalarRidgeModel(alphas=(0.001,))
        model.fit(X_train, y_train, X_val, y_val)

        metrics = model.evaluate(X_val, y_val)

        assert set(metrics) == {"MAE", "RMSE"}
        assert metrics["MAE"] >= 0.0
        assert metrics["RMSE"] >= 0.0

    def test_predict_before_fit_raises_clear_error(self) -> None:
        """
        Unfitted models fail before reaching scikit-learn internals.
        """
        model = ScalarRidgeModel()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(np.asarray([[0.0, 0.0]]))

    def test_from_config_uses_configured_alpha_grid(self) -> None:
        """
        Model config objects initialize scalar Ridge hyperparameters.
        """
        cfg = RidgeModelConfig(alphas=(0.02, 0.2))

        model = ScalarRidgeModel.from_config(cfg)

        assert model.alphas == cfg.alphas


class TestVectorRidgeModel:
    """
    Unit tests for vector Ridge surrogate models.
    """

    def test_fit_predicts_one_column_per_vector_target(self) -> None:
        """
        Multi-output Ridge keeps the target matrix shape.
        """
        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)
        alphas = (0.001, 0.1)

        model = VectorRidgeModel(alphas=alphas)
        model.fit(X_train, y_train, X_val, y_val)

        prediction = model.predict(X_val)

        assert model.best_alpha in alphas
        assert model.validation_results is not None
        assert len(model.validation_results) == len(alphas)
        assert prediction.shape == y_val.shape
        assert np.isfinite(prediction).all()

    def test_from_config_uses_configured_alpha_grid(self) -> None:
        """
        Model config objects initialize vector Ridge hyperparameters.
        """
        cfg = RidgeModelConfig(alphas=(0.03, 0.3))

        model = VectorRidgeModel.from_config(cfg)

        assert model.alphas == cfg.alphas
