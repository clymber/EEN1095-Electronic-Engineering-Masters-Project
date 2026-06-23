"""
Tests for Random Forest surrogate models.
"""

import numpy as np
import pytest

from sparam_surrogate.models import (
    RANDOM_FOREST_MAX_DEPTH_GRID,
    RANDOM_FOREST_MIN_SAMPLES_LEAF_GRID,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_N_JOBS,
    RANDOM_FOREST_RANDOM_STATE,
    RandomForestModel,
)


def _features() -> tuple[np.ndarray, np.ndarray]:
    """
    Return small train and validation feature matrices.
    """
    X_train = np.asarray(  # pylint: disable=invalid-name
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
            [0.0, 2.0],
            [0.5, 2.0],
            [1.0, 2.0],
        ]
    )
    X_val = np.asarray(  # pylint: disable=invalid-name
        [
            [0.25, 0.5],
            [0.75, 1.5],
            [1.0, 0.5],
        ]
    )
    return X_train, X_val


def _scalar_target(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """
    Return a simple nonlinear scalar target.
    """
    return X[:, 0] ** 2 + 0.5 * X[:, 1]


def _vector_target(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """
    Return a two-output nonlinear target matrix.
    """
    return np.column_stack(
        (
            _scalar_target(X),
            np.sin(X[:, 0]) - X[:, 1],
        )
    )


class TestRandomForestModel:
    """
    Unit tests for Random Forest surrogate models.
    """

    def test_default_parameters_match_compact_baseline(self) -> None:
        """
        Defaults keep the compact 256-tree nonlinear baseline configuration.
        """
        model = RandomForestModel()

        assert RANDOM_FOREST_N_ESTIMATORS == 256
        assert RANDOM_FOREST_MAX_DEPTH_GRID == (None,)
        assert RANDOM_FOREST_MIN_SAMPLES_LEAF_GRID == (2,)
        assert RANDOM_FOREST_RANDOM_STATE == 42
        assert RANDOM_FOREST_N_JOBS == -1
        assert model.n_estimators == RANDOM_FOREST_N_ESTIMATORS
        assert model.max_depths == RANDOM_FOREST_MAX_DEPTH_GRID
        assert model.min_samples_leafs == RANDOM_FOREST_MIN_SAMPLES_LEAF_GRID

    def test_fit_records_validation_results_and_predicts_vector_values(self) -> None:
        """
        Fitting sweeps tree candidates and returns vector predictions.
        """
        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)
        max_depths = (None, 3)
        min_samples_leafs = (1, 2)

        model = RandomForestModel(
            n_estimators=8,
            max_depths=max_depths,
            min_samples_leafs=min_samples_leafs,
            random_state=7,
            n_jobs=1,
        )
        result = model.fit(X_train, y_train, X_val, y_val)

        assert result is model
        assert model.best_max_depth in max_depths
        assert model.best_min_samples_leaf in min_samples_leafs
        assert model.validation_results is not None
        assert len(model.validation_results) == len(max_depths) * len(
            min_samples_leafs
        )

        prediction = model.predict(X_val)

        assert prediction.shape == y_val.shape
        assert np.isfinite(prediction).all()

    def test_fit_predicts_scalar_values(self) -> None:
        """
        Scalar targets keep one prediction per row.
        """
        X_train, X_val = _features()
        y_train = _scalar_target(X_train)
        y_val = _scalar_target(X_val)

        model = RandomForestModel(n_estimators=6, min_samples_leafs=(1,), n_jobs=1)
        model.fit(X_train, y_train, X_val, y_val)

        prediction = model.predict(X_val)

        assert prediction.shape == y_val.shape
        assert np.isfinite(prediction).all()

    def test_evaluate_returns_regression_metrics(self) -> None:
        """
        Evaluation delegates to the common regression metric helper.
        """
        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)

        model = RandomForestModel(n_estimators=6, min_samples_leafs=(1,), n_jobs=1)
        model.fit(X_train, y_train, X_val, y_val)

        metrics = model.evaluate(X_val, y_val)

        assert set(metrics) == {"MAE", "RMSE"}
        assert metrics["MAE"] >= 0.0
        assert metrics["RMSE"] >= 0.0

    def test_predict_before_fit_raises_clear_error(self) -> None:
        """
        Unfitted models fail before reaching scikit-learn internals.
        """
        model = RandomForestModel()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(np.asarray([[0.0, 0.0]]))

    def test_rejects_empty_candidate_grids(self) -> None:
        """
        Candidate grids must contain at least one value.
        """
        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)

        with pytest.raises(ValueError, match="max depth"):
            RandomForestModel(max_depths=()).fit(X_train, y_train, X_val, y_val)

        with pytest.raises(ValueError, match="min_samples_leaf"):
            RandomForestModel(min_samples_leafs=()).fit(X_train, y_train, X_val, y_val)
