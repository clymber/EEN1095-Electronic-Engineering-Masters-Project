"""
Tests for neural MLP surrogate model classes.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure

pytest.importorskip("keras")

from sparam_surrogate.config.surrogate_config import (  # noqa: E402
    NeuralMLPModelConfig,
    PolynomialNeuralMLPModelConfig,
)
from sparam_surrogate.models.neural_mlp import (  # noqa: E402
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_LEARNING_RATE,
    POLYNOMIAL_NEURAL_DEGREE,
    PREDICTION_BATCH_SIZE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    VECTOR_MLP_RANDOM_STATE,
    PolynomialVectorMLP,
    VectorMLP,
)


def _features() -> tuple[np.ndarray, np.ndarray]:
    """
    Return small train and validation feature matrices.
    """
    X_train = np.asarray(  # pylint: disable=invalid-name
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )
    X_val = np.asarray(  # pylint: disable=invalid-name
        [
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
        ],
        dtype=float,
    )
    return X_train, X_val


def _vector_target(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """
    Return two-output targets for a small vector-regression problem.
    """
    return np.column_stack(
        (
            2.0 * X[:, 0] - 0.5 * X[:, 1] + 1.0,
            -X[:, 0] + 1.5 * X[:, 1] - 2.0,
        )
    )


class TestVectorMLP:
    """
    Unit tests for raw-feature vector MLP models.
    """

    def test_defaults_match_exported_constants(self) -> None:
        """
        Constructor defaults are available as module-level constants.
        """
        model = VectorMLP()

        assert model.batch_size == BATCH_SIZE
        assert model.epochs == MAX_EPOCHS
        assert model.prediction_batch_size == PREDICTION_BATCH_SIZE
        assert model.learning_rate == LEARNING_RATE
        assert model.gradient_clip_norm == GRADIENT_CLIP_NORM
        assert model.early_stopping_patience == EARLY_STOPPING_PATIENCE
        assert model.reduce_lr_patience == REDUCE_LR_PATIENCE
        assert model.reduce_lr_factor == REDUCE_LR_FACTOR
        assert model.min_learning_rate == MIN_LEARNING_RATE
        assert model.random_state == VECTOR_MLP_RANDOM_STATE

    def test_from_config_uses_configured_training_controls(self) -> None:
        """
        Model config objects initialize raw-feature neural MLP controls.
        """
        cfg = NeuralMLPModelConfig(
            batch_size=8,
            epochs=2,
            prediction_batch_size=16,
            learning_rate=0.001,
            gradient_clip_norm=1.5,
            early_stopping_patience=4,
            reduce_lr_patience=2,
            reduce_lr_factor=0.25,
            min_learning_rate=0.00001,
            random_state=7,
        )

        model = VectorMLP.from_config(cfg)

        assert model.batch_size == cfg.batch_size
        assert model.epochs == cfg.epochs
        assert model.prediction_batch_size == cfg.prediction_batch_size
        assert model.learning_rate == cfg.learning_rate
        assert model.gradient_clip_norm == cfg.gradient_clip_norm
        assert model.early_stopping_patience == cfg.early_stopping_patience
        assert model.reduce_lr_patience == cfg.reduce_lr_patience
        assert model.reduce_lr_factor == cfg.reduce_lr_factor
        assert model.min_learning_rate == cfg.min_learning_rate
        assert model.random_state == cfg.random_state

    def test_model_name_keeps_report_label(self) -> None:
        """
        The renamed class keeps the existing notebook label.
        """
        assert VectorMLP().model_name() == "Neural MLP"

    def test_requires_validation_data(self) -> None:
        """
        Fitting requires validation data for callbacks and reporting.
        """
        X_train, _ = _features()
        y_train = _vector_target(X_train)

        with pytest.raises(ValueError, match="VectorMLP requires validation data"):
            VectorMLP(epochs=1).fit(X_train, y_train, verbose=0)

    def test_predict_before_fit_raises_clear_error(self) -> None:
        """
        Predicting before fitting fails before Keras or scaler internals.
        """
        model = VectorMLP()

        with pytest.raises(RuntimeError, match="VectorMLP must be fitted"):
            model.predict(np.asarray([[0.0, 0.0]]))

        with pytest.raises(RuntimeError, match="VectorMLP must be fitted"):
            _ = model.keras_model

    def test_plot_training_history_before_fit_raises_clear_error(self) -> None:
        """
        Training-history plots require a recorded Keras history.
        """
        with pytest.raises(RuntimeError, match="training history"):
            VectorMLP().plot_training_history()

    def test_fit_predicts_vector_shape_and_plots_history(self) -> None:
        """
        A tiny one-epoch fit returns vector predictions and a training plot.
        """
        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)
        model = VectorMLP(epochs=1, batch_size=4, random_state=3)

        result = model.fit(X_train, y_train, X_val, y_val, verbose=0)
        prediction = model.predict(X_val)
        fig = model.plot_training_history()

        assert result is model
        assert prediction.shape == y_val.shape
        assert np.isfinite(prediction).all()
        assert isinstance(fig, Figure)


class TestPolynomialVectorMLP:
    """
    Unit tests for polynomial-feature vector MLP models.
    """

    def test_defaults_match_exported_constants(self) -> None:
        """
        Constructor defaults include the polynomial feature degree.
        """
        model = PolynomialVectorMLP()

        assert model.polynomial_degree == POLYNOMIAL_NEURAL_DEGREE
        assert model.batch_size == BATCH_SIZE
        assert model.epochs == MAX_EPOCHS
        assert model.prediction_batch_size == PREDICTION_BATCH_SIZE
        assert model.learning_rate == LEARNING_RATE
        assert model.gradient_clip_norm == GRADIENT_CLIP_NORM
        assert model.early_stopping_patience == EARLY_STOPPING_PATIENCE
        assert model.reduce_lr_patience == REDUCE_LR_PATIENCE
        assert model.reduce_lr_factor == REDUCE_LR_FACTOR
        assert model.min_learning_rate == MIN_LEARNING_RATE
        assert model.random_state == VECTOR_MLP_RANDOM_STATE

    def test_from_config_uses_configured_training_controls(self) -> None:
        """
        Model config objects initialize polynomial neural MLP controls.
        """
        cfg = PolynomialNeuralMLPModelConfig(
            polynomial_degree=4,
            batch_size=8,
            epochs=2,
            prediction_batch_size=16,
            learning_rate=0.001,
            gradient_clip_norm=1.5,
            early_stopping_patience=4,
            reduce_lr_patience=2,
            reduce_lr_factor=0.25,
            min_learning_rate=0.00001,
            random_state=7,
        )

        model = PolynomialVectorMLP.from_config(cfg)

        assert model.polynomial_degree == cfg.polynomial_degree
        assert model.batch_size == cfg.batch_size
        assert model.epochs == cfg.epochs
        assert model.prediction_batch_size == cfg.prediction_batch_size
        assert model.learning_rate == cfg.learning_rate
        assert model.gradient_clip_norm == cfg.gradient_clip_norm
        assert model.early_stopping_patience == cfg.early_stopping_patience
        assert model.reduce_lr_patience == cfg.reduce_lr_patience
        assert model.reduce_lr_factor == cfg.reduce_lr_factor
        assert model.min_learning_rate == cfg.min_learning_rate
        assert model.random_state == cfg.random_state

    def test_model_name_keeps_report_label(self) -> None:
        """
        The renamed polynomial class keeps the existing notebook label.
        """
        assert PolynomialVectorMLP().model_name() == "Polynomial Neural MLP"

    def test_requires_validation_data(self) -> None:
        """
        Fitting requires validation data for callbacks and reporting.
        """
        X_train, _ = _features()
        y_train = _vector_target(X_train)

        with pytest.raises(
            ValueError,
            match="PolynomialVectorMLP requires validation data",
        ):
            PolynomialVectorMLP(epochs=1).fit(X_train, y_train, verbose=0)

    def test_predict_before_fit_raises_clear_error(self) -> None:
        """
        Predicting before fitting fails before Keras or scaler internals.
        """
        model = PolynomialVectorMLP()

        with pytest.raises(RuntimeError, match="PolynomialVectorMLP must be fitted"):
            model.predict(np.asarray([[0.0, 0.0]]))

        with pytest.raises(RuntimeError, match="PolynomialVectorMLP must be fitted"):
            _ = model.keras_model

    def test_plot_training_history_before_fit_raises_clear_error(self) -> None:
        """
        Training-history plots require a recorded Keras history.
        """
        with pytest.raises(RuntimeError, match="training history"):
            PolynomialVectorMLP().plot_training_history()

    def test_fit_predicts_vector_shape_and_plots_history(self) -> None:
        """
        A tiny one-epoch fit returns vector predictions and a training plot.
        """
        X_train, X_val = _features()
        y_train = _vector_target(X_train)
        y_val = _vector_target(X_val)
        model = PolynomialVectorMLP(
            polynomial_degree=2,
            epochs=1,
            batch_size=4,
            random_state=3,
        )

        result = model.fit(X_train, y_train, X_val, y_val, verbose=0)
        prediction = model.predict(X_val)
        fig = model.plot_training_history()

        assert result is model
        assert model.expanded_feature_count_ == 4
        assert prediction.shape == y_val.shape
        assert np.isfinite(prediction).all()
        assert isinstance(fig, Figure)
