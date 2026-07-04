"""
Neural MLP surrogate models.
"""

from __future__ import annotations

from typing import Any

import keras
import numpy as np
from keras import Input, layers
from sklearn.preprocessing import StandardScaler

from sparam_surrogate.models.neural import NeuralModel
from sparam_surrogate.models.polynomial import PowersOnlyPolynomialFeatures

PREDICTION_BATCH_SIZE = 4096
BATCH_SIZE = 512
LEARNING_RATE = 3e-5
GRADIENT_CLIP_NORM = 0.5
EARLY_STOPPING_PATIENCE = 18
REDUCE_LR_PATIENCE = 6
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6
MAX_EPOCHS = 100
POLYNOMIAL_NEURAL_DEGREE = 5
VECTOR_MLP_RANDOM_STATE = 128

# Keras accepts float callback deltas, but Pyright infers int from the
# EarlyStopping runtime default of 0.
CALLBACK_MIN_DELTA: Any = 1e-4


def build_vector_mlp(input_width: int, output_width: int) -> keras.Model:
    """
    Return the vector-output MLP architecture used by neural baselines.
    """
    inputs = Input(
        shape=(input_width,),
        name="design_frequency_features",
    )

    hidden = layers.Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        name="dense_128_a",
    )(inputs)

    hidden = layers.Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        name="dense_128_b",
    )(hidden)

    hidden = layers.Dense(
        64,
        activation="relu",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        name="dense_64",
    )(hidden)

    outputs = layers.Dense(
        output_width,
        activation="linear",
        kernel_initializer=keras.initializers.RandomNormal(
            mean=0.0,
            stddev=1e-2,
        ),  # type: ignore[arg-type]
        bias_initializer="zeros",
        name="s_db_outputs",
    )(hidden)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="vector_mlp_baseline",
    )


class VectorMLP(NeuralModel):
    """
    Keras MLP wrapper implementing the common surrogate model interface.
    """

    name = "neural_mlp"

    def __init__(
        self,
        *,
        batch_size: int = BATCH_SIZE,
        epochs: int = MAX_EPOCHS,
        prediction_batch_size: int = PREDICTION_BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        gradient_clip_norm: float = GRADIENT_CLIP_NORM,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
        reduce_lr_patience: int = REDUCE_LR_PATIENCE,
        reduce_lr_factor: float = REDUCE_LR_FACTOR,
        min_learning_rate: float = MIN_LEARNING_RATE,
        random_state: int = VECTOR_MLP_RANDOM_STATE,
    ) -> None:
        """
        Store training controls and scaler state.
        """
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.prediction_batch_size = int(prediction_batch_size)
        self.learning_rate = float(learning_rate)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.early_stopping_patience = int(early_stopping_patience)
        self.reduce_lr_patience = int(reduce_lr_patience)
        self.reduce_lr_factor = float(reduce_lr_factor)
        self.min_learning_rate = float(min_learning_rate)
        self.random_state = int(random_state)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model: keras.Model | None = None
        self.history: keras.callbacks.History | None = None

    def model_name(self) -> str:
        """
        Return the plot label with the expected MLP capitalization.
        """
        return "Neural MLP"

    def fit(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,  # pylint: disable=invalid-name
        y_val: np.ndarray | None = None,
        verbose: str | int = 2,
    ) -> VectorMLP:
        """
        Fit the neural baseline using scaled features and scaled targets.
        """
        if X_val is None or y_val is None:
            raise ValueError("VectorMLP requires validation data.")

        keras.utils.set_random_seed(self.random_state)
        X_train_scaled = self.x_scaler.fit_transform(  # pylint: disable=invalid-name
            np.asarray(X_train, dtype=float)
        ).astype(np.float32)
        X_val_scaled = self.x_scaler.transform(  # pylint: disable=invalid-name
            np.asarray(X_val, dtype=float)
        ).astype(np.float32)
        y_train_scaled = self.y_scaler.fit_transform(
            np.asarray(y_train, dtype=float)
        ).astype(np.float32)
        y_val_scaled = self.y_scaler.transform(np.asarray(y_val, dtype=float)).astype(
            np.float32
        )

        self.model = build_vector_mlp(
            input_width=X_train_scaled.shape[1],
            output_width=y_train_scaled.shape[1],
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=self.gradient_clip_norm,
            ),
            loss="mse",
            steps_per_execution=8,
        )

        self.history = self.model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=self.reduce_lr_factor,
                    patience=self.reduce_lr_patience,
                    min_delta=CALLBACK_MIN_DELTA,
                    min_lr=self.min_learning_rate,
                    verbose=1,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    min_delta=CALLBACK_MIN_DELTA,
                    restore_best_weights=True,
                    verbose=1,
                ),
            ],
            shuffle=True,
            verbose=verbose,  # type: ignore[assignment]
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Return inverse-transformed dB predictions from the fitted Keras model.
        """
        model = self.keras_model
        X_scaled = self.x_scaler.transform(  # noqa: N806
            np.asarray(X, dtype=float)
        ).astype(np.float32)
        y_pred_scaled = model.predict(
            X_scaled,
            batch_size=self.prediction_batch_size,
            verbose=0,  # type: ignore[assignment]
        )
        return self.y_scaler.inverse_transform(np.asarray(y_pred_scaled, dtype=float))

    @property
    def keras_model(self) -> keras.Model:
        """
        Return the fitted Keras model.
        """
        if self.model is None:
            raise RuntimeError("VectorMLP must be fitted before prediction.")
        return self.model


class PolynomialVectorMLP(NeuralModel):
    """
    Keras MLP trained on powers-only polynomial feature expansions.
    """

    name = "polynomial_neural_mlp"

    def __init__(
        self,
        *,
        polynomial_degree: int = POLYNOMIAL_NEURAL_DEGREE,
        batch_size: int = BATCH_SIZE,
        epochs: int = MAX_EPOCHS,
        prediction_batch_size: int = PREDICTION_BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        gradient_clip_norm: float = GRADIENT_CLIP_NORM,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
        reduce_lr_patience: int = REDUCE_LR_PATIENCE,
        reduce_lr_factor: float = REDUCE_LR_FACTOR,
        min_learning_rate: float = MIN_LEARNING_RATE,
        random_state: int = VECTOR_MLP_RANDOM_STATE,
    ) -> None:
        """
        Store training controls and polynomial preprocessing state.
        """
        self.polynomial_degree = int(polynomial_degree)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.prediction_batch_size = int(prediction_batch_size)
        self.learning_rate = float(learning_rate)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.early_stopping_patience = int(early_stopping_patience)
        self.reduce_lr_patience = int(reduce_lr_patience)
        self.reduce_lr_factor = float(reduce_lr_factor)
        self.min_learning_rate = float(min_learning_rate)
        self.random_state = int(random_state)
        self.input_scaler = StandardScaler()
        self.polynomial_features = PowersOnlyPolynomialFeatures(
            degree=self.polynomial_degree
        )
        self.expanded_feature_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model: keras.Model | None = None
        self.history: keras.callbacks.History | None = None
        self.expanded_feature_count_: int | None = None

    def model_name(self) -> str:
        """
        Return the plot label for the polynomial neural baseline.
        """
        return "Polynomial Neural MLP"

    def _fit_transform_features(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
    ) -> np.ndarray:
        """
        Fit the polynomial preprocessing chain and return scaled features.
        """
        X_input_scaled = self.input_scaler.fit_transform(  # noqa: N806
            np.asarray(X, dtype=float)
        )
        X_expanded = self.polynomial_features.fit_transform(  # noqa: N806
            X_input_scaled
        )
        self.expanded_feature_count_ = int(X_expanded.shape[1])
        return self.expanded_feature_scaler.fit_transform(X_expanded).astype(
            np.float32
        )

    def _transform_features(
        self,
        X: np.ndarray,  # pylint: disable=invalid-name
    ) -> np.ndarray:
        """
        Apply the fitted polynomial preprocessing chain to new features.
        """
        X_input_scaled = self.input_scaler.transform(  # noqa: N806
            np.asarray(X, dtype=float)
        )
        X_expanded = self.polynomial_features.transform(X_input_scaled)  # noqa: N806
        return self.expanded_feature_scaler.transform(X_expanded).astype(np.float32)

    def fit(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,  # pylint: disable=invalid-name
        y_val: np.ndarray | None = None,
        verbose: str | int = 2,
    ) -> PolynomialVectorMLP:
        """
        Fit the neural baseline using polynomial features and scaled targets.
        """
        if X_val is None or y_val is None:
            raise ValueError("PolynomialVectorMLP requires validation data.")

        keras.utils.set_random_seed(self.random_state)
        X_train_scaled = self._fit_transform_features(X_train)  # noqa: N806
        X_val_scaled = self._transform_features(X_val)  # noqa: N806
        y_train_scaled = self.y_scaler.fit_transform(
            np.asarray(y_train, dtype=float)
        ).astype(np.float32)
        y_val_scaled = self.y_scaler.transform(np.asarray(y_val, dtype=float)).astype(
            np.float32
        )

        self.model = build_vector_mlp(
            input_width=X_train_scaled.shape[1],
            output_width=y_train_scaled.shape[1],
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=self.gradient_clip_norm,
            ),
            loss="mse",
            steps_per_execution=8,
        )

        self.history = self.model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=self.reduce_lr_factor,
                    patience=self.reduce_lr_patience,
                    min_delta=CALLBACK_MIN_DELTA,
                    min_lr=self.min_learning_rate,
                    verbose=1,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    min_delta=CALLBACK_MIN_DELTA,
                    restore_best_weights=True,
                    verbose=1,
                ),
            ],
            shuffle=True,
            verbose=verbose,  # type: ignore[assignment]
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Return inverse-transformed dB predictions from the fitted Keras model.
        """
        model = self.keras_model
        X_scaled = self._transform_features(X)  # noqa: N806
        y_pred_scaled = model.predict(
            X_scaled,
            batch_size=self.prediction_batch_size,
            verbose=0,  # type: ignore[assignment]
        )
        return self.y_scaler.inverse_transform(np.asarray(y_pred_scaled, dtype=float))

    @property
    def keras_model(self) -> keras.Model:
        """
        Return the fitted Keras model.
        """
        if self.model is None:
            raise RuntimeError("PolynomialVectorMLP must be fitted before prediction.")
        return self.model
