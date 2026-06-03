"""
Feature construction for PCB design-frequency model inputs.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pcb_parameters import PcbParameters


@dataclass(frozen=True)
class PcbFeatureMatrix:
    """
    Expanded PCB feature matrix and aligned row metadata.

    The feature rows use design-major order: each design is repeated across the
    full frequency grid before the next design is emitted.
    """

    X: np.ndarray  # Expanded feature rows, one per design-frequency pair.
    feature_names: tuple[str, ...]  # Column names aligned with X.
    simulation_indices: np.ndarray  # Simulation ID aligned with each row.
    frequencies_ghz: np.ndarray  # Frequency value aligned with each row.


class PcbFeatureTransformer:
    """
    Build and scale ``X = [geometric/material parameters, frequency]``.

    Scaling statistics are fitted from rows whose design-level split label is
    ``"train"`` and are then applied to every split.
    """

    DEFAULT_FEATURE_COLUMNS = (
        "EPS",
        "TAND",
        "PITCH",
        "TRACE_LEN",
        "START",
        "VIAR",
        "ANTIPADR",
        "TDIEL",
        "DISTTL",
        "TLWIDTH",
    )

    def __init__(
        self,
        feature_columns: Sequence[str] | None = None,
        frequency_column_name: str = "FREQ_GHZ",
        scale: bool = True,
    ) -> None:
        """
        Configure feature selection and scaling.

        Parameters
        ----------
        feature_columns:
            Parameter columns to include before the frequency column. When
            omitted, the project's physical PCB parameter columns are used.
        frequency_column_name:
            Name assigned to the appended frequency feature.
        scale:
            Whether to standardize features using train-row mean and standard
            deviation.

        Raises
        ------
        ValueError
            If feature or frequency names are empty.
        """
        columns = (
            self.DEFAULT_FEATURE_COLUMNS
            if feature_columns is None
            else tuple(str(column) for column in feature_columns)
        )
        if len(columns) == 0 or any(not column for column in columns):
            raise ValueError("feature_columns must contain non-empty names.")
        if not frequency_column_name:
            raise ValueError("frequency_column_name must be non-empty.")

        self.feature_columns = tuple(columns)
        self.frequency_column_name = str(frequency_column_name)
        self.scale = bool(scale)
        self.feature_names_ = self.feature_columns + (self.frequency_column_name,)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self._is_fitted = False

    def fit_transform(
        self,
        parameters: PcbParameters | pd.DataFrame,
        frequencies_ghz: Sequence[float] | np.ndarray,
        split_labels: Sequence[str] | np.ndarray,
    ) -> PcbFeatureMatrix:
        """
        Fit train-only scaling statistics and return expanded features.

        Parameters
        ----------
        parameters:
            PCB parameter table containing ``SIMU_INDEX`` and all selected
            feature columns.
        frequencies_ghz:
            One-dimensional Touchstone frequency grid in GHz.
        split_labels:
            One design-level split label per parameter row. Rows labelled
            ``"train"`` are used to fit scaling statistics.

        Returns
        -------
        PcbFeatureMatrix
            Scaled or unscaled feature matrix plus aligned row metadata.

        Raises
        ------
        ValueError
            If selected feature columns are missing, input values are invalid,
            split labels do not match the design count, or no train rows are
            available for scaling.
        """
        feature_values, simulation_indices, frequencies = self._prepare_inputs(
            parameters,
            frequencies_ghz,
        )
        labels = self._validate_split_labels(
            split_labels,
            n_designs=len(simulation_indices),
        )
        raw_X = self._build_feature_matrix(feature_values, frequencies) # pylint: disable=invalid-name

        if self.scale:
            train_mask = np.repeat(labels == "train", len(frequencies))
            if not train_mask.any():
                raise ValueError("split_labels must include at least one train row.")
            self.mean_ = raw_X[train_mask].mean(axis=0)
            raw_scale = raw_X[train_mask].std(axis=0)
            self.scale_ = np.where(raw_scale == 0.0, 1.0, raw_scale)
        else:
            self.mean_ = np.zeros(raw_X.shape[1], dtype=float)
            self.scale_ = np.ones(raw_X.shape[1], dtype=float)

        self._is_fitted = True
        return self._feature_matrix(raw_X, simulation_indices, frequencies)

    def transform(
        self,
        parameters: PcbParameters | pd.DataFrame,
        frequencies_ghz: Sequence[float] | np.ndarray,
    ) -> PcbFeatureMatrix:
        """
        Return expanded features using already-fitted scaling statistics.

        Parameters
        ----------
        parameters:
            PCB parameter table containing ``SIMU_INDEX`` and all selected
            feature columns.
        frequencies_ghz:
            One-dimensional Touchstone frequency grid in GHz.

        Returns
        -------
        PcbFeatureMatrix
            Feature matrix and aligned row metadata transformed with the
            statistics learned during :meth:`fit_transform`.

        Raises
        ------
        RuntimeError
            If the transformer has not been fitted.
        ValueError
            If selected feature columns are missing or input values are invalid.
        """
        if not self._is_fitted:
            raise RuntimeError("PcbFeatureTransformer must be fitted before transform.")

        feature_values, simulation_indices, frequencies = self._prepare_inputs(
            parameters,
            frequencies_ghz,
        )
        raw_X = self._build_feature_matrix(feature_values, frequencies) # pylint: disable=invalid-name
        return self._feature_matrix(raw_X, simulation_indices, frequencies)

    def _feature_matrix(
        self,
        raw_X: np.ndarray, # pylint: disable=invalid-name
        simulation_indices: np.ndarray,
        frequencies_ghz: np.ndarray,
    ) -> PcbFeatureMatrix:
        """
        Package scaled features and expanded metadata.
        """
        return PcbFeatureMatrix(
            X=self._apply_scaling(raw_X),
            feature_names=self.feature_names_,
            simulation_indices=np.repeat(simulation_indices, len(frequencies_ghz)),
            frequencies_ghz=np.tile(frequencies_ghz, len(simulation_indices)),
        )

    def _apply_scaling(self, raw_X: np.ndarray) -> np.ndarray: # pylint: disable=invalid-name
        """
        Apply fitted standardization statistics.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("PcbFeatureTransformer must be fitted before scaling.")
        return (raw_X - self.mean_) / self.scale_

    def _prepare_inputs(
        self,
        parameters: PcbParameters | pd.DataFrame,
        frequencies_ghz: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate parameter and frequency inputs.
        """
        frame = self._parameter_frame(parameters)
        missing = [column for column in self.feature_columns if column not in frame]
        if missing:
            raise ValueError(f"Missing feature columns: {', '.join(missing)}.")
        if "SIMU_INDEX" not in frame:
            raise ValueError("parameters must contain a SIMU_INDEX column.")

        try:
            feature_values = frame.loc[:, self.feature_columns].to_numpy(dtype=float)
        except ValueError as exc:
            raise ValueError("Feature columns must be numeric.") from exc
        if len(feature_values) == 0:
            raise ValueError("parameters must contain at least one design row.")
        if not np.isfinite(feature_values).all():
            raise ValueError("Parameter table contains non-finite feature values.")

        simulation_indices = self._validate_simulation_indices(frame["SIMU_INDEX"])
        frequencies = np.asarray(frequencies_ghz, dtype=float)
        if frequencies.ndim != 1 or len(frequencies) == 0:
            raise ValueError("frequencies_ghz must contain at least one value.")
        if not np.isfinite(frequencies).all():
            raise ValueError("frequencies_ghz contains non-finite frequency values.")

        return feature_values, simulation_indices, frequencies

    @staticmethod
    def _parameter_frame(parameters: PcbParameters | pd.DataFrame) -> pd.DataFrame:
        """
        Return the underlying parameter dataframe.
        """
        if isinstance(parameters, PcbParameters):
            return parameters.dataframe
        if isinstance(parameters, pd.DataFrame):
            return parameters
        raise TypeError("parameters must be a PcbParameters or pandas DataFrame.")

    @staticmethod
    def _validate_simulation_indices(indices: pd.Series) -> np.ndarray:
        """
        Validate ``SIMU_INDEX`` values and return an integer array.
        """
        float_indices = indices.to_numpy(dtype=float)
        if not np.isfinite(float_indices).all():
            raise ValueError("SIMU_INDEX contains non-finite values.")
        integer_indices = float_indices.astype(np.int64)
        if not np.allclose(float_indices, integer_indices):
            raise ValueError("SIMU_INDEX values must be integer-valued.")
        if len(np.unique(integer_indices)) != len(integer_indices):
            raise ValueError("SIMU_INDEX values must be unique.")
        return integer_indices

    @staticmethod
    def _build_feature_matrix(
        feature_values: np.ndarray,
        frequencies_ghz: np.ndarray,
    ) -> np.ndarray:
        """
        Expand design-level parameters over the frequency grid.
        """
        repeated_features = np.repeat(feature_values, len(frequencies_ghz), axis=0)
        tiled_frequencies = np.tile(frequencies_ghz, len(feature_values)).reshape(-1, 1)
        return np.column_stack([repeated_features, tiled_frequencies])

    def _validate_split_labels(
        self,
        split_labels: Sequence[str] | np.ndarray,
        n_designs: int,
    ) -> np.ndarray:
        """
        Validate one design-level split label per parameter row.
        """
        labels = np.asarray(split_labels, dtype=str)
        if labels.shape != (n_designs,):
            raise ValueError("split_labels must contain one label per design.")
        if self.scale and not np.any(labels == "train"):
            raise ValueError("split_labels must include at least one train design.")
        return labels
