"""
Lazy deep-learning dataset views backed by the cleaned preprocessing CSV.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


class DLDataset:
    """
    Represent one train, validation, or test split from the cleaned CSV.

    The dataset stores feature rows and source metadata only. S-parameter
    targets are intentionally loaded later by a callable map function so that
    training does not need the full Touchstone database in memory.
    """

    REQUIRED_METADATA_COLUMNS = (
        "SIMU_INDEX",
        "FREQ_GHZ",
        "TOUCHSTONE_REL_PATH",
        "SPLIT_TYPE",
    )

    def __init__(
        self,
        dataframe: pd.DataFrame,
        feature_columns: Sequence[str],
        split_type: str,
    ) -> None:
        """Create a split-specific lazy dataset view."""
        self._feature_columns = tuple(str(column) for column in feature_columns)
        self._split_type = str(split_type)
        self._dataframe = self._filtered_dataframe(dataframe)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return a copy of the split-specific cleaned dataframe."""
        return self._dataframe.copy()

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return feature columns in tensor order."""
        return self._feature_columns

    @property
    def split_type(self) -> str:
        """Return the split label represented by this dataset."""
        return self._split_type

    @property
    def features(self) -> np.ndarray:
        """Return feature values as a two-dimensional float array."""
        return self._dataframe.loc[:, self._feature_columns].to_numpy(dtype=float)

    @property
    def row_metadata(self) -> pd.DataFrame:
        """Return metadata columns aligned with :attr:`features`."""
        return self._dataframe.loc[
            :, ["SIMU_INDEX", "FREQ_GHZ", "TOUCHSTONE_REL_PATH"]
        ].copy()

    def __len__(self) -> int:
        """Return the number of design-frequency rows in this split."""
        return len(self._dataframe)

    def to_tf_dataset(
        self,
        map_func: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
        batch_size: int,
        shuffle: bool = False,
        prefetch: bool = True,
    ) -> "tf.data.Dataset":
        """Build a ``tf.data.Dataset`` with lazy Touchstone targets."""
        features = self.features.astype(np.float32)
        metadata = self.row_metadata
        simulation_indices = metadata["SIMU_INDEX"].to_numpy(dtype=np.int64)
        frequencies_ghz = metadata["FREQ_GHZ"].to_numpy(dtype=np.float32)
        touchstone_paths = metadata["TOUCHSTONE_REL_PATH"].astype(str).to_numpy(
            dtype=np.bytes_
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            (features, simulation_indices, frequencies_ghz, touchstone_paths)
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self))

        feature_width = len(self._feature_columns)
        target_shape = getattr(map_func, "target_shape", None)

        def mapper(feature_row, simulation_index, frequency_ghz, touchstone_path):
            def python_target(
                feature_value,
                simulation_value,
                frequency_value,
                path_value,
            ):
                def to_numpy(value):
                    return value.numpy() if hasattr(value, "numpy") else value

                raw_path_value = to_numpy(path_value)
                if isinstance(raw_path_value, np.ndarray):
                    raw_path_value = raw_path_value.item()
                raw_path = raw_path_value.decode("utf-8")
                row_metadata = {
                    "SIMU_INDEX": int(to_numpy(simulation_value)),
                    "FREQ_GHZ": float(to_numpy(frequency_value)),
                    "TOUCHSTONE_REL_PATH": raw_path,
                }
                target = map_func(np.asarray(to_numpy(feature_value)), row_metadata)
                return np.asarray(target, dtype=np.float32)

            target = tf.py_function(
                python_target,
                [feature_row, simulation_index, frequency_ghz, touchstone_path],
                Tout=tf.float32,
            )
            feature_row.set_shape((feature_width,))
            if target_shape is not None:
                target.set_shape(tuple(target_shape))
            return feature_row, target

        dataset = dataset.map(mapper)
        dataset = dataset.batch(int(batch_size))
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def _filtered_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Select this dataset's split rows from a cleaned dataframe."""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")

        required = [*self._feature_columns, *self.REQUIRED_METADATA_COLUMNS]
        missing = [column for column in required if column not in dataframe.columns]
        if missing:
            raise ValueError("Required columns missing: " + ", ".join(missing))

        filtered = dataframe.loc[
            dataframe["SPLIT_TYPE"].astype(str) == self._split_type
        ].copy()
        if filtered.empty:
            raise ValueError(f"No rows found for split_type={self._split_type!r}.")

        return filtered.reset_index(drop=True)
