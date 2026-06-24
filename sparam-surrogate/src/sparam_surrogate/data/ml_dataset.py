"""
Lazy deep-learning dataset views backed by the cleaned preprocessing CSV.
"""

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from sparam_surrogate.config import PROJECT_ROOT


class DLDataset:
    """
    Represent one train, validation, or test split from the cleaned CSV.

    The dataset stores feature rows, source metadata, and an optional target
    loader. Eager targets may be cached on disk, while TensorFlow targets remain
    lazy and are loaded as the dataset is iterated.
    """

    REQUIRED_METADATA_COLUMNS = (
        "SIMU_INDEX",
        "FREQ_GHZ",
        "TOUCHSTONE_REL_PATH",
        "SPLIT_TYPE",
    )
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
        "FREQ_GHZ",
    )
    PROGRESS_MODE_ENV = "SPARAM_SURROGATE_PROGRESS"
    FINAL_PROGRESS_MODE = "final"
    CACHE_DIR = PROJECT_ROOT / "data" / "processed"
    CLEANED_FILENAME = "sipi_dataset_cleaned.csv"

    def __init__(
        self,
        dataframe: pd.DataFrame,
        feature_columns: Sequence[str],
        split_type: str,
        target_loader: Callable[[np.ndarray, Mapping[str, Any]], Any] | None = None,
        cache: bool = False,
        source_csv: Path | str | None = None,
    ) -> None:
        """
        Create a split-specific lazy dataset view.
        """
        self._feature_columns = tuple(str(column) for column in feature_columns)
        self._split_type = str(split_type)
        self._dataframe = self._filtered_dataframe(dataframe)
        self._cache = bool(cache)
        self._source_csv = Path(source_csv or self.CACHE_DIR / self.CLEANED_FILENAME)
        if target_loader is None:
            self._target_loader = None
        else:
            self.set_target_loader(target_loader)

    @classmethod
    def from_cleaned_csv(
        cls,
        cleaned_csv: Path | str,
        feature_columns: Sequence[str] | None = None,
        target_loader: Callable[[np.ndarray, Mapping[str, Any]], Any] | None = None,
        cache: bool = False,
    ) -> tuple["DLDataset", "DLDataset", "DLDataset"]:
        """
        Build train, validation, and test split views from a cleaned CSV.

        The default feature order matches the cleaned dataset builder: all PCB
        design parameters followed by ``FREQ_GHZ``.
        """
        source_csv = Path(cleaned_csv)
        cleaned = pd.read_csv(source_csv)
        selected_features = tuple(feature_columns or cls.DEFAULT_FEATURE_COLUMNS)
        return (
            cls(cleaned, selected_features, "train", target_loader, cache, source_csv),
            cls(cleaned, selected_features, "val", target_loader, cache, source_csv),
            cls(cleaned, selected_features, "test", target_loader, cache, source_csv),
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Return a copy of the split-specific cleaned dataframe.
        """
        return self._dataframe.copy()

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """
        Return feature columns in tensor order.
        """
        return self._feature_columns

    @property
    def split_type(self) -> str:
        """
        Return the split label represented by this dataset.
        """
        return self._split_type

    @property
    def features(self) -> np.ndarray:
        """
        Return feature values as a two-dimensional float array.
        """
        return self._dataframe.loc[:, self._feature_columns].to_numpy(dtype=float)

    @property
    def row_metadata(self) -> pd.DataFrame:
        """
        Return metadata columns aligned with :attr:`features`.
        """
        return self._dataframe.loc[
            :, ["SIMU_INDEX", "FREQ_GHZ", "TOUCHSTONE_REL_PATH"]
        ].copy()

    def __len__(self) -> int:
        """
        Return the number of design-frequency rows in this split.
        """
        return len(self._dataframe)

    def load_targets(self) -> np.ndarray:
        """
        Materialize targets by applying the stored loader to every split row.
        """
        loader = self._require_target_loader()
        progress_desc = f"Loading {self.split_type} targets"
        final_progress = (
            os.environ.get(self.PROGRESS_MODE_ENV) == self.FINAL_PROGRESS_MODE
        )
        rows: Iterable[dict[str, Any]] = cast(
            list[dict[str, Any]],
            self.row_metadata.to_dict("records"),
        )

        if not final_progress:
            rows = tqdm(
                rows,
                total=len(self),
                desc=progress_desc,
            )

        # Empty feature array passed only to keep the loader signature.
        _ = np.empty(0, dtype=float)

        progress_start = perf_counter()
        targets = np.stack(
            [np.asarray(loader(_, row_metadata)) for row_metadata in rows]
        )
        elapsed = perf_counter() - progress_start

        if final_progress:
            progress_status = tqdm.format_meter(
                len(self), len(self), elapsed, prefix=progress_desc, ascii=True
            )
            print(progress_status)

        return targets


    def set_target_loader(
        self,
        target_loader: Callable[[np.ndarray, Mapping[str, Any]], Any],
    ) -> None:
        """
        Set the callable used to load target values.
        """
        if not callable(target_loader):
            raise TypeError("target_loader must be callable.")
        self._target_loader = target_loader


    @property
    def targets(self) -> np.ndarray:
        """
        Return targets for this split, using the disk cache when enabled.
        """
        if not self._cache:
            return self.load_targets()

        cache_path = self._target_cache_path()
        if (
            cache_path.is_file()
            and cache_path.stat().st_mtime_ns > self._source_csv.stat().st_mtime_ns
        ):
            with np.load(cache_path, allow_pickle=False) as cached:
                return np.asarray(cached["targets"])

        targets = self.load_targets()
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, targets=targets)
        return targets


    def to_tf_dataset(
        self,
        batch_size: int,
        shuffle: bool = False,
        prefetch: bool = True,
    ) -> "tf.data.Dataset":
        """
        Build a ``tf.data.Dataset`` with targets loaded during iteration.
        """
        target_loader = self._require_target_loader()
        if getattr(target_loader, "representation", None) == "complex":
            raise ValueError(
                "complex target loaders are not supported by to_tf_dataset(); "
                "use a real-valued representation."
            )
        features = self.features.astype(np.float32)
        metadata = self.row_metadata
        simulation_indices = metadata["SIMU_INDEX"].to_numpy(dtype=np.int64)
        frequencies_ghz = metadata["FREQ_GHZ"].to_numpy(dtype=np.float32)
        touchstone_paths = (
            metadata["TOUCHSTONE_REL_PATH"].astype(str).to_numpy(dtype=np.bytes_)
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            (features, simulation_indices, frequencies_ghz, touchstone_paths)
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self))

        feature_width = len(self._feature_columns)
        target_shape = getattr(target_loader, "target_shape", None)

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
                target = target_loader(
                    np.asarray(to_numpy(feature_value)),
                    row_metadata,
                )
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

    def _require_target_loader(
        self,
    ) -> Callable[[np.ndarray, Mapping[str, Any]], Any]:
        """
        Return the configured target loader or raise a clear error.
        """
        if self._target_loader is None:
            raise RuntimeError("A target loader is required for target loading.")
        return self._target_loader

    def _target_cache_path(self) -> Path:
        """
        Return the cache path selected by loader mode and representation.
        """
        loader = self._require_target_loader()
        mode = getattr(loader, "mode", None)
        representation = getattr(loader, "representation", None)
        if mode is None or representation is None:
            raise ValueError(
                "A cache-enabled target loader must define mode and representation."
            )
        return self.CACHE_DIR / f"{mode}_{representation}_{self.split_type}.npz"

    def _filtered_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Select this dataset's split rows from a cleaned dataframe.
        """
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
