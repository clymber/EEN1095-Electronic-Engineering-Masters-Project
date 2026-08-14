"""
Point-wise dataset views backed by the frequency-expanded CSV.
"""

import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from sparam_surrogate.config import PROJECT_ROOT

from ._skrf_compat import rf
from .parameter_dataset_builder import ParameterDatasetBuilder


class PointwiseDataset:
    """
    Represent one split from the frequency-expanded CSV.

    The dataset stores feature rows, source metadata, and an optional target
    loader. Materialized target arrays may be cached on disk.
    """

    PARAMETER_COLUMNS = ParameterDatasetBuilder.PARAMETER_COLUMNS
    FREQUENCY_COLUMN = "FREQ_GHZ"
    SIMULATION_COLUMN = ParameterDatasetBuilder.SIMULATION_COLUMN
    TOUCHSTONE_COLUMN = ParameterDatasetBuilder.TOUCHSTONE_COLUMN
    SPLIT_COLUMN = ParameterDatasetBuilder.SPLIT_COLUMN
    COLUMNS = (
        *PARAMETER_COLUMNS,
        FREQUENCY_COLUMN,
        SIMULATION_COLUMN,
        TOUCHSTONE_COLUMN,
        SPLIT_COLUMN,
    )
    FEATURE_COLUMNS = (*PARAMETER_COLUMNS, FREQUENCY_COLUMN)
    PROGRESS_MODE_ENV = "SPARAM_SURROGATE_PROGRESS"
    FINAL_PROGRESS_MODE = "final"
    CACHE_DIR = PROJECT_ROOT / "data" / "processed"
    FREQUENCY_EXPANDED_FILENAME = "frequency_expanded_dataset.csv"

    def __init__(
        self,
        dataframe: pd.DataFrame,
        split_type: str = "train",
        target_loader: Callable[[np.ndarray, Mapping[str, Any]], Any] | None = None,
        cache: bool = False,
        source_csv: Path | str | None = None,
    ) -> None:
        """
        Create a split-specific lazy dataset view.
        """
        self._split_type = str(split_type)
        self._dataframe = self._filtered_dataframe(dataframe)
        self._cache = bool(cache)
        self._source_csv = Path(
            source_csv or self.CACHE_DIR / self.FREQUENCY_EXPANDED_FILENAME
        )
        if target_loader is None:
            self._target_loader = None
        else:
            self.set_target_loader(target_loader)

    @classmethod
    def from_frequency_expanded_csv(
        cls,
        frequency_expanded_csv: Path | str,
        target_loader: Callable[[np.ndarray, Mapping[str, Any]], Any] | None = None,
        cache: bool = False,
    ) -> tuple["PointwiseDataset", "PointwiseDataset", "PointwiseDataset"]:
        """
        Build train, validation, and test views from a frequency-expanded CSV.

        Feature order is all design parameters followed by ``FREQ_GHZ``.
        """
        source_csv = Path(frequency_expanded_csv)
        expanded = pd.read_csv(source_csv)
        return (
            cls(expanded, "train", target_loader, cache, source_csv),
            cls(expanded, "val", target_loader, cache, source_csv),
            cls(expanded, "test", target_loader, cache, source_csv),
        )

    @classmethod
    def build_frequency_expanded_csv(
        cls,
        split_parameter_csv: Path | str,
        output_csv: Path | str,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Build or load the frequency-expanded point-wise CSV.
        """
        split_parameter_csv = Path(split_parameter_csv)
        output_csv = Path(output_csv)
        if (
            output_csv.is_file()
            and not force
            and output_csv.stat().st_mtime_ns
            >= split_parameter_csv.stat().st_mtime_ns
        ):
            return pd.read_csv(output_csv)

        split_parameters = pd.read_csv(split_parameter_csv)
        touchstone_path = Path(str(split_parameters.iloc[0][cls.TOUCHSTONE_COLUMN]))
        if not touchstone_path.is_absolute():
            touchstone_path = PROJECT_ROOT / touchstone_path
        frequencies_ghz = np.asarray(
            rf.Network(str(touchstone_path.resolve())).f,
            dtype=float,
        ) / 1e9

        rows: list[dict[str, object]] = []
        for _, design in split_parameters.iterrows():
            for frequency_ghz in frequencies_ghz:
                row: dict[str, object] = {
                    column: float(design[column]) for column in cls.PARAMETER_COLUMNS
                }
                row[cls.FREQUENCY_COLUMN] = float(frequency_ghz)
                row[cls.SIMULATION_COLUMN] = int(design[cls.SIMULATION_COLUMN])
                row[cls.TOUCHSTONE_COLUMN] = str(design[cls.TOUCHSTONE_COLUMN])
                row[cls.SPLIT_COLUMN] = str(design[cls.SPLIT_COLUMN])
                rows.append(row)

        expanded = pd.DataFrame(rows, columns=cls.COLUMNS)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        expanded.to_csv(output_csv, index=False)
        return expanded

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
        return self.FEATURE_COLUMNS

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
        return self._dataframe.loc[:, self.FEATURE_COLUMNS].to_numpy(dtype=float)

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
        filtered = dataframe.loc[
            dataframe["SPLIT_TYPE"].astype(str) == self._split_type
        ].copy()
        return filtered.reset_index(drop=True)
