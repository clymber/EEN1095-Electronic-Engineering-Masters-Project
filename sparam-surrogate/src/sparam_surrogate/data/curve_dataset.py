"""
Design-level views backed by one validated whole-curve cache.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from .parameter_dataset_builder import ParameterDatasetBuilder

CacheStatus = Literal["hit", "rebuilt", "disabled"]


@dataclass(frozen=True)
class CurveDataset:
    """
    Hold one eager design split from a shared whole-curve dataset cache.
    """

    rows: pd.DataFrame
    split_type: str
    frequencies_ghz: np.ndarray
    targets: np.ndarray
    target_names: tuple[str, ...]
    cache_path: Path
    cache_status: CacheStatus

    PARAMETER_COLUMNS: ClassVar[tuple[str, ...]] = (
        ParameterDatasetBuilder.PARAMETER_COLUMNS
    )
    SIMULATION_COLUMN: ClassVar[str] = ParameterDatasetBuilder.SIMULATION_COLUMN
    TOUCHSTONE_COLUMN: ClassVar[str] = ParameterDatasetBuilder.TOUCHSTONE_COLUMN
    SPLIT_COLUMN: ClassVar[str] = ParameterDatasetBuilder.SPLIT_COLUMN
    REQUIRED_CACHE_ARRAYS: ClassVar[frozenset[str]] = frozenset(
        {
            "targets",
            "frequencies_ghz",
            "simulation_indices",
            "split_labels",
            "target_names",
        }
    )
    FREQUENCY_TOLERANCE_GHZ: ClassVar[float] = 1e-9
    feature_columns: ClassVar[tuple[str, ...]] = PARAMETER_COLUMNS

    @classmethod
    def from_cleaned_splits_csv(
        cls,
        cleaned_splits_csv: Path | str,
        curve_loader: Any,
        *,
        cache: bool = False,
        cache_dir: Path | str | None = None,
        progress: bool = True,
    ) -> tuple[CurveDataset, CurveDataset, CurveDataset]:
        """
        Load all curves once and return aligned train, validation, and test views.
        """
        cls._validate_loader(curve_loader)
        source_csv = Path(cleaned_splits_csv)
        dataframe = pd.read_csv(source_csv)
        cls._validate_dataframe(dataframe)
        destination = Path(cache_dir) if cache_dir else source_csv.parent
        cache_path = destination / cls._cache_name(curve_loader)

        materialized = (
            cls._read_cache(source_csv, cache_path, dataframe, curve_loader)
            if cache
            else None
        )
        if materialized is None:
            frequencies_ghz, targets = cls._materialize(
                dataframe,
                curve_loader,
                progress=progress,
            )
            cache_status: CacheStatus = "rebuilt" if cache else "disabled"
            if cache:
                cls._write_cache(
                    cache_path,
                    dataframe,
                    curve_loader,
                    frequencies_ghz,
                    targets,
                )
        else:
            frequencies_ghz, targets = materialized
            cache_status = "hit"

        split_labels = dataframe[cls.SPLIT_COLUMN].astype(str).to_numpy()
        target_names = tuple(str(name) for name in curve_loader.target_names)
        return tuple(
            cls(
                dataframe.loc[split_labels == split_type].reset_index(drop=True),
                split_type,
                frequencies_ghz,
                targets[split_labels == split_type],
                target_names,
                cache_path,
                cache_status,
            )
            for split_type in ("train", "val", "test")
        )  # type: ignore[return-value]

    def __len__(self) -> int:
        """
        Return the number of designs in this split.
        """
        return len(self.rows)

    @property
    def features(self) -> np.ndarray:
        """
        Return design features as a two-dimensional float array.
        """
        return self.rows.loc[:, self.feature_columns].to_numpy(dtype=float)

    @property
    def simulation_indices(self) -> np.ndarray:
        """
        Return simulation identifiers aligned with features and targets.
        """
        return self.rows[self.SIMULATION_COLUMN].to_numpy(dtype=np.int64)

    @classmethod
    def _cache_name(cls, curve_loader: Any) -> str:
        """
        Return a cache name that cannot collide with point-wise target caches.
        """
        return (
            f"{curve_loader.mode}_{curve_loader.representation}"
            "_curve_dataset.npz"
        )

    @classmethod
    def _read_cache(
        cls,
        source_csv: Path,
        cache_path: Path,
        dataframe: pd.DataFrame,
        curve_loader: Any,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Return compatible cached arrays or ``None`` for any cache miss.
        """
        if (
            not cache_path.is_file()
            or cache_path.stat().st_mtime_ns <= source_csv.stat().st_mtime_ns
        ):
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if not cls.REQUIRED_CACHE_ARRAYS.issubset(cached.files):
                    return None
                frequencies_ghz = np.asarray(cached["frequencies_ghz"])
                targets = np.asarray(cached["targets"])
                metadata = {
                    name: np.asarray(cached[name])
                    for name in (
                        "simulation_indices",
                        "split_labels",
                        "target_names",
                    )
                }
            cls._validate_arrays(
                dataframe,
                curve_loader,
                frequencies_ghz,
                targets,
                **metadata,
            )
        except (EOFError, KeyError, OSError, TypeError, ValueError):
            return None
        return frequencies_ghz, targets

    @classmethod
    def _materialize(
        cls,
        dataframe: pd.DataFrame,
        curve_loader: Any,
        *,
        progress: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load every design curve and require one common frequency grid.
        """
        records: Any = dataframe.loc[
            :,
            [cls.SIMULATION_COLUMN, cls.TOUCHSTONE_COLUMN],
        ].to_dict("records")
        if progress:
            records = tqdm(records, total=len(dataframe), desc="Loading curves")

        reference_grid: np.ndarray | None = None
        curves: list[np.ndarray] = []
        target_width = len(curve_loader.target_names)
        for row in records:
            frequencies_ghz, targets = curve_loader.load_curve(row)
            frequencies_ghz = np.asarray(frequencies_ghz, dtype=float)
            targets = np.asarray(targets, dtype=float)
            expected_shape = (len(frequencies_ghz), target_width)
            if (
                frequencies_ghz.ndim != 1
                or not len(frequencies_ghz)
                or targets.shape != expected_shape
                or not np.isfinite(frequencies_ghz).all()
                or not np.isfinite(targets).all()
            ):
                raise ValueError("Invalid whole-curve loader output.")
            if reference_grid is None:
                reference_grid = frequencies_ghz.copy()
            elif not np.allclose(
                reference_grid,
                frequencies_ghz,
                rtol=0.0,
                atol=float(
                    getattr(
                        curve_loader,
                        "FREQUENCY_TOLERANCE_GHZ",
                        cls.FREQUENCY_TOLERANCE_GHZ,
                    )
                ),
            ):
                raise ValueError(
                    "Touchstone frequency grid mismatch for "
                    f"SIMU_INDEX {row[cls.SIMULATION_COLUMN]}."
                )
            curves.append(targets)

        if reference_grid is None:
            raise ValueError("The cleaned design table contains no rows.")
        stacked = np.stack(curves)
        cls._validate_arrays(dataframe, curve_loader, reference_grid, stacked)
        return reference_grid, stacked

    @classmethod
    def _validate_arrays(
        cls,
        dataframe: pd.DataFrame,
        curve_loader: Any,
        frequencies_ghz: np.ndarray,
        targets: np.ndarray,
        *,
        simulation_indices: np.ndarray | None = None,
        split_labels: np.ndarray | None = None,
        target_names: np.ndarray | None = None,
    ) -> None:
        """
        Validate array shapes and optional cache-alignment metadata.
        """
        expected_shape = (
            len(dataframe),
            len(frequencies_ghz),
            len(curve_loader.target_names),
        )
        if (
            frequencies_ghz.ndim != 1
            or not len(frequencies_ghz)
            or targets.shape != expected_shape
            or not np.isfinite(frequencies_ghz).all()
            or not np.isfinite(targets).all()
        ):
            raise ValueError("Whole-curve cache arrays are incompatible.")

        expected_metadata = {
            "simulation_indices": dataframe[cls.SIMULATION_COLUMN].to_numpy(
                dtype=np.int64
            ),
            "split_labels": dataframe[cls.SPLIT_COLUMN].to_numpy(dtype=str),
            "target_names": np.asarray(curve_loader.target_names, dtype=str),
        }
        observed_metadata = {
            "simulation_indices": simulation_indices,
            "split_labels": split_labels,
            "target_names": target_names,
        }
        for name, observed in observed_metadata.items():
            if observed is not None and not np.array_equal(
                np.asarray(observed).astype(expected_metadata[name].dtype),
                expected_metadata[name],
            ):
                raise ValueError(f"Cached {name} do not match the cleaned data.")

    @classmethod
    def _write_cache(
        cls,
        cache_path: Path,
        dataframe: pd.DataFrame,
        curve_loader: Any,
        frequencies_ghz: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """
        Atomically replace the consolidated cache after validation.
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            dir=cache_path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with temporary_path.open("wb") as cache_file:
                np.savez(
                    cache_file,
                    targets=targets,
                    frequencies_ghz=frequencies_ghz,
                    simulation_indices=dataframe[
                        cls.SIMULATION_COLUMN
                    ].to_numpy(dtype=np.int64),
                    split_labels=dataframe[cls.SPLIT_COLUMN].to_numpy(dtype=str),
                    target_names=np.asarray(curve_loader.target_names, dtype=str),
                )
            os.replace(temporary_path, cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    def _validate_loader(cls, curve_loader: Any) -> None:
        """
        Require the minimal whole-curve loader interface.
        """
        required = ("mode", "representation", "target_names", "load_curve")
        missing = [name for name in required if not hasattr(curve_loader, name)]
        if missing or not callable(getattr(curve_loader, "load_curve", None)):
            names = ", ".join(missing or ["load_curve"])
            raise ValueError(f"Curve loader is missing required attributes: {names}.")

    @classmethod
    def _validate_dataframe(cls, dataframe: pd.DataFrame) -> None:
        """
        Require design features, identifiers, paths, and split labels.
        """
        required = {
            *cls.PARAMETER_COLUMNS,
            cls.SIMULATION_COLUMN,
            cls.TOUCHSTONE_COLUMN,
            cls.SPLIT_COLUMN,
        }
        missing = required.difference(dataframe.columns)
        if missing:
            raise ValueError(
                "Cleaned split dataframe is missing columns: "
                + ", ".join(sorted(missing))
            )
        split_labels = set(dataframe[cls.SPLIT_COLUMN].astype(str))
        if split_labels != {"train", "val", "test"}:
            raise ValueError(
                "Cleaned split dataframe must contain train, val, and test rows."
            )
