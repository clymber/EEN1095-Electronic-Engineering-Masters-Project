"""
Tests for the consolidated whole-curve dataset cache.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from sparam_surrogate.data import CurveDataset, ParameterDatasetBuilder


class FakeCurveLoader:
    """
    Return deterministic whole-curve targets from design metadata.
    """

    mode = "vector"
    representation = "il"

    def __init__(
        self,
        *,
        target_names: tuple[str, ...] = ("IL_S2_1_DB", "IL_S1_2_DB"),
        grids: Mapping[int, np.ndarray] | None = None,
    ) -> None:
        """
        Configure target names and optional simulation-specific grids.
        """
        self.target_names = target_names
        self.grids = dict(grids or {})
        self.calls = 0

    def load_curve(
        self,
        row_metadata: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return one deterministic curve for the requested simulation.
        """
        self.calls += 1
        simulation_index = int(row_metadata["SIMU_INDEX"])
        frequencies = np.asarray(
            self.grids.get(simulation_index, [1.0, 2.0, 3.0]),
            dtype=float,
        )
        targets = np.column_stack(
            [
                frequencies + simulation_index,
                2.0 * frequencies + simulation_index,
            ]
        )
        return frequencies, targets[:, : len(self.target_names)]


def _cleaned_frame() -> pd.DataFrame:
    """
    Return a design-level dataframe containing all standard splits.
    """
    rows: list[dict[str, object]] = []
    for simulation_index, split_type in (
        (10, "train"),
        (11, "train"),
        (20, "val"),
        (30, "test"),
    ):
        row: dict[str, object] = {
            column: float(simulation_index + offset)
            for offset, column in enumerate(
                ParameterDatasetBuilder.PARAMETER_COLUMNS
            )
        }
        row.update(
            {
                "SIMU_INDEX": simulation_index,
                "TOUCHSTONE_REL_PATH": (
                    f"data/raw/test/variation/simu_{simulation_index}.s2p"
                ),
                "SPLIT_TYPE": split_type,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=ParameterDatasetBuilder.SPLIT_COLUMNS)


def _write_cleaned_csv(tmp_path: Path) -> Path:
    """
    Write the standard design-level dataframe to a temporary CSV.
    """
    source_csv = tmp_path / "cleaned_splits_parameter.csv"
    _cleaned_frame().to_csv(source_csv, index=False)
    return source_csv


def _write_cache(
    cache_path: Path,
    *,
    simulation_indices: np.ndarray | None = None,
    split_labels: np.ndarray | None = None,
    target_names: np.ndarray | None = None,
    targets: np.ndarray | None = None,
) -> None:
    """
    Write a valid consolidated cache with optional metadata overrides.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        targets=(
            np.asarray(targets)
            if targets is not None
            else np.full((4, 3, 2), 99.0)
        ),
        frequencies_ghz=np.asarray([1.0, 2.0, 3.0]),
        simulation_indices=(
            np.asarray(simulation_indices)
            if simulation_indices is not None
            else np.asarray([10, 11, 20, 30])
        ),
        split_labels=(
            np.asarray(split_labels)
            if split_labels is not None
            else np.asarray(["train", "train", "val", "test"])
        ),
        target_names=(
            np.asarray(target_names)
            if target_names is not None
            else np.asarray(["IL_S2_1_DB", "IL_S1_2_DB"])
        ),
    )


def _set_cache_times(
    source_csv: Path,
    cache_path: Path,
    cache_offset_ns: int,
) -> None:
    """
    Set deterministic source and cache modification times.
    """
    timestamp_ns = 1_700_000_000_000_000_000
    os.utime(source_csv, ns=(timestamp_ns, timestamp_ns))
    os.utime(
        cache_path,
        ns=(timestamp_ns + cache_offset_ns, timestamp_ns + cache_offset_ns),
    )


def _load(
    source_csv: Path,
    loader: FakeCurveLoader,
    cache_dir: Path,
) -> tuple[CurveDataset, CurveDataset, CurveDataset]:
    """
    Load the standard three curve splits with caching enabled.
    """
    return CurveDataset.from_cleaned_splits_csv(
        source_csv,
        curve_loader=loader,
        cache=True,
        cache_dir=cache_dir,
        progress=False,
    )


def test_factory_preserves_split_feature_and_target_order(tmp_path: Path) -> None:
    """
    The factory returns aligned eager train, validation, and test views.
    """
    loader = FakeCurveLoader()
    datasets = CurveDataset.from_cleaned_splits_csv(
        _write_cleaned_csv(tmp_path),
        curve_loader=loader,
        cache=False,
        progress=False,
    )

    assert tuple(dataset.split_type for dataset in datasets) == (
        "train",
        "val",
        "test",
    )
    assert tuple(len(dataset) for dataset in datasets) == (2, 1, 1)
    assert datasets[0].feature_columns == ParameterDatasetBuilder.PARAMETER_COLUMNS
    assert datasets[0].features.shape == (2, 10)
    assert datasets[0].simulation_indices.tolist() == [10, 11]
    assert datasets[0].targets.shape == (2, 3, 2)
    assert datasets[0].targets.dtype == np.float32
    assert all(dataset.cache_status == "disabled" for dataset in datasets)
    assert loader.calls == 4


def test_missing_cache_builds_one_consolidated_npz(tmp_path: Path) -> None:
    """
    A cold load writes one validated cache covering every design and split.
    """
    source_csv = _write_cleaned_csv(tmp_path)
    loader = FakeCurveLoader()
    datasets = _load(source_csv, loader, tmp_path / "cache")
    cache_path = datasets[0].cache_path

    assert loader.calls == 4
    assert all(dataset.cache_status == "rebuilt" for dataset in datasets)
    assert all(dataset.cache_path == cache_path for dataset in datasets)
    assert cache_path.name == "vector_il_curve_dataset.npz"
    with np.load(cache_path, allow_pickle=False) as cached:
        assert set(cached.files) == {
            "targets",
            "frequencies_ghz",
            "simulation_indices",
            "split_labels",
            "target_names",
        }
        assert cached["targets"].shape == (4, 3, 2)
        np.testing.assert_array_equal(
            cached["simulation_indices"],
            [10, 11, 20, 30],
        )
        np.testing.assert_array_equal(
            cached["split_labels"],
            ["train", "train", "val", "test"],
        )


def test_newer_compatible_cache_is_reused(tmp_path: Path) -> None:
    """
    A compatible consolidated cache avoids every Touchstone loader call.
    """
    source_csv = _write_cleaned_csv(tmp_path)
    cache_path = tmp_path / "cache" / "vector_il_curve_dataset.npz"
    _write_cache(cache_path)
    _set_cache_times(source_csv, cache_path, 1_000_000)
    loader = FakeCurveLoader()

    train_set, val_set, test_set = _load(
        source_csv,
        loader,
        cache_path.parent,
    )

    assert loader.calls == 0
    assert all(
        dataset.cache_status == "hit"
        for dataset in (train_set, val_set, test_set)
    )
    np.testing.assert_allclose(train_set.targets, 99.0)
    np.testing.assert_allclose(val_set.targets, 99.0)
    np.testing.assert_allclose(test_set.targets, 99.0)


@pytest.mark.parametrize("cache_offset_ns", [-1_000_000, 0])
def test_non_newer_cache_is_rebuilt(
    tmp_path: Path,
    cache_offset_ns: int,
) -> None:
    """
    Older and equal-time caches are rebuilt once for all splits.
    """
    source_csv = _write_cleaned_csv(tmp_path)
    cache_path = tmp_path / "cache" / "vector_il_curve_dataset.npz"
    _write_cache(cache_path)
    _set_cache_times(source_csv, cache_path, cache_offset_ns)
    loader = FakeCurveLoader()

    datasets = _load(source_csv, loader, cache_path.parent)

    assert loader.calls == 4
    assert all(dataset.cache_status == "rebuilt" for dataset in datasets)
    assert not np.all(datasets[0].targets == 99.0)


@pytest.mark.parametrize(
    ("simulation_indices", "split_labels", "target_names"),
    [
        (np.asarray([11, 10, 20, 30]), None, None),
        (None, np.asarray(["train", "val", "train", "test"]), None),
        (None, None, np.asarray(["IL_S1_2_DB", "IL_S2_1_DB"])),
    ],
)
def test_incompatible_cache_metadata_forces_rebuild(
    tmp_path: Path,
    simulation_indices: np.ndarray | None,
    split_labels: np.ndarray | None,
    target_names: np.ndarray | None,
) -> None:
    """
    Changed design, split, or target order invalidates the shared cache.
    """
    source_csv = _write_cleaned_csv(tmp_path)
    cache_path = tmp_path / "cache" / "vector_il_curve_dataset.npz"
    _write_cache(
        cache_path,
        simulation_indices=simulation_indices,
        split_labels=split_labels,
        target_names=target_names,
    )
    _set_cache_times(source_csv, cache_path, 1_000_000)
    loader = FakeCurveLoader()

    datasets = _load(source_csv, loader, cache_path.parent)

    assert loader.calls == 4
    assert all(dataset.cache_status == "rebuilt" for dataset in datasets)


def test_malformed_fresh_cache_forces_rebuild(tmp_path: Path) -> None:
    """
    A fresh cache missing required arrays is treated as a cache miss.
    """
    source_csv = _write_cleaned_csv(tmp_path)
    cache_path = tmp_path / "cache" / "vector_il_curve_dataset.npz"
    cache_path.parent.mkdir()
    np.savez(cache_path, targets=np.full((4, 3, 2), 99.0))
    _set_cache_times(source_csv, cache_path, 1_000_000)
    loader = FakeCurveLoader()

    datasets = _load(source_csv, loader, cache_path.parent)

    assert loader.calls == 4
    assert all(dataset.cache_status == "rebuilt" for dataset in datasets)


def test_mismatched_frequency_grids_fail_before_cache_write(
    tmp_path: Path,
) -> None:
    """
    Inconsistent design grids raise without leaving a partial cache.
    """
    source_csv = _write_cleaned_csv(tmp_path)
    cache_dir = tmp_path / "cache"
    loader = FakeCurveLoader(
        grids={20: np.asarray([1.0, 2.1, 3.0])}
    )

    with pytest.raises(ValueError, match="frequency grid"):
        _load(source_csv, loader, cache_dir)

    assert not (cache_dir / "vector_il_curve_dataset.npz").exists()


def test_curve_cache_name_cannot_collide_with_pointwise_cache(
    tmp_path: Path,
) -> None:
    """
    The curve-dataset qualifier keeps the cache separate from point-wise data.
    """
    train_set = _load(
        _write_cleaned_csv(tmp_path),
        FakeCurveLoader(),
        tmp_path / "cache",
    )[0]

    assert train_set.cache_path.name == "vector_il_curve_dataset.npz"
    assert train_set.cache_path.name != "vector_il_train.npz"
