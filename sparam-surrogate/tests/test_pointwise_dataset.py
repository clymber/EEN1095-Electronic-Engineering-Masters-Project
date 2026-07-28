"""
Tests for :class:`PointwiseDataset` split views.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from sparam_surrogate.data import PointwiseDataset


def _cleaned_frame() -> pd.DataFrame:
    """
    Return a minimal cleaned dataframe containing two splits.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with feature, split, and Touchstone metadata columns.
    """
    values = {
        column: [1.0, 1.0, 2.0]
        for column in PointwiseDataset.PARAMETER_COLUMNS
    }
    values.update(
        {
            "EPS": [3.0, 3.0, 4.0],
            "FREQ_GHZ": [1.0, 2.0, 1.0],
            "SIMU_INDEX": [0, 0, 1],
            "TOUCHSTONE_REL_PATH": [
                "raw/variation/simu_0.s2p",
                "raw/variation/simu_0.s2p",
                "raw/variation/simu_1.s2p",
            ],
            "SPLIT_TYPE": ["train", "train", "val"],
        }
    )
    return pd.DataFrame(values)


class FakeMetadataLoader:
    """
    Small target loader used to test eager metadata-driven loading.
    """

    mode = "vector"
    representation = "db"
    target_names = ("S1_1_DB",)

    def __init__(self, multiplier: float = 1.0, mode: str = "vector") -> None:
        self.multiplier = float(multiplier)
        self.mode = mode
        self.calls = 0

    def __call__(
        self,
        features: np.ndarray,
        row_metadata: Mapping[str, Any],
    ) -> np.ndarray:
        """
        Return a deterministic target from metadata only.
        """
        self.calls += 1
        assert len(features) == 0
        return np.asarray(
            [self.multiplier * float(row_metadata["FREQ_GHZ"])],
            dtype=np.float32,
        )


class FakeComplexLoader(FakeMetadataLoader):
    """
    Return complex targets to verify eager dtype preservation.
    """

    representation = "complex"
    target_names = ("S1_1",)

    def __call__(
        self,
        features: np.ndarray,
        row_metadata: Mapping[str, Any],
    ) -> np.ndarray:
        self.calls += 1
        assert len(features) == 0
        frequency = float(row_metadata["FREQ_GHZ"])
        return np.asarray([frequency + 2j * frequency], dtype=np.complex128)


def _cleaned_csv(tmp_path: Path) -> Path:
    """
    Write a cleaned CSV containing all three standard splits.
    """
    cleaned_path = tmp_path / "cleaned.csv"
    cleaned = _cleaned_frame()
    test_row = {
        column: 3.0 for column in PointwiseDataset.PARAMETER_COLUMNS
    }
    test_row.update(
        {
            "EPS": 5.0,
            "FREQ_GHZ": 3.0,
            "SIMU_INDEX": 2,
            "TOUCHSTONE_REL_PATH": "raw/variation/simu_2.s2p",
            "SPLIT_TYPE": "test",
        }
    )
    cleaned.loc[len(cleaned)] = test_row
    cleaned.to_csv(cleaned_path, index=False)
    return cleaned_path


class TestPointwiseDataset:
    """
    Unit tests for cleaned-data split views.
    """

    def test_uses_fixed_feature_columns(self) -> None:
        """
        Direct construction uses the fixed standard feature order.
        """
        row = {
            column: 1.0 for column in PointwiseDataset.FEATURE_COLUMNS
        }
        row.update(
            {
                "SIMU_INDEX": 0,
                "TOUCHSTONE_REL_PATH": "raw/variation/simu_0.s2p",
                "SPLIT_TYPE": "train",
            }
        )

        dataset = PointwiseDataset(
            pd.DataFrame([row]),
            split_type="train",
        )

        assert dataset.feature_columns == PointwiseDataset.FEATURE_COLUMNS
        assert dataset.features.shape == (
            1,
            len(PointwiseDataset.FEATURE_COLUMNS),
        )

    def test_filters_split_and_exposes_features_and_metadata(self) -> None:
        """
        A split view preserves feature order and aligned row metadata.
        """
        dataset = PointwiseDataset(_cleaned_frame(), "train")

        assert len(dataset) == 2
        assert dataset.feature_columns == PointwiseDataset.FEATURE_COLUMNS
        np.testing.assert_allclose(
            dataset.features[:, [0, -1]],
            [[3.0, 1.0], [3.0, 2.0]],
        )
        assert dataset.row_metadata["SIMU_INDEX"].tolist() == [0, 0]
        assert dataset.row_metadata["TOUCHSTONE_REL_PATH"].tolist() == [
            "raw/variation/simu_0.s2p",
            "raw/variation/simu_0.s2p",
        ]

    def test_from_frequency_expanded_csv_returns_train_val_test_splits(
        self,
        tmp_path,
    ) -> None:
        """
        A frequency-expanded CSV can be reopened as split-specific datasets.
        """
        cleaned_path = _cleaned_csv(tmp_path)

        train_set, val_set, test_set = (
            PointwiseDataset.from_frequency_expanded_csv(cleaned_path)
        )

        assert train_set.split_type == "train"
        assert val_set.split_type == "val"
        assert test_set.split_type == "test"
        assert len(train_set) == 2
        assert len(val_set) == 1
        assert len(test_set) == 1

    def test_from_frequency_expanded_csv_propagates_loader_to_all_splits(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Every split uses the loader supplied to the CSV factory.
        """
        loader = FakeMetadataLoader()

        datasets = PointwiseDataset.from_frequency_expanded_csv(
            _cleaned_csv(tmp_path),
            target_loader=loader,
            cache=False,
        )

        loaded = [dataset.load_targets() for dataset in datasets]
        np.testing.assert_allclose(loaded[0], [[1.0], [2.0]])
        np.testing.assert_allclose(loaded[1], [[1.0]])
        np.testing.assert_allclose(loaded[2], [[3.0]])
        assert loader.calls == 4

    @pytest.mark.parametrize("operation", ["load_targets", "targets"])
    def test_target_operations_reject_missing_loader(self, operation: str) -> None:
        """
        Target operations fail clearly when the dataset has no loader.
        """
        dataset = PointwiseDataset(_cleaned_frame(), "train")

        with pytest.raises(RuntimeError, match="target loader"):
            if operation == "load_targets":
                dataset.load_targets()
            else:
                _ = dataset.targets

    def test_set_target_loader_attaches_and_replaces_loader(self) -> None:
        """
        The setter controls which loader eager target loading uses.
        """
        dataset = PointwiseDataset(_cleaned_frame(), "train")
        first_loader = FakeMetadataLoader(multiplier=1.0)
        second_loader = FakeMetadataLoader(multiplier=10.0)

        dataset.set_target_loader(first_loader)
        np.testing.assert_allclose(dataset.load_targets(), [[1.0], [2.0]])

        dataset.set_target_loader(second_loader)
        np.testing.assert_allclose(dataset.load_targets(), [[10.0], [20.0]])

    def test_load_targets_materializes_callable_targets(self) -> None:
        """
        Eager target loading reuses the same feature and metadata alignment.
        """
        dataset = PointwiseDataset(
            _cleaned_frame(),
            "train",
            target_loader=FakeMetadataLoader(),
        )

        targets = dataset.load_targets()

        np.testing.assert_allclose(targets, [[1.0], [2.0]])

    def test_load_targets_can_print_final_progress_only(
        self,
        monkeypatch,
        capsys,
    ) -> None:
        """
        Final-only progress mode prints one completed progress summary.
        """
        monkeypatch.setenv(
            PointwiseDataset.PROGRESS_MODE_ENV,
            PointwiseDataset.FINAL_PROGRESS_MODE,
        )
        dataset = PointwiseDataset(
            _cleaned_frame(),
            "train",
            target_loader=FakeMetadataLoader(),
        )

        targets = dataset.load_targets()

        np.testing.assert_allclose(targets, [[1.0], [2.0]])
        captured = capsys.readouterr()
        assert "Loading train targets" in captured.out
        assert "2/2" in captured.out

    def test_load_targets_preserves_complex_dtype(self) -> None:
        """
        Direct eager loading does not discard complex target values.
        """
        dataset = PointwiseDataset(
            _cleaned_frame(),
            "train",
            target_loader=FakeComplexLoader(),
        )

        targets = dataset.load_targets()

        assert targets.dtype == np.complex128
        np.testing.assert_allclose(targets, [[1.0 + 2.0j], [2.0 + 4.0j]])

    def test_targets_without_cache_does_not_create_file(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """
        Disabled caching always loads directly and leaves no NPZ artifact.
        """
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(PointwiseDataset, "CACHE_DIR", cache_dir, raising=False)
        loader = FakeMetadataLoader()
        dataset = PointwiseDataset(
            _cleaned_frame(),
            "train",
            target_loader=loader,
            cache=False,
            source_csv=tmp_path / "cleaned.csv",
        )

        np.testing.assert_allclose(dataset.targets, [[1.0], [2.0]])

        assert loader.calls == 2
        assert not cache_dir.exists()

    def test_cache_miss_loads_and_saves_split_targets(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """
        A cold cache creates the expected target-only split NPZ.
        """
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(PointwiseDataset, "CACHE_DIR", cache_dir, raising=False)
        source_csv = tmp_path / "cleaned.csv"
        source_csv.touch()
        loader = FakeMetadataLoader()
        dataset = PointwiseDataset(
            _cleaned_frame(),
            "train",
            target_loader=loader,
            cache=True,
            source_csv=source_csv,
        )

        np.testing.assert_allclose(dataset.targets, [[1.0], [2.0]])

        cache_path = cache_dir / "vector_db_train.npz"
        assert loader.calls == 2
        assert cache_path.is_file()
        with np.load(cache_path, allow_pickle=False) as cached:
            assert cached.files == ["targets"]
            np.testing.assert_allclose(cached["targets"], [[1.0], [2.0]])

    def test_scalar_and_vector_loaders_use_separate_cache_files(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """
        Loader mode keeps scalar and vector experiment caches independent.
        """
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(PointwiseDataset, "CACHE_DIR", cache_dir, raising=False)
        source_csv = tmp_path / "cleaned.csv"
        source_csv.touch()

        scalar_dataset = PointwiseDataset(
            _cleaned_frame(),
            "train",
            target_loader=FakeMetadataLoader(mode="scalar"),
            cache=True,
            source_csv=source_csv,
        )
        vector_dataset = PointwiseDataset(
            _cleaned_frame(),
            "train",
            target_loader=FakeMetadataLoader(mode="vector"),
            cache=True,
            source_csv=source_csv,
        )

        _ = scalar_dataset.targets
        _ = vector_dataset.targets

        assert (cache_dir / "scalar_db_train.npz").is_file()
        assert (cache_dir / "vector_db_train.npz").is_file()

    def test_newer_cache_is_reused_without_calling_loader(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """
        A cache newer than its source CSV is a cache hit.
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(PointwiseDataset, "CACHE_DIR", cache_dir, raising=False)
        source_csv = tmp_path / "cleaned.csv"
        source_csv.touch()
        cache_path = cache_dir / "vector_db_train.npz"
        np.savez(cache_path, targets=np.asarray([[11.0], [22.0]]))
        timestamp_ns = 1_700_000_000_000_000_000
        os.utime(source_csv, ns=(timestamp_ns, timestamp_ns))
        os.utime(cache_path, ns=(timestamp_ns + 1_000_000, timestamp_ns + 1_000_000))
        loader = FakeMetadataLoader()
        dataset = PointwiseDataset(
            _cleaned_frame(),
            "train",
            target_loader=loader,
            cache=True,
            source_csv=source_csv,
        )

        np.testing.assert_allclose(dataset.targets, [[11.0], [22.0]])

        assert loader.calls == 0

    @pytest.mark.parametrize("cache_offset_ns", [-1_000_000, 0])
    def test_non_newer_cache_is_rebuilt(
        self,
        tmp_path: Path,
        monkeypatch,
        cache_offset_ns: int,
    ) -> None:
        """
        Older and equally old caches are rebuilt from the loader.
        """
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(PointwiseDataset, "CACHE_DIR", cache_dir, raising=False)
        source_csv = tmp_path / "cleaned.csv"
        source_csv.touch()
        cache_path = cache_dir / "vector_db_train.npz"
        np.savez(cache_path, targets=np.asarray([[11.0], [22.0]]))
        timestamp_ns = 1_700_000_000_000_000_000
        os.utime(source_csv, ns=(timestamp_ns, timestamp_ns))
        os.utime(
            cache_path,
            ns=(timestamp_ns + cache_offset_ns, timestamp_ns + cache_offset_ns),
        )
        loader = FakeMetadataLoader()
        dataset = PointwiseDataset(
            _cleaned_frame(),
            "train",
            target_loader=loader,
            cache=True,
            source_csv=source_csv,
        )

        np.testing.assert_allclose(dataset.targets, [[1.0], [2.0]])

        assert loader.calls == 2

    def test_cache_requires_loader_naming_attributes(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """
        A cached loader must expose values needed for its cache filename.
        """
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(PointwiseDataset, "CACHE_DIR", cache_dir, raising=False)
        source_csv = tmp_path / "cleaned.csv"
        source_csv.touch()

        dataset = PointwiseDataset(
            _cleaned_frame(),
            "train",
            target_loader=lambda _features, _metadata: np.asarray([1.0]),
            cache=True,
            source_csv=source_csv,
        )

        with pytest.raises(ValueError, match="mode.*representation"):
            _ = dataset.targets
