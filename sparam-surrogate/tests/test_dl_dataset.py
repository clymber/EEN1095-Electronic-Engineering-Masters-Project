"""
Tests for :class:`DLDataset` split views.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from sparam_surrogate.data import DLDataset


def _cleaned_frame() -> pd.DataFrame:
    """
    Return a minimal cleaned dataframe containing two splits.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with feature, split, and Touchstone metadata columns.
    """
    return pd.DataFrame(
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


class FakeMapFunc:
    """
    Small callable used to test TensorFlow mapping without Touchstone I/O.
    """

    target_shape = (1,)

    def __call__(
        self,
        features: np.ndarray,
        row_metadata: Mapping[str, Any],
    ) -> np.ndarray:
        """
        Return a deterministic target derived from features and metadata.

        Parameters
        ----------
        features:
            Feature vector from ``DLDataset``.
        row_metadata:
            Metadata containing ``SIMU_INDEX`` and ``FREQ_GHZ``.

        Returns
        -------
        np.ndarray
            Single-value target for smoke testing.
        """
        value = float(features[0]) + float(row_metadata["FREQ_GHZ"])
        return np.asarray([value], dtype=np.float32)


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
    cleaned.loc[len(cleaned)] = {
        "EPS": 5.0,
        "FREQ_GHZ": 3.0,
        "SIMU_INDEX": 2,
        "TOUCHSTONE_REL_PATH": "raw/variation/simu_2.s2p",
        "SPLIT_TYPE": "test",
    }
    cleaned.to_csv(cleaned_path, index=False)
    return cleaned_path


class TestDLDataset:
    """
    Unit tests for cleaned-data split views.
    """

    def test_filters_split_and_exposes_features_and_metadata(self) -> None:
        """
        A split view preserves feature order and aligned row metadata.
        """
        dataset = DLDataset(_cleaned_frame(), ["EPS", "FREQ_GHZ"], "train")

        assert len(dataset) == 2
        assert dataset.feature_columns == ("EPS", "FREQ_GHZ")
        np.testing.assert_allclose(dataset.features, [[3.0, 1.0], [3.0, 2.0]])
        assert dataset.row_metadata["SIMU_INDEX"].tolist() == [0, 0]
        assert dataset.row_metadata["TOUCHSTONE_REL_PATH"].tolist() == [
            "raw/variation/simu_0.s2p",
            "raw/variation/simu_0.s2p",
        ]

    def test_from_cleaned_csv_returns_train_val_test_splits(
        self,
        tmp_path,
    ) -> None:
        """
        A cleaned CSV can be reopened as standard split-specific datasets.
        """
        cleaned_path = _cleaned_csv(tmp_path)

        train_set, val_set, test_set = DLDataset.from_cleaned_csv(
            cleaned_path,
            feature_columns=["EPS", "FREQ_GHZ"],
        )

        assert train_set.split_type == "train"
        assert val_set.split_type == "val"
        assert test_set.split_type == "test"
        assert len(train_set) == 2
        assert len(val_set) == 1
        assert len(test_set) == 1

    def test_from_cleaned_csv_propagates_loader_to_all_splits(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Every split uses the loader supplied to the CSV factory.
        """
        loader = FakeMetadataLoader()

        datasets = DLDataset.from_cleaned_csv(
            _cleaned_csv(tmp_path),
            feature_columns=["EPS", "FREQ_GHZ"],
            target_loader=loader,
            cache=False,
        )

        loaded = [dataset.load_targets() for dataset in datasets]
        np.testing.assert_allclose(loaded[0], [[1.0], [2.0]])
        np.testing.assert_allclose(loaded[1], [[1.0]])
        np.testing.assert_allclose(loaded[2], [[3.0]])
        assert loader.calls == 4

    @pytest.mark.parametrize("operation", ["load_targets", "targets", "tf_dataset"])
    def test_target_operations_reject_missing_loader(self, operation: str) -> None:
        """
        Target operations fail clearly when the dataset has no loader.
        """
        dataset = DLDataset(_cleaned_frame(), ["EPS", "FREQ_GHZ"], "train")

        with pytest.raises(RuntimeError, match="target loader"):
            if operation == "load_targets":
                dataset.load_targets()
            elif operation == "targets":
                _ = dataset.targets
            else:
                dataset.to_tf_dataset(batch_size=2, prefetch=False)

    def test_set_target_loader_attaches_and_replaces_loader(self) -> None:
        """
        The setter controls which loader eager target loading uses.
        """
        dataset = DLDataset(_cleaned_frame(), ["EPS", "FREQ_GHZ"], "train")
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
        dataset = DLDataset(
            _cleaned_frame(),
            ["EPS", "FREQ_GHZ"],
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
        monkeypatch.setenv(DLDataset.PROGRESS_MODE_ENV, DLDataset.FINAL_PROGRESS_MODE)
        dataset = DLDataset(
            _cleaned_frame(),
            ["EPS", "FREQ_GHZ"],
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
        dataset = DLDataset(
            _cleaned_frame(),
            ["EPS", "FREQ_GHZ"],
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
        monkeypatch.setattr(DLDataset, "CACHE_DIR", cache_dir, raising=False)
        loader = FakeMetadataLoader()
        dataset = DLDataset(
            _cleaned_frame(),
            ["EPS", "FREQ_GHZ"],
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
        monkeypatch.setattr(DLDataset, "CACHE_DIR", cache_dir, raising=False)
        source_csv = tmp_path / "cleaned.csv"
        source_csv.touch()
        loader = FakeMetadataLoader()
        dataset = DLDataset(
            _cleaned_frame(),
            ["EPS", "FREQ_GHZ"],
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
        monkeypatch.setattr(DLDataset, "CACHE_DIR", cache_dir, raising=False)
        source_csv = tmp_path / "cleaned.csv"
        source_csv.touch()

        scalar_dataset = DLDataset(
            _cleaned_frame(),
            ["EPS", "FREQ_GHZ"],
            "train",
            target_loader=FakeMetadataLoader(mode="scalar"),
            cache=True,
            source_csv=source_csv,
        )
        vector_dataset = DLDataset(
            _cleaned_frame(),
            ["EPS", "FREQ_GHZ"],
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
        monkeypatch.setattr(DLDataset, "CACHE_DIR", cache_dir, raising=False)
        source_csv = tmp_path / "cleaned.csv"
        source_csv.touch()
        cache_path = cache_dir / "vector_db_train.npz"
        np.savez(cache_path, targets=np.asarray([[11.0], [22.0]]))
        timestamp_ns = 1_700_000_000_000_000_000
        os.utime(source_csv, ns=(timestamp_ns, timestamp_ns))
        os.utime(cache_path, ns=(timestamp_ns + 1_000_000, timestamp_ns + 1_000_000))
        loader = FakeMetadataLoader()
        dataset = DLDataset(
            _cleaned_frame(),
            ["EPS", "FREQ_GHZ"],
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
        monkeypatch.setattr(DLDataset, "CACHE_DIR", cache_dir, raising=False)
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
        dataset = DLDataset(
            _cleaned_frame(),
            ["EPS", "FREQ_GHZ"],
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
        monkeypatch.setattr(DLDataset, "CACHE_DIR", cache_dir, raising=False)
        source_csv = tmp_path / "cleaned.csv"
        source_csv.touch()

        dataset = DLDataset(
            _cleaned_frame(),
            ["EPS", "FREQ_GHZ"],
            "train",
            target_loader=lambda _features, _metadata: np.asarray([1.0]),
            cache=True,
            source_csv=source_csv,
        )

        with pytest.raises(ValueError, match="mode.*representation"):
            _ = dataset.targets

    def test_rejects_missing_required_columns(self) -> None:
        """
        Missing metadata columns are reported clearly.
        """
        frame = _cleaned_frame().drop(columns=["TOUCHSTONE_REL_PATH"])

        with pytest.raises(ValueError, match="TOUCHSTONE_REL_PATH"):
            DLDataset(frame, ["EPS", "FREQ_GHZ"], "train")

    def test_to_tf_dataset_maps_fake_callable_when_tensorflow_is_available(
        self,
    ) -> None:
        """
        TensorFlow mapping yields feature and target batches.
        """
        pytest.importorskip("tensorflow")
        dataset = DLDataset(
            _cleaned_frame(),
            ["EPS", "FREQ_GHZ"],
            "train",
            target_loader=FakeMapFunc(),
        )

        tf_dataset = dataset.to_tf_dataset(
            batch_size=2,
            shuffle=False,
            prefetch=False,
        )
        features, targets = next(iter(tf_dataset))

        np.testing.assert_allclose(features.numpy(), [[3.0, 1.0], [3.0, 2.0]])
        np.testing.assert_allclose(targets.numpy(), [[4.0], [5.0]])

    def test_to_tf_dataset_rejects_complex_loader(self) -> None:
        """
        TensorFlow loading rejects unsupported complex targets clearly.
        """
        pytest.importorskip("tensorflow")
        dataset = DLDataset(
            _cleaned_frame(),
            ["EPS", "FREQ_GHZ"],
            "train",
            target_loader=FakeComplexLoader(),
        )

        with pytest.raises(ValueError, match="complex"):
            dataset.to_tf_dataset(batch_size=2, prefetch=False)
