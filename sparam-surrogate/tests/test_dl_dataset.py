"""
Tests for :class:`DLDataset` split views.
"""

from collections.abc import Mapping
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

    def __call__(
        self,
        features: np.ndarray,
        row_metadata: Mapping[str, Any],
    ) -> np.ndarray:
        """
        Return a deterministic target from metadata only.
        """
        assert len(features) == 0
        return np.asarray([row_metadata["FREQ_GHZ"]], dtype=np.float32)


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

    def test_load_targets_materializes_callable_targets(self) -> None:
        """
        Eager target loading reuses the same feature and metadata alignment.
        """
        dataset = DLDataset(_cleaned_frame(), ["EPS", "FREQ_GHZ"], "train")

        targets = dataset.load_targets(FakeMetadataLoader())

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
        dataset = DLDataset(_cleaned_frame(), ["EPS", "FREQ_GHZ"], "train")

        targets = dataset.load_targets(FakeMetadataLoader())

        np.testing.assert_allclose(targets, [[1.0], [2.0]])
        captured = capsys.readouterr()
        assert "Loading train targets" in captured.out
        assert "2/2" in captured.out

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
        dataset = DLDataset(_cleaned_frame(), ["EPS", "FREQ_GHZ"], "train")

        tf_dataset = dataset.to_tf_dataset(
            map_func=FakeMapFunc(),
            batch_size=2,
            shuffle=False,
            prefetch=False,
        )
        features, targets = next(iter(tf_dataset))

        np.testing.assert_allclose(features.numpy(), [[3.0, 1.0], [3.0, 2.0]])
        np.testing.assert_allclose(targets.numpy(), [[4.0], [5.0]])
