"""
Tests for design-level split assignment by :class:`MLDatasetBuilder`.
"""

from pathlib import Path

import pandas as pd
import pytest

from sparam_surrogate.data import DLDataset, MLDatasetBuilder, RawData


def _write_touchstone(path: Path) -> None:
    """
    Write a two-frequency two-port Touchstone fixture.
    """
    path.write_text(
        "# GHz S RI R 50\n"
        "1 0.1 0 0.5 0 0.2 0 0.3 0\n"
        "2 0.1 0 0.4 0 0.2 0 0.3 0\n",
        encoding="utf-8",
    )


def _raw_dataset(tmp_path: Path, n_designs: int = 6) -> RawData:
    """
    Create a raw-data fixture with complete parameter and Touchstone records.
    """
    raw_path = tmp_path / "raw"
    variation_path = raw_path / "variation"
    variation_path.mkdir(parents=True)
    rows = []
    for index in range(n_designs):
        rows.append(
            {
                "SIMU_INDEX": index,
                "EPS": 3.0 + index,
                "TAND": 0.01,
                "PITCH": 1.0,
                "TRACE_LEN": 10.0,
                "START": 2.0,
                "VIAR": 0.1,
                "ANTIPADR": 0.2,
                "TDIEL": 0.3,
                "DISTTL": 0.4,
                "TLWIDTH": 0.05,
            }
        )
        _write_touchstone(variation_path / f"simu_{index}.s2p")
    pd.DataFrame(rows).to_csv(raw_path / "parameter.csv", index=False)
    return RawData(raw_path, nports=2)


class TestMLDatasetBuilderSplit:
    """
    Unit tests for split creation from the cleaned dataframe.
    """

    def test_split_returns_datasets_and_persists_split_labels(self, tmp_path: Path) -> None:
        """
        Split labels are assigned by design and written back to the cleaned CSV.
        """
        builder = MLDatasetBuilder(_raw_dataset(tmp_path), tmp_path / "processed")

        train_set, val_set, test_set = builder.split(
            val_fraction=1 / 6,
            test_fraction=1 / 6,
            seed=123,
            force=True,
        )

        assert isinstance(train_set, DLDataset)
        assert isinstance(val_set, DLDataset)
        assert isinstance(test_set, DLDataset)
        assert len(train_set) == 8
        assert len(val_set) == 2
        assert len(test_set) == 2

        saved = pd.read_csv(builder.cleaned_path)
        assert set(saved["SPLIT_TYPE"]) == {"train", "val", "test"}
        for simulation_index, group in saved.groupby("SIMU_INDEX"):
            assert len(set(group["SPLIT_TYPE"])) == 1, simulation_index

    def test_split_is_deterministic_for_fixed_seed(self, tmp_path: Path) -> None:
        """
        Re-running split with the same seed produces identical labels.
        """
        builder = MLDatasetBuilder(_raw_dataset(tmp_path), tmp_path / "processed")

        first = builder.split(
            val_fraction=1 / 6,
            test_fraction=1 / 6,
            seed=42,
            force=True,
        )
        first_labels = first[0].dataframe[["SIMU_INDEX", "SPLIT_TYPE"]]
        second = builder.split(
            val_fraction=1 / 6,
            test_fraction=1 / 6,
            seed=42,
            force=False,
        )
        second_labels = second[0].dataframe[["SIMU_INDEX", "SPLIT_TYPE"]]

        pd.testing.assert_frame_equal(first_labels, second_labels)
