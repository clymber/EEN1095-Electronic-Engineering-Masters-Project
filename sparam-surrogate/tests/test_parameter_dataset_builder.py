"""
Tests for design-level parameter cleaning and split assignment.
"""

from pathlib import Path

import pandas as pd

from sparam_surrogate.data import ParameterDatasetBuilder, RawData

PARAMETER_VALUES = {
    "EPS": 3.8,
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


def _write_touchstone(path: Path) -> None:
    """
    Write a two-frequency, two-port Touchstone fixture.
    """
    path.write_text(
        "# GHz S RI R 50\n" "1 0.1 0 0.5 0 0.2 0 0.3 0\n" "2 0.1 0 0.4 0 0.2 0 0.3 0\n",
        encoding="utf-8",
    )


def _make_raw_dataset(
    tmp_path: Path,
    parameter_indices: list[int],
    touchstone_indices: list[int],
) -> RawData:
    """
    Create raw parameter and Touchstone fixtures.
    """
    raw_path = tmp_path / "raw"
    variation_path = raw_path / "variation"
    variation_path.mkdir(parents=True)
    rows = []
    for index in parameter_indices:
        row = dict(PARAMETER_VALUES)
        row["SIMU_INDEX"] = float(index)
        rows.append(row)
    pd.DataFrame(rows).to_csv(raw_path / "parameter.csv", index=False)
    for index in touchstone_indices:
        _write_touchstone(variation_path / f"simu_{index}.s2p")
    return RawData(raw_path, nports=2)


class TestParameterDatasetBuilder:
    """
    Tests for the one-row-per-design processed dataset.
    """

    def test_clean_keeps_only_parameters_with_touchstones(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Cleaning keeps matched parameters and adds one Touchstone path per design.
        """
        raw_data = _make_raw_dataset(
            tmp_path,
            parameter_indices=[2, 1, 3],
            touchstone_indices=[1, 2, 99],
        )
        builder = ParameterDatasetBuilder(
            raw_data,
            tmp_path / "processed" / "cleaned_splits_parameter.csv",
        )

        cleaned = builder.clean()

        assert tuple(cleaned.columns) == builder.CLEANED_COLUMNS
        assert cleaned["SIMU_INDEX"].tolist() == [2, 1]
        assert cleaned["SIMU_INDEX"].dtype == "int64"
        assert len(cleaned) == cleaned["SIMU_INDEX"].nunique()
        assert cleaned["TOUCHSTONE_REL_PATH"].str.len().gt(0).all()
        assert not builder.cleaned_splits_path.exists()

    def test_split_writes_expected_processed_csv_without_design_leakage(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Splitting writes one deterministic label per design.
        """
        raw_data = _make_raw_dataset(
            tmp_path,
            parameter_indices=list(range(6)),
            touchstone_indices=list(range(6)),
        )
        expected_path = (
            tmp_path / "processed" / "cleaned_splits_parameter.csv"
        )
        builder = ParameterDatasetBuilder(raw_data, expected_path)

        split = builder.split(
            builder.clean(),
            val_fraction=1 / 6,
            test_fraction=1 / 6,
            seed=123,
        )

        assert builder.cleaned_splits_path == expected_path
        assert expected_path.is_file()
        assert tuple(split.columns) == builder.SPLIT_COLUMNS
        assert set(split["SPLIT_TYPE"]) == {"train", "val", "test"}
        assert len(split) == split["SIMU_INDEX"].nunique()

        saved = pd.read_csv(expected_path)
        pd.testing.assert_frame_equal(saved, split)

    def test_load_normalizes_simulation_indices_to_integers(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Reused CSVs expose integer simulation indices.
        """
        raw_data = _make_raw_dataset(
            tmp_path,
            parameter_indices=list(range(6)),
            touchstone_indices=list(range(6)),
        )
        cleaned_splits_path = (
            tmp_path / "processed" / "cleaned_splits_parameter.csv"
        )
        builder = ParameterDatasetBuilder(raw_data, cleaned_splits_path)
        split = builder.split(
            builder.clean(),
            val_fraction=1 / 6,
            test_fraction=1 / 6,
            seed=123,
        )
        split["SIMU_INDEX"] = split["SIMU_INDEX"].astype(float)
        split.to_csv(cleaned_splits_path, index=False)

        loaded = builder.load()

        assert loaded["SIMU_INDEX"].dtype == "int64"

    def test_split_is_deterministic_for_fixed_seed(self, tmp_path: Path) -> None:
        """
        Repeating a split with the same seed produces identical labels.
        """
        raw_data = _make_raw_dataset(
            tmp_path,
            parameter_indices=list(range(6)),
            touchstone_indices=list(range(6)),
        )
        builder = ParameterDatasetBuilder(
            raw_data,
            tmp_path / "processed" / "cleaned_splits_parameter.csv",
        )
        cleaned = builder.clean()

        first = builder.split(
            cleaned,
            val_fraction=1 / 6,
            test_fraction=1 / 6,
            seed=42,
        )
        second = builder.split(
            cleaned,
            val_fraction=1 / 6,
            test_fraction=1 / 6,
            seed=42,
        )

        pd.testing.assert_series_equal(first["SPLIT_TYPE"], second["SPLIT_TYPE"])

    def test_build_reuses_existing_processed_csv(self, tmp_path: Path) -> None:
        """
        An existing processed CSV skips raw cleaning and split assignment.
        """
        raw_data = _make_raw_dataset(
            tmp_path,
            parameter_indices=list(range(6)),
            touchstone_indices=list(range(6)),
        )
        builder = ParameterDatasetBuilder(
            raw_data,
            tmp_path / "processed" / "cleaned_splits_parameter.csv",
        )
        first = builder.build(
            val_fraction=1 / 6,
            test_fraction=1 / 6,
            seed=42,
        )
        raw_data.parameter_csv.write_text("invalid,raw,csv\n", encoding="utf-8")

        cached = builder.build(
            val_fraction=1 / 6,
            test_fraction=1 / 6,
            seed=42,
        )

        pd.testing.assert_frame_equal(cached, first)

    def test_build_force_repeats_raw_cleaning(self, tmp_path: Path) -> None:
        """
        Force bypasses an existing processed CSV and reads raw parameters again.
        """
        raw_data = _make_raw_dataset(
            tmp_path,
            parameter_indices=list(range(6)),
            touchstone_indices=list(range(6)),
        )
        builder = ParameterDatasetBuilder(
            raw_data,
            tmp_path / "processed" / "cleaned_splits_parameter.csv",
        )
        builder.build(
            val_fraction=1 / 6,
            test_fraction=1 / 6,
            seed=42,
        )
        parameters = pd.read_csv(raw_data.parameter_csv)
        parameters.loc[0, "EPS"] = 4.2
        parameters.to_csv(raw_data.parameter_csv, index=False)

        rebuilt = builder.build(
            val_fraction=1 / 6,
            test_fraction=1 / 6,
            seed=42,
            force=True,
        )

        assert rebuilt.loc[0, "EPS"] == 4.2
