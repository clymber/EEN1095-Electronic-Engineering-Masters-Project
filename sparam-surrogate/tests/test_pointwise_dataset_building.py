"""
Tests for building the frequency-expanded point-wise CSV.
"""

import os
from pathlib import Path

import pandas as pd

from sparam_surrogate.data import (
    ParameterDatasetBuilder,
    PointwiseDataset,
    RawData,
)

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


def _write_split_parameter_csv(tmp_path: Path) -> tuple[RawData, Path]:
    """
    Build a six-design parameter CSV for frequency-expansion tests.
    """
    raw_path = tmp_path / "raw"
    variation_path = raw_path / "variation"
    variation_path.mkdir(parents=True)
    rows = []
    for index in range(6):
        row = dict(PARAMETER_VALUES)
        row["SIMU_INDEX"] = index
        rows.append(row)
        _write_touchstone(variation_path / f"simu_{index}.s2p")
    pd.DataFrame(rows).to_csv(raw_path / "parameter.csv", index=False)

    raw_data = RawData(raw_path, nports=2)
    split_parameter_csv = (
        tmp_path / "processed" / "cleaned_splits_parameter.csv"
    )
    parameter_builder = ParameterDatasetBuilder(raw_data, split_parameter_csv)
    parameter_builder.split(
        parameter_builder.clean(),
        val_fraction=1 / 6,
        test_fraction=1 / 6,
        seed=42,
    )
    return raw_data, parameter_builder.cleaned_splits_path


class TestPointwiseDatasetBuilding:
    """
    Unit tests for the frequency-expanded processed CSV.
    """

    def test_build_expands_split_parameters_over_frequency(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Each design-level parameter row expands over the common frequency grid.
        """
        _, split_parameter_csv = _write_split_parameter_csv(tmp_path)
        output_csv = (
            tmp_path / "processed" / PointwiseDataset.FREQUENCY_EXPANDED_FILENAME
        )

        cleaned = PointwiseDataset.build_frequency_expanded_csv(
            split_parameter_csv,
            output_csv,
            force=True,
        )

        assert tuple(cleaned.columns) == PointwiseDataset.COLUMNS
        assert len(cleaned) == 12
        assert cleaned["FREQ_GHZ"].tolist() == [1.0, 2.0] * 6
        assert set(cleaned["SPLIT_TYPE"]) == {"train", "val", "test"}
        for _, design_rows in cleaned.groupby("SIMU_INDEX"):
            assert len(design_rows) == 2
            assert design_rows["SPLIT_TYPE"].nunique() == 1
        assert output_csv.is_file()

    def test_build_reuses_frequency_expanded_csv_when_current(
        self,
        tmp_path: Path,
    ) -> None:
        """
        A processed CSV at least as new as its source is not rewritten.
        """
        _, split_parameter_csv = _write_split_parameter_csv(tmp_path)
        output_csv = tmp_path / "processed" / "frequency_expanded_dataset.csv"
        expected = PointwiseDataset.build_frequency_expanded_csv(
            split_parameter_csv,
            output_csv,
            force=True,
        )
        cached_mtime = output_csv.stat().st_mtime_ns

        cached = PointwiseDataset.build_frequency_expanded_csv(
            split_parameter_csv,
            output_csv,
        )

        pd.testing.assert_frame_equal(cached, expected)
        assert output_csv.stat().st_mtime_ns == cached_mtime

    def test_build_rebuilds_frequency_expanded_csv_when_source_is_newer(
        self,
        tmp_path: Path,
    ) -> None:
        """
        A frequency-expanded CSV older than its source is rebuilt.
        """
        _, split_parameter_csv = _write_split_parameter_csv(tmp_path)
        output_csv = tmp_path / "processed" / "frequency_expanded_dataset.csv"
        PointwiseDataset.build_frequency_expanded_csv(
            split_parameter_csv,
            output_csv,
            force=True,
        )
        output_csv.write_text("stale\n", encoding="utf-8")
        os.utime(output_csv, ns=(1_000_000_000, 1_000_000_000))
        os.utime(split_parameter_csv, ns=(2_000_000_000, 2_000_000_000))

        rebuilt = PointwiseDataset.build_frequency_expanded_csv(
            split_parameter_csv,
            output_csv,
        )

        assert tuple(rebuilt.columns) == PointwiseDataset.COLUMNS
        assert len(rebuilt) == 12

    def test_build_does_not_read_raw_parameter_csv(self, tmp_path: Path) -> None:
        """
        Frequency expansion depends on the cleaned CSV, not raw parameters.
        """
        raw_data, split_parameter_csv = _write_split_parameter_csv(tmp_path)
        raw_data.parameter_csv.write_text("invalid,raw,csv\n", encoding="utf-8")

        cleaned = PointwiseDataset.build_frequency_expanded_csv(
            split_parameter_csv,
            tmp_path / "processed" / "frequency_expanded_dataset.csv",
            force=True,
        )

        assert len(cleaned) == 12
