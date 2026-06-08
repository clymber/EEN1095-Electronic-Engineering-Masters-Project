"""
Tests for cleaned CSV construction by :class:`MLDatasetBuilder`.
"""

from pathlib import Path

import pandas as pd
import pytest

from sparam_surrogate.data import MLDatasetBuilder, RawData


PARAMETER_COLUMNS = {
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


def _write_touchstone(path: Path, frequencies_ghz: list[float]) -> None:
    """
    Write a minimal two-port Touchstone file in RI format.

    Parameters
    ----------
    path:
        Destination ``.s2p`` file.
    frequencies_ghz:
        Frequency samples to write in GHz.
    """
    lines = ["# GHz S RI R 50"]
    for frequency in frequencies_ghz:
        lines.append(f"{frequency:g} 0.1 0.0 0.5 0.0 0.2 0.0 0.3 0.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_raw_dataset(
    tmp_path: Path,
    parameter_indices: list[int],
    touchstone_indices: list[int],
) -> RawData:
    """
    Create a synthetic raw-data tree with parameter rows and Touchstones.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    parameter_indices:
        ``SIMU_INDEX`` values written to ``parameter.csv``.
    touchstone_indices:
        Indices for which ``simu_<index>.s2p`` files are written.

    Returns
    -------
    RawData
        Raw-data locator for the generated two-port dataset.
    """
    raw_path = tmp_path / "raw"
    variation_path = raw_path / "variation"
    variation_path.mkdir(parents=True)
    rows = []
    for index in parameter_indices:
        row = dict(PARAMETER_COLUMNS)
        row["SIMU_INDEX"] = index
        rows.append(row)
    pd.DataFrame(rows).to_csv(raw_path / "parameter.csv", index=False)
    for index in touchstone_indices:
        _write_touchstone(variation_path / f"simu_{index}.s2p", [1.0, 2.0])
    return RawData(raw_path, nports=2)


class TestMLDatasetBuilderCleaning:
    """
    Unit tests for the cleaned CSV artifact.
    """

    def test_data_cleaning_writes_expected_schema_and_paths(self, tmp_path: Path) -> None:
        """
        Cleaning expands matched designs over frequency rows and writes one CSV.
        """
        raw_data = _make_raw_dataset(
            tmp_path,
            parameter_indices=[2, 1, 3],
            touchstone_indices=[1, 2, 99],
        )
        builder = MLDatasetBuilder(raw_data, tmp_path / "processed")

        cleaned = builder.data_cleaning(force=True)

        assert tuple(cleaned.columns) == MLDatasetBuilder.CLEANED_COLUMNS
        assert cleaned["SIMU_INDEX"].tolist() == [2, 2, 1, 1]
        assert cleaned["FREQ_GHZ"].tolist() == [1.0, 2.0, 1.0, 2.0]
        assert cleaned["SPLIT_TYPE"].tolist() == ["", "", "", ""]
        assert cleaned["TOUCHSTONE_REL_PATH"].str.startswith("raw/variation/").all()
        assert not cleaned["TOUCHSTONE_REL_PATH"].map(Path).map(Path.is_absolute).any()
        assert builder.cleaned_path.is_file()

        cached = builder.data_cleaning(force=False)
        assert tuple(cached.columns) == MLDatasetBuilder.CLEANED_COLUMNS
        assert cached.shape == cleaned.shape

    def test_data_cleaning_rejects_duplicate_simulation_indices(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Duplicate ``SIMU_INDEX`` rows are rejected before expansion.
        """
        raw_data = _make_raw_dataset(
            tmp_path,
            parameter_indices=[1, 1],
            touchstone_indices=[1],
        )
        builder = MLDatasetBuilder(raw_data, tmp_path / "processed")

        with pytest.raises(ValueError, match="unique"):
            builder.data_cleaning(force=True)

    def test_data_cleaning_does_not_write_old_npz_outputs(self, tmp_path: Path) -> None:
        """
        The reconstructed preprocessing stage writes a CSV, not eager arrays.
        """
        raw_data = _make_raw_dataset(
            tmp_path,
            parameter_indices=[0, 1],
            touchstone_indices=[0, 1],
        )
        processed_dir = tmp_path / "processed"

        MLDatasetBuilder(raw_data, processed_dir).data_cleaning(force=True)

        assert (processed_dir / "sipi_dataset_cleaned.csv").is_file()
        assert not (processed_dir / "scalar_baseline_dataset.npz").exists()
        assert not (processed_dir / "full_smatrix_dataset.npz").exists()
