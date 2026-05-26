#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Unit test for class RawData in raw_data.py.
"""

from pathlib import Path
from sparam_surrogate.data.raw_data import RawData

class TestRawData:
    """
    Unit tests for RawData class.
    """

    @staticmethod
    def _make_rawdata_fixture(
        tmp_path: Path,
        parameter_indices: list[int],
        touchstone_indices: list[int],
        nports: int = 2,
    ) -> Path:
        """
        Create the minimum raw-data directory structure needed by RawData.
        """
        rawdata_path = tmp_path / "rawdata"
        variation_path = rawdata_path / "variation"
        variation_path.mkdir(parents=True)

        parameter_rows = ["SIMU_INDEX,WIDTH"]
        parameter_rows.extend(f"{idx},{idx * 0.1}" for idx in parameter_indices)
        (rawdata_path / "parameter.csv").write_text(
            "\n".join(parameter_rows) + "\n",
            encoding="utf-8",
        )

        for idx in touchstone_indices:
            (variation_path / f"simu_{idx}.s{nports}p").write_text(
                "# Hz S RI R 50\n",
                encoding="utf-8",
            )

        return rawdata_path

    def test_get_param_csv_path(self, tmp_path):
        """
        Return path to parameter.csv
        """
        rawdata_path = self._make_rawdata_fixture(
            tmp_path,
            parameter_indices=[0, 1],
            touchstone_indices=[0, 1],
            nports=2
        )
        raw_data = RawData(rawdata_path, nports=2)
        csv_path = raw_data.parameter_csv
        expected = rawdata_path / "parameter.csv"

        assert isinstance(csv_path, Path)
        assert csv_path == expected
        assert csv_path.is_file()  # Check if the file actually exists

    def test_variation_path(self, tmp_path):
        """
        Return path to variation directory
        """
        rawdata_path = self._make_rawdata_fixture(
            tmp_path,
            parameter_indices=[0, 1],
            touchstone_indices=[0, 1],
            nports=2
        )
        raw_data = RawData(rawdata_path, nports=2)
        variation_path = raw_data.variation_path
        expected = rawdata_path / "variation"

        assert isinstance(variation_path, Path)
        assert variation_path == expected
        assert variation_path.is_dir()  # Check if the directory actually exists

    def test_get_touchstone_file(self, tmp_path):
        """
        Return path to touchstone file for given variation and port
        """
        rawdata_path = self._make_rawdata_fixture(
            tmp_path,
            parameter_indices=[0, 1, 2],
            touchstone_indices=[0, 1, 2],
            nports=2
        )
        raw_data = RawData(rawdata_path, nports=2)
        touchstonefile = raw_data.touchstone(2)
        expected = rawdata_path / "variation/simu_2.s2p"

        assert isinstance(touchstonefile, Path)
        assert touchstonefile == expected
        assert touchstonefile.is_file()  # Check if the file actually exists

    def test_get_touchstone_list(self, tmp_path):
        """
        Return list of touchstone file paths for all variations
        """
        rawdata_path = self._make_rawdata_fixture(
            tmp_path,
            parameter_indices=[0, 1, 2],
            touchstone_indices=[0, 1, 2],
            nports=2
        )
        raw_data = RawData(rawdata_path, nports=2)

        touchstone_list = raw_data.touchstones()

        assert isinstance(touchstone_list, list)
        assert len(touchstone_list) == 3

    def test_index_consistency_reports_ok(self, tmp_path):
        """
        Return an empty mismatch report when parameter and Touchstone indices match.
        """
        rawdata_path = self._make_rawdata_fixture(
            tmp_path,
            parameter_indices=[0, 1, 2],
            touchstone_indices=[0, 1, 2],
        )
        raw_data = RawData(path=rawdata_path, nports=2)

        report = raw_data.check_index_consistency()

        assert report == {
            "parameter_count": 3,
            "touchstone_count": 3,
            "missing_parameter_records": [],
            "missing_touchstones": [],
            "extra_touchstone_files": [],
        }

    def test_index_consistency_missing_parameter_records(self, tmp_path):
        """
        Report Touchstone files that have no matching parameter.csv record.
        """
        rawdata_path = self._make_rawdata_fixture(
            tmp_path,
            parameter_indices=[0, 2],
            touchstone_indices=[0, 1, 2, 4],
        )
        raw_data = RawData(path=rawdata_path, nports=2)

        report = raw_data.check_index_consistency()

        assert report["parameter_count"] == 2
        assert report["touchstone_count"] == 4
        assert report["missing_parameter_records"] == [1, 4]
        assert report["missing_touchstones"] == []
        assert report["extra_touchstone_files"] == ["simu_1.s2p", "simu_4.s2p"]

    def test_index_consistency_missing_touchstones(self, tmp_path):
        """
        Report parameter.csv records that have no matching Touchstone file.
        """
        rawdata_path = self._make_rawdata_fixture(
            tmp_path,
            parameter_indices=[0, 1, 3],
            touchstone_indices=[0, 3],
        )
        raw_data = RawData(path=rawdata_path, nports=2)

        report = raw_data.check_index_consistency()

        assert report["parameter_count"] == 3
        assert report["touchstone_count"] == 2
        assert report["missing_parameter_records"] == []
        assert report["missing_touchstones"] == [1]
        assert report["extra_touchstone_files"] == []
