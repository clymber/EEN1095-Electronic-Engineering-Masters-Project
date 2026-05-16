#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Unit test for class RawData in raw_data.py.
"""

from pathlib import Path
import pytest

from sparam_surrogate.config import PROJECT_ROOT
from sparam_surrogate.data.raw_data import RawData

class TestRawData:
    """
    Unit tests for RawData class.
    """

    rawdata_path = PROJECT_ROOT / "data/raw/linkOn8CavityStackBetween10x10Array_19_08_2021/"
    nports = 12

    def test_get_param_csv_path(self):
        """
        Return path to parameter.csv
        """
        raw_data = RawData(path=self.rawdata_path, nports=self.nports)
        csv_path = raw_data.parameter_csv
        expected = self.rawdata_path / "parameter.csv"

        assert isinstance(csv_path, Path)
        assert csv_path == expected
        assert csv_path.is_file()  # Check if the file actually exists

    def test_variation_path(self):
        """
        Return path to variation directory
        """
        raw_data = RawData(path=self.rawdata_path, nports=self.nports)
        variation_path = raw_data.variation_path
        expected = self.rawdata_path / "variation"

        assert isinstance(variation_path, Path)
        assert variation_path == expected
        assert variation_path.is_dir()  # Check if the directory actually exists

    def test_get_touchstone_file(self):
        """
        Return path to touchstone file for given variation and port
        """
        raw_data = RawData(path=self.rawdata_path, nports=self.nports)
        touchstonefile = raw_data.touchstone(24)
        expected = self.rawdata_path / f"variation/simu_24.s{self.nports}p"

        assert isinstance(touchstonefile, Path)
        assert touchstonefile == expected
        assert touchstonefile.is_file()  # Check if the file actually exists

    def test_get_touchstone_list(self):
        """
        Return list of touchstone file paths for all variations
        """
        raw_data = RawData(path=self.rawdata_path, nports=self.nports)
        touchstone_list = raw_data.touchstones()

        assert isinstance(touchstone_list, list)
