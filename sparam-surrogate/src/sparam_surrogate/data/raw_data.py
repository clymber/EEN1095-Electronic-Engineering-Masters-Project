# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (sparam-surrogate)
#     language: python
#     name: sparam-surrogate
# ---

# %%
"""
Raw data(unzipped dataset) processing.
"""
import re
import textwrap
from pathlib import Path
from typing import TypedDict

import pandas as pd


# %%
class IndexConsistencyReport(TypedDict):
    """
    Report describing consistency between parameter.csv and Touchstone files.
    """
    parameter_count: int
    touchstone_count: int
    missing_parameter_records: list[int]
    missing_touchstones: list[int]
    extra_touchstone_files: list[str]

# %%
class RawData:
    """
    Utility class for handling raw data from the unzipped dataset.
    """
    def __init__(self, path: Path|str, nports: int) -> None:
        """
        - path: Path to the directory containing the raw data files.
        - nports: Number of ports in the S-parameter data.
        """
        self._path = Path(path)
        self._nports = nports

    @property
    def path(self) -> Path:
        """
        Return the root directory for one unzipped raw dataset.
        """
        return self._path

    @property
    def parameter_csv(self) -> Path:
        """
        Return the path to ``parameter.csv``.
        """
        return self._path / "parameter.csv"

    @property
    def nports(self) -> int:
        """
        Return the expected Touchstone port count.
        """
        return self._nports

    @property
    def variation_path(self) -> Path:
        """
        Return the directory containing Touchstone files.
        """
        return self._path / "variation"

    def touchstone(self, idx: int) -> Path:
        """
        Return the Touchstone path for a simulation index.
        """
        return self.variation_path / f"simu_{idx}.s{self._nports}p"

    def touchstones(self) -> list[Path]:
        """
        Return all Touchstone paths in the variation directory.
        """
        return sorted(self.variation_path.glob(f"simu_*.s{self._nports}p"))

    def touchstone_indices(self) -> list[int]:
        """
        Return sorted simulation indices with matching Touchstone files.
        """
        return sorted(self._get_touchstone_indices())

    def _get_touchstone_indices(self) -> dict[int, list[str]]:
        """
        Map simulation indices to Touchstone file names.
        """
        touchstone_paths = self.touchstones()
        regex = re.compile(r"simu_(\d+)\.s\d+p$", re.IGNORECASE)

        touchstone_by_index: dict[int, list[str]] = {}

        for path in touchstone_paths:
            match = regex.match(path.name)
            if not match:
                continue
            idx = int(match.group(1))
            touchstone_by_index.setdefault(idx, []).append(path.name)

        return touchstone_by_index

    def check_index_consistency(self) -> IndexConsistencyReport:
        """
        Identify mismatches between parameter.csv records and Touchstone files.
        
        Returns:
        {
            "parameter_count": int,
            "touchstone_count": int,
            "missing_parameter_records": list[int],
            "missing_touchstones": list[int],
            "extra_touchstone_files": list[str],
        }
        """
        params = pd.read_csv(self.parameter_csv)
        index_to_touchstone = self._get_touchstone_indices()

        ts_index_set = set(index_to_touchstone.keys())
        param_index_set = set(params["SIMU_INDEX"].astype(int).tolist())

        missing_params = sorted(
            ts_index_set - param_index_set
        )
        missing_touchstones = sorted(
            param_index_set - ts_index_set
        )
        extra_touchstone_files = [
            path
            for idx in missing_params
            for path in index_to_touchstone[idx]
        ]

        return {
            "parameter_count": len(params),
            "touchstone_count": len(self.touchstones()),
            "missing_parameter_records": missing_params,
            "missing_touchstones": missing_touchstones,
            "extra_touchstone_files": extra_touchstone_files,
        }

    def report_index_consistency(self) -> None:
        """
        Print an index consistency report to stdout.
        """
        report = self.check_index_consistency()
        print(f"Total parameter record: {report['parameter_count']}")
        print(f"Total touchstone files: {report['touchstone_count']}")

        if report['extra_touchstone_files']:
            missing = " ".join(report['extra_touchstone_files'])
            print("{} Touchstones with no parameter record:\n\t{}".format(
                len(report['extra_touchstone_files']),
                textwrap.shorten(missing, 80)
            ))

        if report['missing_touchstones']:
            missing = " ".join([str(i) for i in report['missing_touchstones']])
            print("{} parameter records with no Touchstones:\n\t{}".format(
                len(report['missing_touchstones']),
                textwrap.shorten(missing, 80)
            ))
