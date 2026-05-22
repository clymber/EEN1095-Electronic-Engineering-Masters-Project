# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python (sparam-surrogate)
#     language: python
#     name: sparam-surrogate
# ---

# %%
"""
Raw data(unzipped dataset) processing.
"""
import pandas as pd
import re
from pathlib import Path
from typing import TypedDict


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
    def parameter_csv(self) -> Path:
        """
        Returns the path to the parameter CSV file.
        """
        return self._path / "parameter.csv"

    @property
    def variation_path(self) -> Path:
        """
        Returns the path to the directory containing the touchstone files.
        """
        return self._path / "variation"

    def touchstone(self, idx: int) -> Path:
        """
        Returns the path to the touchstone file for a given index.
        - idx: Index of the touchstone file (0-based).
        """
        return self.variation_path / f"simu_{idx}.s{self._nports}p"

    def touchstones(self) -> list[Path]:
        """
        Returns a list of paths to all touchstone files in the variation directory.
        """
        return sorted(self.variation_path.glob(f"simu_*.s{self._nports}p"))

    def _get_touchstone_indices(self) -> dict[int, list[str]]:
        """
        Returns a dictionary that maps indices to Touchstone file names.
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
