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
from pathlib import Path


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
        return sorted(self.variation_path.glob(f"variation_*.s{self._nports}p"))

