"""
sparam_surrogate

Machine learning surrogate models for predicting PCB interconnect
S-parameters and signal integrity metrics.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sparam-surrogate")
except PackageNotFoundError:
    __version__ = "0.0.0"

__author__ = "Chunyu Long"

__app_name__ = "sparam-surrogate"

__all__ = [
    "__version__",
    "__author__",
    "__app_name__",
]
