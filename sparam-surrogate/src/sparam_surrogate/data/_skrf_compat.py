"""
Import scikit-rf with compatibility for NumPy versions before 2.0.
"""

# scikit-rf 2.0.1 accesses ``np.typing`` without importing the submodule first.
# Importing it here exposes that attribute when the project uses NumPy 1.26.
import numpy.typing  # noqa: F401
import skrf as rf

__all__ = ["rf"]
