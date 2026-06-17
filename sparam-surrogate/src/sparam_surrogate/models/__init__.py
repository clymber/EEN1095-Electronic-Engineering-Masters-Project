"""
Public model interfaces for S-parameter surrogate modelling.
"""

from .base import SparamModel
from .polynomial import (
    POLYNOMIAL_ALPHA_GRID,
    POLYNOMIAL_DEGREE_GRID,
    PolynomialModel,
    PowersOnlyPolynomialFeatures,
)
from .ridge import RIDGE_ALPHA_GRID, RidgeModel, ScalarRidgeModel, VectorRidgeModel

__all__ = [
    "POLYNOMIAL_ALPHA_GRID",
    "POLYNOMIAL_DEGREE_GRID",
    "RIDGE_ALPHA_GRID",
    "PolynomialModel",
    "PowersOnlyPolynomialFeatures",
    "RidgeModel",
    "ScalarRidgeModel",
    "SparamModel",
    "VectorRidgeModel",
]
