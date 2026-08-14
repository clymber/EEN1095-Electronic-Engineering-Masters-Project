"""
Tests for shared model prediction curve plots.
"""

from collections.abc import Mapping
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sparam_surrogate.data import PointwiseDataset  # noqa: E402
from sparam_surrogate.models.base import SparamModel  # noqa: E402
from sparam_surrogate.utils.model_prediction_plots import (  # noqa: E402
    plot_design_model_comparison_curves,
    plot_design_prediction_curves,
)


class _ScalarFrequencyModel(SparamModel):
    """
    Small fitted scalar model stub for plotting tests.
    """

    name = "scalar_ridge"

    def fit(
        self,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,  # pylint: disable=invalid-name
        y_val: np.ndarray | None = None,
    ) -> SparamModel:
        """
        Return the fitted model stub.
        """
        del X_train, y_train, X_val, y_val
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Return a deterministic scalar prediction from frequency.
        """
        return -np.asarray(X)[:, -1]


class _VectorFrequencyModel(_ScalarFrequencyModel):
    """
    Small fitted vector model stub for plotting tests.
    """

    name = "vector_ridge"

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Return deterministic vector predictions from frequency.
        """
        frequency = np.asarray(X)[:, -1]
        return np.column_stack((-frequency, -2.0 * frequency))


class _ScalarTargetLoader:
    """
    Small TouchstoneLoader-like callable for scalar plotting tests.
    """

    target_names = ("S2_1_DB",)

    def __call__(
        self,
        features: np.ndarray,
        row_metadata: Mapping[str, Any],
    ) -> np.ndarray:
        """
        Return the scalar target for one row.
        """
        _ = features
        frequency = float(row_metadata["FREQ_GHZ"])
        return np.asarray([-frequency])


class _VectorTargetLoader(_ScalarTargetLoader):
    """
    Small TouchstoneLoader-like callable for vector plotting tests.
    """

    target_names = ("S2_1_DB", "S3_1_DB")

    def __call__(
        self,
        features: np.ndarray,
        row_metadata: Mapping[str, Any],
    ) -> np.ndarray:
        """
        Return vector targets for one row.
        """
        _ = features
        frequency = float(row_metadata["FREQ_GHZ"])
        return np.asarray([-frequency, -2.0 * frequency])


def _test_dataset() -> PointwiseDataset:
    """
    Return a small two-design test dataset.
    """
    frame = pd.DataFrame(
        {
            "EPS": [3.1, 3.1, 4.2, 4.2],
            "SIMU_INDEX": [101, 101, 102, 102],
            "FREQ_GHZ": [2.0, 1.0, 2.0, 1.0],
            "TOUCHSTONE_REL_PATH": ["design_101.s6p"] * 2 + ["design_102.s6p"] * 2,
            "SPLIT_TYPE": ["test"] * 4,
        }
    )
    for column in PointwiseDataset.PARAMETER_COLUMNS:
        if column not in frame:
            frame[column] = 1.0
    return PointwiseDataset(frame, "test")


def test_scalar_model_uses_one_target_row_and_design_columns() -> None:
    """
    Scalar predictions are plotted as one row with one column per selected design.
    """
    dataset = _test_dataset()

    fig = plot_design_prediction_curves(
        _ScalarFrequencyModel(),
        dataset,
        _ScalarTargetLoader(),
        (102, 101),
    )

    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == [
        "SIMU_INDEX 102",
        "SIMU_INDEX 101",
    ]
    assert fig.axes[0].get_ylabel() == "S2_1_DB"
    assert fig.axes[0].get_xlabel() == "Frequency (GHz)"
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Scalar Ridge: Selected Test Design Curves"
    assert len(fig.axes[0].lines) == 2
    np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [1.0, 2.0])
    plt.close(fig)


def test_vector_model_uses_target_rows_and_design_columns() -> None:
    """
    Vector predictions transpose the grid: target rows by sampled design columns.
    """
    dataset = _test_dataset()

    fig = plot_design_prediction_curves(
        _VectorFrequencyModel(),
        dataset,
        _VectorTargetLoader(),
        (102, 101),
    )

    assert len(fig.axes) == 4
    assert [ax.get_title() for ax in fig.axes[:2]] == [
        "SIMU_INDEX 102",
        "SIMU_INDEX 101",
    ]
    assert fig.axes[2].get_title() == ""
    assert fig.axes[0].get_ylabel() == "S2_1_DB"
    assert fig.axes[2].get_ylabel() == "S3_1_DB"
    assert fig.axes[2].get_xlabel() == "Frequency (GHz)"
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Vector Ridge: Selected Test Design Curves"
    plt.close(fig)


def test_model_comparison_plots_all_models_against_truth() -> None:
    """
    Multi-model comparison plots true and predicted curves together.
    """
    dataset = _test_dataset()

    fig = plot_design_model_comparison_curves(
        {
            "Polynomial Ridge": _VectorFrequencyModel(),
            "Random Forest": _VectorFrequencyModel(),
        },
        dataset,
        _VectorTargetLoader(),
        (101,),
    )

    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "SIMU_INDEX 101"
    assert fig.axes[0].get_ylabel() == "S2_1_DB"
    assert fig.axes[1].get_ylabel() == "S3_1_DB"
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Selected Test Design Curve Comparison"
    assert len(fig.axes[0].lines) == 3
    assert [line.get_label() for line in fig.axes[0].lines] == [
        "True",
        "Polynomial Ridge",
        "Random Forest",
    ]
    plt.close(fig)
