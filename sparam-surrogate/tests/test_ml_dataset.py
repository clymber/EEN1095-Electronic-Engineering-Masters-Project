"""
Unit tests for :class:`MLDataset`.
"""

from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest

from sparam_surrogate.data import MLDataset


class _CommonKwargs(TypedDict):
    """
    Shared keyword arguments for constructing small MLDataset fixtures.
    """

    split_labels: list[str]
    simulation_indices: list[int]
    frequencies_ghz: list[float]
    feature_names: list[str]


def _feature_matrix() -> np.ndarray:
    """
    Return a small design-frequency feature matrix.
    """
    return np.array(
        [
            [3.1, 0.01, 1.0],
            [3.1, 0.01, 2.0],
            [3.4, 0.02, 1.0],
            [3.4, 0.02, 2.0],
        ],
        dtype=float,
    )


def _common_kwargs() -> _CommonKwargs:
    """
    Return metadata shared by the test datasets.
    """
    return {
        "split_labels": ["train", "train", "test", "test"],
        "simulation_indices": [10, 10, 20, 20],
        "frequencies_ghz": [1.0, 2.0, 1.0, 2.0],
        "feature_names": ["EPS", "TAND", "FREQ_GHZ"],
    }


class TestMLDataset:
    """
    Unit tests for the model-ready dataset container.
    """

    def test_accepts_scalar_target_dataset(self) -> None:
        """Store a scalar target as a two-dimensional target array."""
        dataset = MLDataset(
            X=_feature_matrix(),
            target=[-1.0, -2.0, -3.0, -4.0],
            target_names=["S2_1_DB"],
            metadata={"target_mode": "scalar"},
            **_common_kwargs(),
        )

        assert dataset.X.shape == (4, 3)
        assert dataset.target.shape == (4, 1)
        assert dataset.feature_names == ("EPS", "TAND", "FREQ_GHZ")
        assert dataset.target_names == ("S2_1_DB",)
        assert dataset.metadata == {"target_mode": "scalar"}
        np.testing.assert_allclose(dataset.target[:, 0], [-1.0, -2.0, -3.0, -4.0])

    def test_accepts_multi_output_full_smatrix_target_dataset(self) -> None:
        """Store a flattened real/imaginary full-S-matrix target."""
        target = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5, 1.6],
            ],
            dtype=float,
        )
        dataset = MLDataset(
            X=_feature_matrix(),
            target=target,
            target_names=["REAL_S1_1", "IMAG_S1_1", "REAL_S2_1", "IMAG_S2_1"],
            metadata={"target_mode": "full_smatrix"},
            **_common_kwargs(),
        )

        assert dataset.target.shape == (4, 4)
        assert dataset.target_names == (
            "REAL_S1_1",
            "IMAG_S1_1",
            "REAL_S2_1",
            "IMAG_S2_1",
        )
        np.testing.assert_allclose(dataset.target, target)

    def test_rejects_mismatched_row_counts(self) -> None:
        """Reject metadata or targets whose first dimension differs from X."""
        with pytest.raises(ValueError, match="target rows"):
            MLDataset(
                X=_feature_matrix(),
                target=[-1.0, -2.0, -3.0],
                target_names=["S2_1_DB"],
                **_common_kwargs(),
            )

        kwargs = _common_kwargs()
        kwargs["split_labels"] = ["train", "test"]
        with pytest.raises(ValueError, match="split labels"):
            MLDataset(
                X=_feature_matrix(),
                target=[-1.0, -2.0, -3.0, -4.0],
                target_names=["S2_1_DB"],
                **kwargs,
            )

        kwargs = _common_kwargs()
        kwargs["simulation_indices"] = [10, 20]
        with pytest.raises(ValueError, match="simulation indices"):
            MLDataset(
                X=_feature_matrix(),
                target=[-1.0, -2.0, -3.0, -4.0],
                target_names=["S2_1_DB"],
                **kwargs,
            )

        kwargs = _common_kwargs()
        kwargs["frequencies_ghz"] = [1.0, 2.0]
        with pytest.raises(ValueError, match="frequency metadata"):
            MLDataset(
                X=_feature_matrix(),
                target=[-1.0, -2.0, -3.0, -4.0],
                target_names=["S2_1_DB"],
                **kwargs,
            )

    def test_rejects_mismatched_feature_and_target_names(self) -> None:
        """Reject name metadata that does not match array column counts."""
        with pytest.raises(ValueError, match="feature names"):
            MLDataset(
                X=_feature_matrix(),
                target=[-1.0, -2.0, -3.0, -4.0],
                feature_names=["EPS", "TAND"],
                target_names=["S2_1_DB"],
                split_labels=["train", "train", "test", "test"],
                simulation_indices=[10, 10, 20, 20],
                frequencies_ghz=[1.0, 2.0, 1.0, 2.0],
            )

        with pytest.raises(ValueError, match="target names"):
            MLDataset(
                X=_feature_matrix(),
                target=np.ones((4, 2)),
                target_names=["S2_1_DB"],
                **_common_kwargs(),
            )

    def test_save_load_round_trip_preserves_arrays_and_metadata(
        self, tmp_path: Path
    ) -> None:
        """Save and reload a dataset without losing arrays or metadata."""
        original = MLDataset(
            X=_feature_matrix(),
            target=[-1.0, -2.0, -3.0, -4.0],
            target_names=["S2_1_DB"],
            metadata={"target_mode": "scalar", "frequency_unit": "GHz"},
            **_common_kwargs(),
        )
        path = tmp_path / "processed" / "scalar_baseline_dataset.npz"

        original.save(path)
        loaded = MLDataset.load(path)

        assert path.is_file()
        np.testing.assert_allclose(loaded.X, original.X)
        np.testing.assert_allclose(loaded.target, original.target)
        np.testing.assert_array_equal(loaded.split_labels, original.split_labels)
        np.testing.assert_array_equal(
            loaded.simulation_indices, original.simulation_indices
        )
        np.testing.assert_allclose(loaded.frequencies_ghz, original.frequencies_ghz)
        assert loaded.feature_names == original.feature_names
        assert loaded.target_names == original.target_names
        assert loaded.metadata == original.metadata
