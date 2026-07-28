"""
Tests for dataset simulation-index sampling helpers.
"""

import numpy as np
import pandas as pd

from sparam_surrogate.data import PointwiseDataset, random_simu_indices


def _sample_dataset() -> PointwiseDataset:
    """
    Return a small test split with repeated simulation indices.
    """
    frame = pd.DataFrame(
        {
            "EPS": [3.0, 3.0, 3.4, 3.4, 4.0, 4.0, 4.4, 4.4],
            "FREQ_GHZ": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "SIMU_INDEX": [10, 10, 20, 20, 30, 30, 40, 40],
            "TOUCHSTONE_REL_PATH": [
                "raw/variation/simu_10.s2p",
                "raw/variation/simu_10.s2p",
                "raw/variation/simu_20.s2p",
                "raw/variation/simu_20.s2p",
                "raw/variation/simu_30.s2p",
                "raw/variation/simu_30.s2p",
                "raw/variation/simu_40.s2p",
                "raw/variation/simu_40.s2p",
            ],
            "SPLIT_TYPE": ["test"] * 8,
        }
    )
    for column in PointwiseDataset.PARAMETER_COLUMNS:
        if column not in frame:
            frame[column] = 1.0
    return PointwiseDataset(frame, "test")


def test_random_simu_indices_is_seed_reproducible() -> None:
    """
    Seeded sampling returns the same simulation-index subset each time.
    """
    dataset = _sample_dataset()

    first = random_simu_indices(dataset, 3, seed=123)
    second = random_simu_indices(dataset, 3, seed=123)

    np.testing.assert_array_equal(first, second)


def test_random_simu_indices_caps_count_to_available_simulations() -> None:
    """
    Requesting too many simulations returns all available unique indices.
    """
    dataset = _sample_dataset()

    selected = random_simu_indices(dataset, 10, seed=123)

    assert len(selected) == 4
    assert set(selected.tolist()) == {10, 20, 30, 40}


def test_random_simu_indices_allows_zero_samples() -> None:
    """
    Requesting zero simulations returns an empty array.
    """
    dataset = _sample_dataset()

    selected = random_simu_indices(dataset, 0, seed=123)

    assert selected.size == 0
