"""
Unit tests for design-level train/validation/test splitting.
"""

import numpy as np
import pytest

from sparam_surrogate.data import DesignFrequencySplitter


class TestDesignFrequencySplitter:
    """
    Unit tests for reproducible SIMU_INDEX-based splitting.
    """

    def test_deterministic_split_with_fixed_seed(self) -> None:
        """
        Split membership is reproducible for the same seed.
        """
        simulation_indices = np.arange(10)
        splitter = DesignFrequencySplitter(
            test_size=0.2,
            val_size=0.2,
            random_state=123,
        )

        first = splitter.split(simulation_indices)
        second = splitter.split(simulation_indices)

        np.testing.assert_array_equal(first.train_indices, second.train_indices)
        np.testing.assert_array_equal(first.val_indices, second.val_indices)
        np.testing.assert_array_equal(first.test_indices, second.test_indices)

    def test_assigns_each_simulation_index_exactly_once(self) -> None:
        """
        Every design belongs to exactly one split.
        """
        simulation_indices = np.arange(10, 20)
        split = DesignFrequencySplitter(
            test_size=0.2,
            val_size=0.2,
            random_state=7,
        ).split(simulation_indices)

        combined = np.concatenate(
            [split.train_indices, split.val_indices, split.test_indices]
        )
        np.testing.assert_array_equal(np.sort(combined), simulation_indices)
        assert len(combined) == len(set(combined.tolist()))

    def test_has_no_overlap_across_splits(self) -> None:
        """
        Train, validation, and test split memberships are disjoint.
        """
        split = DesignFrequencySplitter(
            test_size=0.2,
            val_size=0.2,
            random_state=99,
        ).split(np.arange(10))

        train = set(split.train_indices.tolist())
        val = set(split.val_indices.tolist())
        test = set(split.test_indices.tolist())

        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

    def test_respects_configured_split_sizes(self) -> None:
        """
        Fractional split sizes are interpreted against the full design count.
        """
        split = DesignFrequencySplitter(
            test_size=0.2,
            val_size=0.3,
            random_state=42,
        ).split(np.arange(10))

        assert len(split.train_indices) == 5
        assert len(split.val_indices) == 3
        assert len(split.test_indices) == 2

    def test_rejects_empty_or_duplicate_simulation_indices(self) -> None:
        """
        Invalid design index inputs are rejected before random splitting.
        """
        splitter = DesignFrequencySplitter()

        with pytest.raises(ValueError, match="at least one"):
            splitter.split([])

        with pytest.raises(ValueError, match="unique"):
            splitter.split([0, 1, 1, 2])

    def test_expands_split_labels_after_design_frequency_expansion(self) -> None:
        """
        Split labels repeat once per frequency for each design row.
        """
        simulation_indices = np.arange(100, 110)
        split = DesignFrequencySplitter(
            test_size=0.2,
            val_size=0.2,
            random_state=123,
        ).split(simulation_indices)

        design_labels = split.labels_for(simulation_indices)
        expanded_labels = split.expand_labels(simulation_indices, n_frequencies=3)

        assert design_labels.shape == (10,)
        assert expanded_labels.shape == (30,)
        assert set(design_labels.tolist()) == {"train", "val", "test"}
        for design_index, label in enumerate(design_labels):
            start = design_index * 3
            np.testing.assert_array_equal(
                expanded_labels[start : start + 3],
                np.array([label, label, label]),
            )
