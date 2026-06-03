"""
Reproducible design-level splitting for design-frequency datasets.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DesignFrequencySplit:
    """
    Train/validation/test split membership for PCB design simulations.

    The arrays contain source ``SIMU_INDEX`` values, not expanded
    design-frequency row numbers.
    """

    train_indices: np.ndarray  # Array of unique SIMU_INDEX values for training designs.
    val_indices: np.ndarray  # Array of unique SIMU_INDEX values for validation designs.
    test_indices: np.ndarray  # Array of unique SIMU_INDEX values for test designs.

    def labels_for(self, simulation_indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """
        Return one split label per requested simulation index.

        Labels are returned in the same order as ``simulation_indices``.

        Parameters
        ----------
        simulation_indices:
            One-dimensional sequence of source ``SIMU_INDEX`` values to label.
            Every value must be present in exactly one of this split's train,
            validation, or test index arrays.

        Returns
        -------
        np.ndarray
            One-dimensional string array containing ``"train"``, ``"val"``,
            or ``"test"`` for each requested simulation index.

        Raises
        ------
        ValueError
            If any requested ``SIMU_INDEX`` value is not present in this split.
        """
        indices = np.asarray(simulation_indices, dtype=np.int64)
        label_by_index = self._label_by_index()
        labels: list[str] = []
        for index in indices:
            try:
                labels.append(label_by_index[int(index)])
            except KeyError as exc:
                raise ValueError(
                    f"SIMU_INDEX {int(index)} is not present in this split."
                ) from exc
        return np.asarray(labels, dtype=str)

    def expand_labels(
        self,
        simulation_indices: Sequence[int] | np.ndarray,
        n_frequencies: int,
    ) -> np.ndarray:
        """
        Return labels for design-frequency rows in design-major order.

        For each design label, the label is repeated ``n_frequencies`` times.
        This matches the planned feature-row order where each design is expanded
        over the full frequency grid before moving to the next design.

        Parameters
        ----------
        simulation_indices:
            One-dimensional sequence of source ``SIMU_INDEX`` values whose
            labels should be expanded over the frequency grid.
        n_frequencies:
            Number of frequency rows associated with each design.

        Returns
        -------
        np.ndarray
            One-dimensional string array of row-level split labels with length
            ``len(simulation_indices) * n_frequencies``.

        Raises
        ------
        ValueError
            If ``n_frequencies`` is not positive or if any requested
            ``SIMU_INDEX`` value is not present in this split.
        """
        if n_frequencies <= 0:
            raise ValueError("n_frequencies must be positive.")
        return np.repeat(self.labels_for(simulation_indices), int(n_frequencies))

    def _label_by_index(self) -> dict[int, str]:
        """
        Build a lookup from each ``SIMU_INDEX`` value to its split label.

        Returns
        -------
        dict[int, str]
            Mapping from source ``SIMU_INDEX`` values to ``"train"``, ``"val"``,
            or ``"test"``.
        """
        mapping: dict[int, str] = {}
        for index in self.train_indices:
            mapping[int(index)] = "train"
        for index in self.val_indices:
            mapping[int(index)] = "val"
        for index in self.test_indices:
            mapping[int(index)] = "test"
        return mapping


class DesignFrequencySplitter:
    """
    Create reproducible train/validation/test splits by ``SIMU_INDEX``.

    Splitting happens before frequency expansion to prevent the same physical
    design from appearing in more than one split.
    """

    def __init__(
        self,
        test_size: float | int = 0.15,
        val_size: float | int = 0.15,
        random_state: int | None = 42,
    ) -> None:
        """
        Configure a design-level splitter.

        Parameters
        ----------
        test_size:
            Test-set size as either a fraction of the full design count or an absolute
            number of designs.
        val_size:
            Validation-set size as either a fraction of the full design count
            or an absolute number of designs.
        random_state:
            Seed passed to NumPy's random generator. A fixed integer gives
            reproducible split membership.

        Returns
        -------
        None
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(
        self, simulation_indices: Sequence[int] | np.ndarray
    ) -> DesignFrequencySplit:
        """
        Split unique simulation indices into train, validation, and test sets.

        Parameters
        ----------
        simulation_indices:
            One-dimensional sequence of unique integer-valued ``SIMU_INDEX``
            values to split at the design level.

        Returns
        -------
        DesignFrequencySplit
            Reproducible train, validation, and test membership containing
            sorted source ``SIMU_INDEX`` values for each split.

        Raises
        ------
        ValueError
            If ``simulation_indices`` is empty, non-one-dimensional,
            non-integer-valued, duplicated, or if ``test_size`` and
            ``val_size`` leave no training designs.
        """
        indices = self._validate_simulation_indices(simulation_indices)
        n_designs = len(indices)
        n_test = self._size_to_count(self.test_size, n_designs, "test_size")
        n_val = self._size_to_count(self.val_size, n_designs, "val_size")
        if n_test + n_val >= n_designs:
            raise ValueError(
                "test_size and val_size must leave at least one training design."
            )

        shuffled = indices.copy()
        np.random.default_rng(self.random_state).shuffle(shuffled)

        test_indices = np.sort(shuffled[:n_test])
        val_indices = np.sort(shuffled[n_test : n_test + n_val])
        train_indices = np.sort(shuffled[n_test + n_val :])

        return DesignFrequencySplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )

    @staticmethod
    def _validate_simulation_indices(
        simulation_indices: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """
        Validate and normalize simulation indices for design-level splitting.

        Parameters
        ----------
        simulation_indices:
            Candidate source ``SIMU_INDEX`` values supplied by the caller.

        Returns
        -------
        np.ndarray
            One-dimensional integer array containing the validated simulation indices.

        Raises
        ------
        ValueError
            If ``simulation_indices`` is empty, non-one-dimensional,
            non-integer-valued, or contains duplicate values.
        """
        raw_indices = np.asarray(simulation_indices)
        if raw_indices.ndim != 1 or len(raw_indices) == 0:
            raise ValueError("simulation_indices must contain at least one value.")
        float_indices = raw_indices.astype(float)
        integer_indices = float_indices.astype(np.int64)
        if not np.allclose(float_indices, integer_indices):
            raise ValueError("simulation_indices must be integer-valued.")
        if len(np.unique(integer_indices)) != len(integer_indices):
            raise ValueError("simulation_indices must be unique.")
        return integer_indices

    @staticmethod
    def _size_to_count(size: float | int, n_designs: int, name: str) -> int:
        """
        Convert a fractional or absolute split size into a design count.

        Parameters
        ----------
        size:
            Split size as a fraction of ``n_designs`` or an absolute count.
        n_designs:
            Total number of available designs.
        name:
            Parameter name to include in validation error messages.

        Returns
        -------
        int
            Number of designs assigned to the requested split.

        Raises
        ------
        ValueError
            If a fractional ``size`` is not between 0 and 1, if the resulting
            count is zero, or if the count is greater than or equal to
            ``n_designs``.
        """
        if isinstance(size, float):
            if size <= 0.0 or size >= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
            count = int(round(n_designs * size))
        else:
            count = int(size)
        if count <= 0:
            raise ValueError(f"{name} selects no simulation indices.")
        if count >= n_designs:
            raise ValueError(f"{name} must be smaller than the design count.")
        return count
