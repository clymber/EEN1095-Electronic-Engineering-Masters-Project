"""
Sampling helpers for S-parameter dataset views.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .ml_dataset import DLDataset

SIMULATION_COLUMN = "SIMU_INDEX"


def random_simu_indices(
    dataset: DLDataset, n_simu: int, seed: int | None = None
) -> np.ndarray:
    """
    Select a random subset of simulation indices from a dataset.
    """
    simulation_ids = np.asarray(dataset.dataframe[SIMULATION_COLUMN].drop_duplicates())
    return np.random.default_rng(seed).choice(
        simulation_ids,
        size=min(n_simu, len(simulation_ids)),
        replace=False,
    )
