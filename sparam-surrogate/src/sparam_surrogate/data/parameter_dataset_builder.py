"""
Build the cleaned and split design-level parameter dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sparam_surrogate.config import PROJECT_ROOT

from .raw_data import RawData


class ParameterDatasetBuilder:
    """
    Clean raw parameter rows and assign design-level dataset splits.

    The persisted CSV contains one row per simulation and is the shared
    source for later point-wise and whole-curve preprocessing.
    """

    PARAMETER_COLUMNS = (
        "EPS",
        "TAND",
        "PITCH",
        "TRACE_LEN",
        "START",
        "VIAR",
        "ANTIPADR",
        "TDIEL",
        "DISTTL",
        "TLWIDTH",
    )
    SIMULATION_COLUMN = "SIMU_INDEX"
    TOUCHSTONE_COLUMN = "TOUCHSTONE_REL_PATH"
    SPLIT_COLUMN = "SPLIT_TYPE"
    CLEANED_COLUMNS = (
        *PARAMETER_COLUMNS,
        SIMULATION_COLUMN,
        TOUCHSTONE_COLUMN,
    )
    SPLIT_COLUMNS = (*CLEANED_COLUMNS, SPLIT_COLUMN)

    def __init__(self, raw_data: RawData, cleaned_splits_path: Path | str) -> None:
        """
        Configure design-level parameter preprocessing.
        """
        self.raw_data = raw_data
        self.cleaned_splits_path = Path(cleaned_splits_path)

    def load(self) -> pd.DataFrame:
        """
        Load the existing cleaned and split parameter CSV.
        """
        split_parameters = pd.read_csv(self.cleaned_splits_path)
        split_parameters[self.SIMULATION_COLUMN] = split_parameters[
            self.SIMULATION_COLUMN
        ].astype(np.int64)
        return split_parameters

    def build(
        self,
        *,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 42,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Load the processed CSV or clean and split raw parameters when required.
        """
        if self.cleaned_splits_path.is_file() and not force:
            return self.load()
        cleaned = self.clean()
        return self.split(
            cleaned,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )

    def clean(self) -> pd.DataFrame:
        """
        Retain parameter rows with Touchstone files and add their paths.
        """
        columns = [*self.PARAMETER_COLUMNS, self.SIMULATION_COLUMN]
        parameters = pd.read_csv(self.raw_data.parameter_csv).loc[:, columns]
        touchstone_indices = set(self.raw_data.touchstone_indices())
        aligned = parameters.loc[
            parameters[self.SIMULATION_COLUMN].isin(touchstone_indices)
        ].copy()
        aligned[self.SIMULATION_COLUMN] = aligned[self.SIMULATION_COLUMN].astype(
            np.int64
        )
        aligned[self.TOUCHSTONE_COLUMN] = [
            self._portable_touchstone_path(
                self.raw_data.touchstone(int(simulation_index))
            )
            for simulation_index in aligned[self.SIMULATION_COLUMN]
        ]
        return aligned.reindex(columns=self.CLEANED_COLUMNS).reset_index(drop=True)

    def split(
        self,
        cleaned: pd.DataFrame,
        *,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Assign deterministic design-level splits and write the processed CSV.
        """
        split_parameters = cleaned.reindex(columns=self.CLEANED_COLUMNS).copy()
        simulation_indices = split_parameters[self.SIMULATION_COLUMN].to_numpy(
            dtype=np.int64
        )
        split_by_index = self._split_labels_by_index(
            simulation_indices,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        split_parameters[self.SPLIT_COLUMN] = split_parameters[
            self.SIMULATION_COLUMN
        ].map(split_by_index)
        self.cleaned_splits_path.parent.mkdir(parents=True, exist_ok=True)
        split_parameters.to_csv(self.cleaned_splits_path, index=False)
        return split_parameters

    def _portable_touchstone_path(self, touchstone_path: Path) -> str:
        """
        Return a project-relative path or an absolute external test path.
        """
        resolved = touchstone_path.resolve()
        try:
            return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()

    def _split_labels_by_index(
        self,
        simulation_indices: np.ndarray,
        *,
        val_fraction: float,
        test_fraction: float,
        seed: int,
    ) -> dict[int, str]:
        """
        Assign deterministic train, validation, and test labels by design.
        """
        n_designs = len(simulation_indices)
        n_test = int(round(n_designs * test_fraction))
        n_val = int(round(n_designs * val_fraction))
        if min(n_test, n_val, n_designs - n_test - n_val) < 1:
            raise ValueError(
                "Split fractions must create non-empty train, validation, "
                "and test sets."
            )
        shuffled = np.asarray(simulation_indices, dtype=np.int64).copy()
        np.random.default_rng(seed).shuffle(shuffled)
        labels: dict[int, str] = {}
        for index in shuffled[:n_test]:
            labels[int(index)] = "test"
        for index in shuffled[n_test : n_test + n_val]:
            labels[int(index)] = "val"
        for index in shuffled[n_test + n_val :]:
            labels[int(index)] = "train"
        return labels
