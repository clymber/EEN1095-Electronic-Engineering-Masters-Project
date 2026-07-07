"""
Build the lightweight cleaned CSV used by lazy S-parameter training datasets.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import skrf as rf

from sparam_surrogate.config import PROJECT_ROOT

from .ml_dataset import DLDataset
from .pcb_parameters import PcbParameters
from .raw_data import RawData


class MLDatasetBuilder:
    """
    Create and split the cleaned design-frequency preprocessing dataframe.

    The builder writes one durable CSV artifact. It does not build model
    targets and does not save eager NumPy arrays.
    """

    CLEANED_FILENAME = "sipi_dataset_cleaned.csv"
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
    FREQUENCY_COLUMN = "FREQ_GHZ"
    SIMULATION_COLUMN = "SIMU_INDEX"
    TOUCHSTONE_COLUMN = "TOUCHSTONE_REL_PATH"
    SPLIT_COLUMN = "SPLIT_TYPE"
    CLEANED_COLUMNS = (
        *PARAMETER_COLUMNS,
        FREQUENCY_COLUMN,
        SIMULATION_COLUMN,
        TOUCHSTONE_COLUMN,
        SPLIT_COLUMN,
    )

    def __init__(
        self,
        raw_data: RawData,
        processed_dir: Path | str,
        feature_columns: Sequence[str] | None = None,
    ) -> None:
        """
        Configure cleaned CSV construction.
        """
        if not isinstance(raw_data, RawData):
            raise TypeError("raw_data must be a RawData instance.")
        self.raw_data = raw_data
        self.processed_dir = Path(processed_dir)
        default_features = DLDataset.DEFAULT_FEATURE_COLUMNS
        self.feature_columns = tuple(feature_columns or default_features)
        self._validate_feature_columns()

    @property
    def cleaned_path(self) -> Path:
        """
        Return the path to the cleaned CSV artifact.
        """
        return self.processed_dir / self.CLEANED_FILENAME

    def data_cleaning(self, force: bool = False) -> pd.DataFrame:
        """
        Build or load the cleaned design-frequency dataframe.

        Set ``force=True`` to rebuild even when a valid cached CSV exists.
        """
        if self.cleaned_path.is_file() and not force:
            cleaned = pd.read_csv(self.cleaned_path)
            return self._validate_cleaned_dataframe(cleaned)

        parameters = PcbParameters(self.raw_data.parameter_csv)
        parameter_frame = self._valid_parameter_frame(parameters)
        aligned = self._aligned_parameter_frame(parameter_frame)
        frequencies_ghz = self._frequency_grid_for(aligned)

        rows: list[dict[str, object]] = []
        for _, design in aligned.iterrows():
            simulation_index = int(design[self.SIMULATION_COLUMN])
            touchstone_rel_path = self._relative_touchstone_path(
                self.raw_data.touchstone(simulation_index)
            )
            for frequency_ghz in frequencies_ghz:
                row: dict[str, object] = {
                    column: float(design[column]) for column in self.PARAMETER_COLUMNS
                }
                row[self.FREQUENCY_COLUMN] = float(frequency_ghz)
                row[self.SIMULATION_COLUMN] = simulation_index
                row[self.TOUCHSTONE_COLUMN] = touchstone_rel_path
                row[self.SPLIT_COLUMN] = ""
                rows.append(row)

        cleaned = pd.DataFrame(rows, columns=self.CLEANED_COLUMNS)
        cleaned = self._validate_cleaned_dataframe(cleaned)
        self._write_cleaned(cleaned)
        return cleaned

    def split(
        self,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 42,
        force: bool = False,
    ) -> tuple[DLDataset, DLDataset, DLDataset]:
        """
        Assign deterministic split labels and return split-specific datasets.

        Fractions are design-level proportions in the open interval ``(0, 1)``.
        """
        cleaned = self.data_cleaning(force=force).copy()
        simulation_indices = self._unique_simulation_indices(cleaned)
        split_by_index = self._split_labels_by_index(
            simulation_indices,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        cleaned[self.SPLIT_COLUMN] = cleaned[self.SIMULATION_COLUMN].map(split_by_index)
        cleaned = self._validate_cleaned_dataframe(cleaned, require_split=True)
        self._write_cleaned(cleaned)
        return (
            DLDataset(cleaned, self.feature_columns, "train"),
            DLDataset(cleaned, self.feature_columns, "val"),
            DLDataset(cleaned, self.feature_columns, "test"),
        )

    def _validate_feature_columns(self) -> None:
        """
        Validate feature columns for returned ``DLDataset`` objects.
        """
        if not self.feature_columns:
            raise ValueError("feature_columns must contain at least one column.")
        allowed = {*self.PARAMETER_COLUMNS, self.FREQUENCY_COLUMN}
        unknown = [column for column in self.feature_columns if column not in allowed]
        if unknown:
            raise ValueError("Unsupported feature columns: " + ", ".join(unknown))

    def _valid_parameter_frame(self, parameters: PcbParameters) -> pd.DataFrame:
        """
        Validate and normalize the source parameter table.
        """
        frame = parameters.dataframe.copy()
        required = [*self.PARAMETER_COLUMNS, self.SIMULATION_COLUMN]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError("parameter.csv is missing columns: " + ", ".join(missing))

        raw_indices = pd.to_numeric(frame[self.SIMULATION_COLUMN], errors="coerce")
        if raw_indices.isna().any():
            raise ValueError("SIMU_INDEX values must be numeric.")
        integer_indices = raw_indices.astype(np.int64)
        if not np.allclose(raw_indices.to_numpy(dtype=float), integer_indices):
            raise ValueError("SIMU_INDEX values must be integer-valued.")
        if integer_indices.duplicated().any():
            raise ValueError("SIMU_INDEX values must be unique.")
        frame[self.SIMULATION_COLUMN] = integer_indices

        features = frame.loc[:, self.PARAMETER_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("Parameter feature values must be finite.")
        frame.loc[:, self.PARAMETER_COLUMNS] = features
        return frame

    def _aligned_parameter_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Drop parameter rows without Touchstone files and ignore orphan files.
        """
        self.raw_data.check_index_consistency()
        touchstone_indices = set(self.raw_data.touchstone_indices())
        aligned = frame.loc[
            frame[self.SIMULATION_COLUMN].isin(touchstone_indices)
        ].copy()
        if aligned.empty:
            raise ValueError("No parameter rows have matching Touchstone files.")
        return aligned.reset_index(drop=True)

    def _frequency_grid_for(self, aligned: pd.DataFrame) -> np.ndarray:
        """
        Read the common frequency grid from a representative Touchstone file.
        """
        simulation_index = int(aligned.iloc[0][self.SIMULATION_COLUMN])
        network = rf.Network(str(self.raw_data.touchstone(simulation_index)))
        if network.nports != self.raw_data.nports:
            raise ValueError(
                f"Touchstone for SIMU_INDEX {simulation_index} has "
                f"{network.nports} ports; expected {self.raw_data.nports}."
            )
        frequencies_ghz = np.asarray(network.f, dtype=float) / 1e9
        if frequencies_ghz.ndim != 1 or len(frequencies_ghz) == 0:
            raise ValueError("Touchstone frequency grid must contain values.")
        if not np.isfinite(frequencies_ghz).all():
            raise ValueError("Touchstone frequency grid contains non-finite values.")
        if np.any(np.diff(frequencies_ghz) <= 0):
            raise ValueError("Touchstone frequency grid must be strictly increasing.")
        return frequencies_ghz

    def _relative_touchstone_path(self, touchstone_path: Path) -> str:
        """
        Convert a Touchstone path to a portable relative metadata path.
        """
        resolved = touchstone_path.resolve()
        for root in (PROJECT_ROOT.resolve(), self.raw_data.path.resolve().parent):
            try:
                return resolved.relative_to(root).as_posix()
            except ValueError:
                continue
        return resolved.as_posix()

    def _validate_cleaned_dataframe(
        self,
        cleaned: pd.DataFrame,
        *,
        require_split: bool = False,
    ) -> pd.DataFrame:
        """
        Validate the cleaned CSV schema and values.
        """
        missing = [column for column in self.CLEANED_COLUMNS if column not in cleaned]
        if missing:
            raise ValueError("Required columns missing: " + ", ".join(missing))
        validated = cleaned.reindex(columns=self.CLEANED_COLUMNS).copy()
        numeric_cols = [*self.PARAMETER_COLUMNS, self.FREQUENCY_COLUMN]
        validated.loc[:, numeric_cols] = validated.loc[:, numeric_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if not np.isfinite(validated.loc[:, numeric_cols].to_numpy(dtype=float)).all():
            raise ValueError("Cleaned feature and frequency values must be finite.")
        raw_indices = pd.to_numeric(validated[self.SIMULATION_COLUMN], errors="coerce")
        if raw_indices.isna().any():
            raise ValueError("Cleaned SIMU_INDEX values must be numeric.")
        integer_indices = raw_indices.astype(np.int64)
        if not np.allclose(
            raw_indices.to_numpy(dtype=float),
            integer_indices,
        ):
            raise ValueError("Cleaned SIMU_INDEX values must be integer-valued.")
        validated[self.SIMULATION_COLUMN] = integer_indices
        if validated[self.TOUCHSTONE_COLUMN].astype(str).str.len().eq(0).any():
            raise ValueError("TOUCHSTONE_REL_PATH values must be non-empty.")
        if require_split:
            allowed = {"train", "val", "test"}
            labels = set(validated[self.SPLIT_COLUMN].astype(str))
            if labels != allowed:
                raise ValueError("SPLIT_TYPE must contain train, val, and test labels.")
        return validated

    def _write_cleaned(self, cleaned: pd.DataFrame) -> None:
        """
        Persist the cleaned dataframe.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(self.cleaned_path, index=False)

    def _unique_simulation_indices(self, cleaned: pd.DataFrame) -> np.ndarray:
        """
        Return unique design indices in cleaned-data order.
        """
        return np.asarray(
            pd.unique(cleaned[self.SIMULATION_COLUMN]),
            dtype=np.int64,
        )

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
        n_test = self._fraction_to_count(
            test_fraction,
            n_designs,
            "test_fraction",
        )
        n_val = self._fraction_to_count(
            val_fraction,
            n_designs,
            "val_fraction",
        )
        if n_test + n_val >= n_designs:
            raise ValueError(
                "test_fraction and val_fraction must leave at least "
                "one training design."
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

    @staticmethod
    def _fraction_to_count(fraction: float, n_designs: int, name: str) -> int:
        """
        Convert a split fraction to a design count.
        """
        fraction = float(fraction)
        if fraction <= 0.0 or fraction >= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
        count = int(round(n_designs * fraction))
        if count <= 0:
            raise ValueError(f"{name} selects no simulation indices.")
        if count >= n_designs:
            raise ValueError(f"{name} must be smaller than the design count.")
        return count
