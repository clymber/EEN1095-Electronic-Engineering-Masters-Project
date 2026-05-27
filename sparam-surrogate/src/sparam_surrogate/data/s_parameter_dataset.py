"""
Aligned, compact S-parameter responses extracted from Touchstone files.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .pcb_parameters import PcbParameters
from .raw_data import IndexConsistencyReport, RawData


class SParameterDataset:
    """
    Frequency-dependent transmission responses aligned to PCB simulations.

    Selected S-parameter paths are stored as magnitude in dB with shape
    ``(simulation, frequency, path)``. Port-pair tuples use Touchstone's
    one-based ``(receiver, source)`` convention.
    """

    CACHE_SCHEMA_VERSION = 1

    def __init__(
        self,
        simulation_indices: Sequence[int] | np.ndarray,
        frequencies_ghz: Sequence[float] | np.ndarray,
        port_pairs: Sequence[tuple[int, int]],
        through_s_db: np.ndarray,
        alignment_report: IndexConsistencyReport | None = None,
    ) -> None:
        self._simulation_indices = np.asarray(simulation_indices, dtype=np.int64)
        self._frequencies_ghz = np.asarray(frequencies_ghz, dtype=float)
        self._port_pairs = self._normalize_port_pairs(port_pairs)
        self._through_s_db = np.asarray(through_s_db, dtype=float)
        self._alignment_report = alignment_report
        self._validate_arrays()

    @property
    def simulation_indices(self) -> np.ndarray:
        """
        Return simulation indices with both parameter and response records.
        """
        return self._simulation_indices

    @property
    def frequencies_ghz(self) -> np.ndarray:
        """
        Return the common Touchstone frequency grid in GHz.
        """
        return self._frequencies_ghz

    @property
    def port_pairs(self) -> tuple[tuple[int, int], ...]:
        """
        Return selected one-based ``(receiver, source)`` port paths.
        """
        return self._port_pairs

    @property
    def through_s_db(self) -> np.ndarray:
        """
        Return selected response curves in dB.
        """
        return self._through_s_db

    @property
    def alignment_report(self) -> IndexConsistencyReport | None:
        """
        Return the raw parameter/Touchstone consistency report, when available.
        """
        return self._alignment_report

    @classmethod
    def from_touchstones(
        cls,
        parameters: PcbParameters,
        raw_data: RawData,
        port_pairs: list[tuple[int, int]],
        *,
        cache_path: Path | None = None,
        rebuild_cache: bool = False,
    ) -> "SParameterDataset":
        """
        Parse aligned Touchstone files and extract selected S-parameters in dB.

        When ``cache_path`` exists and ``rebuild_cache`` is false, the compact
        extracted response array is loaded from cache after checking that its
        simulation alignment and selected paths still match this request.
        """
        normalized_pairs = cls._normalize_port_pairs(port_pairs)
        cls._validate_pairs_for_network(normalized_pairs, raw_data.nports)
        alignment_report = raw_data.check_index_consistency()
        simulation_indices = cls._aligned_simulation_indices(parameters, raw_data)

        if not simulation_indices:
            raise ValueError("No parameter records have matching Touchstone files.")

        if cache_path is not None and cache_path.is_file() and not rebuild_cache:
            return cls._from_cache(
                cache_path,
                simulation_indices,
                normalized_pairs,
                alignment_report,
            )

        try:
            import skrf as rf
        except ImportError as exc:
            raise ImportError(
                "Reading Touchstone files requires the project dependency 'scikit-rf'."
            ) from exc

        frequencies_ghz: np.ndarray | None = None
        responses: list[np.ndarray] = []
        for simulation_index in simulation_indices:
            network = rf.Network(str(raw_data.touchstone(simulation_index)))
            if network.nports != raw_data.nports:
                raise ValueError(
                    f"Touchstone for SIMU_INDEX {simulation_index} has "
                    f"{network.nports} ports; expected {raw_data.nports}."
                )

            network_frequencies_ghz = np.asarray(network.f, dtype=float) / 1e9
            if frequencies_ghz is None:
                frequencies_ghz = network_frequencies_ghz
            elif not np.allclose(
                network_frequencies_ghz,
                frequencies_ghz,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "Touchstone files do not share a common frequency grid; "
                    f"mismatch found at SIMU_INDEX {simulation_index}."
                )

            with np.errstate(divide="ignore"):
                response_db = network.s_db
                selected = np.column_stack(
                    [
                        response_db[:, receiver - 1, source - 1]
                        for receiver, source in normalized_pairs
                    ]
                )
            if not np.isfinite(selected).all():
                raise ValueError(
                    "Non-finite dB response found for "
                    f"SIMU_INDEX {simulation_index}."
                )
            responses.append(selected)

        assert frequencies_ghz is not None
        dataset = cls(
            simulation_indices,
            frequencies_ghz,
            normalized_pairs,
            np.stack(responses, axis=0),
            alignment_report,
        )
        if cache_path is not None:
            dataset._write_cache(cache_path)
        return dataset

    def at_frequency(self, frequency_ghz: float) -> pd.DataFrame:
        """
        Return selected dB responses at one frequency that exists in the grid.
        """
        matches = np.flatnonzero(
            np.isclose(self._frequencies_ghz, frequency_ghz, rtol=0.0, atol=1e-9)
        )
        if len(matches) == 0:
            raise ValueError(
                f"Frequency {frequency_ghz:g} GHz is not present in the "
                "Touchstone frequency grid."
            )

        values = self._through_s_db[:, int(matches[0]), :]
        response_frame = pd.DataFrame({"SIMU_INDEX": self._simulation_indices})
        for pair_index, pair in enumerate(self._port_pairs):
            response_frame[self.response_column(pair)] = values[:, pair_index]
        return response_frame

    @staticmethod
    def response_column(pair: tuple[int, int]) -> str:
        """
        Return the dataframe column name for one response port pair.
        """
        receiver, source = pair
        return f"S{receiver}_{source}_DB"

    @classmethod
    def _from_cache(
        cls,
        cache_path: Path,
        expected_indices: list[int],
        expected_pairs: tuple[tuple[int, int], ...],
        alignment_report: IndexConsistencyReport,
    ) -> "SParameterDataset":
        with np.load(cache_path, allow_pickle=False) as cache:
            schema_version = int(np.asarray(cache["schema_version"]).item())
            if schema_version != cls.CACHE_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported S-parameter cache schema version {schema_version}."
                )
            dataset = cls(
                cache["simulation_indices"],
                cache["frequencies_ghz"],
                [tuple(pair) for pair in cache["port_pairs"].tolist()],
                cache["through_s_db"],
                alignment_report,
            )

        if not np.array_equal(dataset.simulation_indices, expected_indices):
            raise ValueError(
                "Cached response alignment does not match current raw data; "
                "rebuild the cache."
            )
        if dataset.port_pairs != expected_pairs:
            raise ValueError(
                "Cached response paths do not match requested port pairs; "
                "rebuild the cache."
            )
        return dataset

    def _write_cache(self, cache_path: Path) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as cache_file:
            np.savez_compressed(
                cache_file,
                schema_version=np.asarray(self.CACHE_SCHEMA_VERSION, dtype=np.int64),
                simulation_indices=self._simulation_indices,
                frequencies_ghz=self._frequencies_ghz,
                port_pairs=np.asarray(self._port_pairs, dtype=np.int64),
                through_s_db=self._through_s_db,
            )

    @staticmethod
    def _normalize_port_pairs(
        port_pairs: Sequence[tuple[int, int]],
    ) -> tuple[tuple[int, int], ...]:
        normalized = tuple(
            (int(receiver), int(source)) for receiver, source in port_pairs
        )
        if not normalized:
            raise ValueError("At least one response port pair must be selected.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Response port pairs must be unique.")
        if any(receiver < 1 or source < 1 for receiver, source in normalized):
            raise ValueError("Response port pairs use one-based positive port numbers.")
        return normalized

    @staticmethod
    def _validate_pairs_for_network(
        port_pairs: tuple[tuple[int, int], ...], nports: int
    ) -> None:
        if any(receiver > nports or source > nports for receiver, source in port_pairs):
            raise ValueError(
                f"Response port pairs must reference ports between 1 and {nports}."
            )

    @staticmethod
    def _aligned_simulation_indices(
        parameters: PcbParameters, raw_data: RawData
    ) -> list[int]:
        if "SIMU_INDEX" not in parameters.dataframe.columns:
            raise ValueError("PCB parameters must include a SIMU_INDEX column.")

        raw_indices = parameters.dataframe["SIMU_INDEX"].to_numpy(dtype=float)
        integer_indices = raw_indices.astype(np.int64)
        if not np.allclose(raw_indices, integer_indices):
            raise ValueError("SIMU_INDEX values must be integer-valued.")
        if len(np.unique(integer_indices)) != len(integer_indices):
            raise ValueError("SIMU_INDEX values must be unique.")

        return [
            int(index)
            for index in integer_indices
            if raw_data.touchstone(int(index)).is_file()
        ]

    def _validate_arrays(self) -> None:
        if self._simulation_indices.ndim != 1:
            raise ValueError("Simulation indices must be one-dimensional.")
        if len(np.unique(self._simulation_indices)) != len(self._simulation_indices):
            raise ValueError("Simulation indices must be unique.")
        if self._frequencies_ghz.ndim != 1 or len(self._frequencies_ghz) == 0:
            raise ValueError(
                "Frequency grid must be a non-empty one-dimensional array."
            )
        if not np.isfinite(self._frequencies_ghz).all() or not np.all(
            np.diff(self._frequencies_ghz) > 0
        ):
            raise ValueError("Frequency grid must be finite and strictly increasing.")
        expected_shape = (
            len(self._simulation_indices),
            len(self._frequencies_ghz),
            len(self._port_pairs),
        )
        if self._through_s_db.shape != expected_shape:
            raise ValueError(
                "Response array has shape "
                f"{self._through_s_db.shape}; expected {expected_shape}."
            )
        if not np.isfinite(self._through_s_db).all():
            raise ValueError("Response array contains non-finite dB values.")
