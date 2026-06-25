"""
Lazy Touchstone target loading for TensorFlow dataset mapping.
"""

from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import skrf as rf

from sparam_surrogate.config import PROJECT_ROOT, SurrogateConfig


class TouchstoneLoader:
    """
    Callable map function that loads S-parameter targets from Touchstone files.

    Instances are suitable for use from ``DLDataset.to_tf_dataset(...)`` and
    can also be called directly in tests or small smoke checks.
    """

    FREQUENCY_TOLERANCE_GHZ = 1e-9

    def __init__(
        self,
        mode: Literal["scalar", "vector", "smatrix"],
        config: SurrogateConfig | Path | str | None = None,
        representation: Literal["db", "il", "real_imag", "complex"] = "db",
        cache_size: int = 256,
    ) -> None:
        """
        Configure lazy S-parameter target extraction.

        Scalar mode reads the first configured one-based port pair. Vector mode
        reads all configured port pairs. S-matrix mode returns the flattened
        full S-parameter matrix.
        """
        self.mode: str = self._normalise_mode(mode)
        self.project_root = PROJECT_ROOT.resolve()
        self.representation = str(representation)
        self.cache_size = int(cache_size)
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive.")
        if self.representation not in {"db", "il", "real_imag", "complex"}:
            raise ValueError(
                "representation must be 'db', 'il', 'real_imag', or 'complex'."
            )

        cfg = self._load_config(config)
        self.nports = cfg.dataset.nports
        self.port_pairs = cfg.dataset.ports
        self._cached_network = lru_cache(maxsize=self.cache_size)(self._read_network)

    @property
    def target_names(self) -> tuple[str, ...]:
        """
        Return target names in the same order as loader outputs.
        """
        if self.mode in {"scalar", "vector"}:
            return self._target_names_for_pairs(self._target_port_pairs())
        return self._target_names_for_pairs(self._smatrix_port_pairs(self.nports))

    def _target_names_for_pairs(
        self,
        port_pairs: Iterable[tuple[int, int]],
    ) -> tuple[str, ...]:
        """
        Return target names for port pairs in this loader's representation.
        """
        pairs = tuple(port_pairs)
        if self.representation == "db":
            return tuple(self.response_column_name(pair) for pair in pairs)
        if self.representation == "il":
            return tuple(
                f"IL_S{receiver}_{source}_DB" for receiver, source in pairs
            )
        if self.representation == "complex":
            return tuple(f"S{receiver}_{source}" for receiver, source in pairs)
        real_names = [f"REAL_S{receiver}_{source}" for receiver, source in pairs]
        imag_names = [f"IMAG_S{receiver}_{source}" for receiver, source in pairs]
        return tuple([*real_names, *imag_names])

    def _target_port_pairs(self) -> tuple[tuple[int, int], ...]:
        """
        Return the configured port pairs selected by the current mode.
        """
        if self.mode == "scalar":
            return self.port_pairs[:1]
        return self.port_pairs

    @staticmethod
    def _normalise_mode(
        mode: Literal["scalar", "vector", "smatrix"],
    ) -> str:
        """
        Return the canonical target-loading mode.
        """
        raw_mode = str(mode)
        if raw_mode in {"scalar", "vector", "smatrix"}:
            return raw_mode
        raise ValueError("mode must be 'scalar', 'vector', or 'smatrix'.")

    @staticmethod
    def _smatrix_port_pairs(nports: int) -> tuple[tuple[int, int], ...]:
        """
        Return all one-based S-matrix port pairs in flattened matrix order.
        """
        return tuple(
            (receiver, source)
            for receiver in range(1, nports + 1)
            for source in range(1, nports + 1)
        )

    @property
    def target_shape(self) -> tuple[int, ...]:
        """
        Return the one-sample target shape.
        """
        return (len(self.target_names),)

    def __call__(
        self,
        features: np.ndarray,
        row_metadata: Mapping[str, Any],
    ) -> np.ndarray:
        """
        Load the target for one design-frequency row.

        ``features`` is accepted to match ``DLDataset`` map callables; target
        lookup uses ``FREQ_GHZ`` and ``TOUCHSTONE_REL_PATH`` metadata.
        """
        _ = features
        path = self._resolve_path(row_metadata)
        network = self._network(path)
        if network.nports != self.nports:
            raise ValueError(
                f"Touchstone {path} has {network.nports} ports; expected {self.nports}."
            )
        frequency_ghz = self._metadata_frequency(row_metadata)
        frequency_index = self._target_frequency_index(network, frequency_ghz)
        if self.mode in {"scalar", "vector"}:
            return self._port_pair_target(network, frequency_index)
        return self._smatrix_target(network, frequency_index)

    def cache_info(self) -> object:
        """
        Return current Touchstone file cache statistics.
        """
        return self._cached_network.cache_info()

    def clear_cache(self) -> None:
        """
        Release cached Touchstone networks held by this loader.
        """
        self._cached_network.cache_clear()

    @staticmethod
    def response_column_name(pair: tuple[int, int]) -> str:
        """
        Return the scalar dB column name for one port pair.
        """
        receiver, source = pair
        return f"S{receiver}_{source}_DB"

    def _load_config(
        self,
        config: SurrogateConfig | Path | str | None,
    ) -> SurrogateConfig:
        """
        Load configuration from a typed config, JSON path, or project defaults.
        """
        if isinstance(config, SurrogateConfig):
            return config
        return SurrogateConfig.from_csv(config)

    def _resolve_path(self, row_metadata: Mapping[str, Any]) -> Path:
        """
        Resolve a metadata Touchstone path against the project root.
        """
        try:
            raw_path = str(row_metadata["TOUCHSTONE_REL_PATH"])
        except KeyError as exc:
            raise ValueError("row_metadata must contain TOUCHSTONE_REL_PATH.") from exc
        if not raw_path:
            raise ValueError("TOUCHSTONE_REL_PATH must be non-empty.")
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.project_root / path
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Touchstone file not found: {resolved}")
        return resolved

    def _metadata_frequency(self, row_metadata: Mapping[str, Any]) -> float:
        """
        Return the requested frequency from row metadata.
        """
        try:
            frequency_ghz = float(row_metadata["FREQ_GHZ"])
        except KeyError as exc:
            raise ValueError("row_metadata must contain FREQ_GHZ.") from exc
        return frequency_ghz

    def _network(self, path: Path) -> rf.Network:
        """
        Return a cached scikit-rf network for a Touchstone path.
        """
        return self._cached_network(path.as_posix())

    def _read_network(self, path: str) -> rf.Network:
        """
        Parse one Touchstone file from disk.
        """
        return rf.Network(path)

    def _target_frequency_index(self, network: rf.Network, frequency_ghz: float) -> int:
        """
        Locate the requested frequency in a Touchstone network.
        """
        frequencies_ghz = np.asarray(network.f, dtype=float) / 1e9
        if frequencies_ghz.ndim != 1 or not np.isfinite(frequencies_ghz).all():
            raise ValueError("Touchstone frequency grid is invalid.")
        matches = np.flatnonzero(
            np.isclose(
                frequencies_ghz,
                frequency_ghz,
                rtol=0.0,
                atol=self.FREQUENCY_TOLERANCE_GHZ,
            )
        )
        if len(matches) == 0:
            raise ValueError(
                f"Frequency {frequency_ghz:g} GHz is not present in the "
                "Touchstone frequency grid."
            )
        return int(matches[0])

    def _port_pair_target(
        self,
        network: rf.Network,
        frequency_index: int,
    ) -> np.ndarray:
        """
        Extract configured scalar or vector targets at one frequency.
        """
        values: list[complex] = []
        for receiver, source in self._target_port_pairs():
            if receiver > network.nports or source > network.nports:
                raise ValueError(
                    f"Configured port pair {(receiver, source)} is unavailable "
                    f"for a {network.nports}-port network."
                )
            values.append(network.s[frequency_index, receiver - 1, source - 1])
        return self._represent_complex_values(
            np.asarray(values, dtype=complex),
            f"{self.mode} target",
        )

    def _smatrix_target(
        self,
        network: rf.Network,
        frequency_index: int,
    ) -> np.ndarray:
        """
        Extract and flatten the complete S-matrix at one frequency.
        """
        matrix = np.asarray(network.s[frequency_index], dtype=complex)
        if not np.isfinite(matrix).all():
            raise ValueError("Full S-matrix target contains non-finite values.")
        return self._represent_complex_values(matrix.reshape(-1), "S-matrix target")

    def _represent_complex_values(
        self,
        complex_values: np.ndarray,
        label: str,
    ) -> np.ndarray:
        """
        Convert complex S-parameters to the configured representation.
        """
        if self.representation == "complex":
            if not np.isfinite(complex_values).all():
                raise ValueError(f"{label} contains non-finite values.")
            return complex_values
        if self.representation == "real_imag":
            target = np.concatenate([complex_values.real, complex_values.imag])
        else:
            with np.errstate(divide="ignore"):
                target = 20.0 * np.log10(np.abs(complex_values))
            if self.representation == "il":
                target = -target
        if not np.isfinite(target).all():
            raise ValueError(f"{label} contains non-finite values.")
        return np.asarray(target, dtype=float)
