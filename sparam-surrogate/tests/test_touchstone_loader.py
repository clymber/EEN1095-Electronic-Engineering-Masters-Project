"""
Tests for lazy Touchstone target loading.
"""

from pathlib import Path

import numpy as np
import pytest

import sparam_surrogate.data.touchstone_loader as touchstone_loader_module
from sparam_surrogate.config.surrogate_config import (
    DatasetConfig,
    PathsConfig,
    PreprocessingConfig,
    ProjectConfig,
    SurrogateConfig,
)
from sparam_surrogate.data import TouchstoneLoader


def _write_s2p(path: Path, matrices: list[np.ndarray]) -> None:
    """
    Write a two-frequency two-port Touchstone file in RI format.
    """
    lines = ["# GHz S RI R 50"]
    for offset, matrix in enumerate(matrices, start=1):
        values = [
            matrix[0, 0],
            matrix[1, 0],
            matrix[0, 1],
            matrix[1, 1],
        ]
        row = [str(offset)]
        for value in values:
            row.extend([f"{value.real:g}", f"{value.imag:g}"])
        lines.append(" ".join(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metadata(path: str, frequency_ghz: float = 1.0) -> dict[str, object]:
    """
    Return row metadata for one lazy target lookup.
    """
    return {
        "SIMU_INDEX": 0,
        "FREQ_GHZ": frequency_ghz,
        "TOUCHSTONE_REL_PATH": path,
    }


def _config(ports: tuple[tuple[int, int], ...] | None = None) -> SurrogateConfig:
    """
    Return a minimal typed loader configuration.
    """
    raw_data = Path("data/raw")
    dataset_name = "test-dataset"
    return SurrogateConfig(
        project=ProjectConfig(name="test-project", seed=128),
        paths=PathsConfig(raw_data=raw_data, processed_data=Path("data/processed")),
        dataset=DatasetConfig(
            name=dataset_name,
            path=raw_data / dataset_name,
            parameter_csv=raw_data / dataset_name / "parameter.csv",
            nports=2,
            ports=ports or ((2, 1), (1, 2)),
        ),
        preprocessing=PreprocessingConfig(
            processed_csv=Path("data/processed/sipi_dataset_cleaned.csv"),
            val_fraction=0.2,
            test_fraction=0.2,
        ),
    )


class TestTouchstoneLoader:
    """
    Unit tests for lazy target extraction from Touchstone files.
    """

    def test_load_scalar_db_target(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Scalar mode returns the first configured one-based port pair by default.
        """
        rel_path = "raw/variation/simu_0.s2p"
        matrices = [
            np.array([[0.1 + 0.0j, 0.2 + 0.0j], [0.5 + 0.0j, 0.3 + 0.0j]]),
            np.array([[0.1 + 0.0j, 0.2 + 0.0j], [0.25 + 0.0j, 0.3 + 0.0j]]),
        ]
        _write_s2p(tmp_path / rel_path, matrices)
        monkeypatch.setattr(touchstone_loader_module, "PROJECT_ROOT", tmp_path)
        loader = TouchstoneLoader(
            mode="scalar",
            config=_config(),
            representation="db",
        )

        target = loader(np.zeros(2), _metadata(rel_path, frequency_ghz=2.0))

        assert loader.target_names == ("S2_1_DB",)
        np.testing.assert_allclose(target, [20 * np.log10(0.25)])

    def test_load_vector_il_target(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Vector mode returns all configured port pairs as positive insertion loss.
        """
        rel_path = "raw/variation/simu_0.s2p"
        matrix = np.array([[0.1 + 0.0j, 0.2 + 0.0j], [0.5 + 0.0j, 0.3 + 0.0j]])
        _write_s2p(tmp_path / rel_path, [matrix])
        monkeypatch.setattr(touchstone_loader_module, "PROJECT_ROOT", tmp_path)
        loader = TouchstoneLoader(
            mode="vector",
            config=_config(),
            representation="il",
        )

        target = loader(np.zeros(2), _metadata(rel_path))

        assert loader.target_names == ("IL_S2_1_DB", "IL_S1_2_DB")
        np.testing.assert_allclose(
            target,
            [-20 * np.log10(0.5), -20 * np.log10(0.2)],
        )

    def test_load_smatrix_target(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        S-matrix mode returns all real components followed by all imaginary parts.
        """
        rel_path = "raw/variation/simu_0.s2p"
        matrix = np.array(
            [[1.0 + 0.1j, 2.0 + 0.2j], [3.0 + 0.3j, 4.0 + 0.4j]],
            dtype=complex,
        )
        _write_s2p(tmp_path / rel_path, [matrix])
        monkeypatch.setattr(touchstone_loader_module, "PROJECT_ROOT", tmp_path)
        loader = TouchstoneLoader(
            mode="smatrix",
            config=_config(),
            representation="real_imag",
        )

        target = loader(np.zeros(2), _metadata(rel_path))

        assert loader.target_names == (
            "REAL_S1_1",
            "REAL_S1_2",
            "REAL_S2_1",
            "REAL_S2_2",
            "IMAG_S1_1",
            "IMAG_S1_2",
            "IMAG_S2_1",
            "IMAG_S2_2",
        )
        np.testing.assert_allclose(target, [1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4])

    def test_frequency_lookup_uses_tolerance(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Frequency metadata can match within the documented tolerance.
        """
        rel_path = "raw/variation/simu_0.s2p"
        matrix = np.array([[0.1 + 0j, 0.2 + 0j], [0.5 + 0j, 0.3 + 0j]])
        _write_s2p(tmp_path / rel_path, [matrix])
        monkeypatch.setattr(touchstone_loader_module, "PROJECT_ROOT", tmp_path)
        loader = TouchstoneLoader("scalar", _config())

        target = loader(
            np.zeros(2),
            _metadata(rel_path, frequency_ghz=1.0 + 0.5e-9),
        )

        np.testing.assert_allclose(target, [20 * np.log10(0.5)])

    def test_caches_loaded_networks_by_touchstone_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Repeated rows for the same Touchstone path reuse the cached network.
        """
        rel_path = "raw/variation/simu_0.s2p"
        matrix = np.array([[0.1 + 0j, 0.2 + 0j], [0.5 + 0j, 0.3 + 0j]])
        _write_s2p(tmp_path / rel_path, [matrix])
        monkeypatch.setattr(touchstone_loader_module, "PROJECT_ROOT", tmp_path)
        loader = TouchstoneLoader(
            "scalar",
            _config(),
            cache_size=4,
        )

        loader(np.zeros(2), _metadata(rel_path))
        loader(np.zeros(2), _metadata(rel_path))

        info = loader.cache_info()
        assert info.misses == 1
        assert info.hits == 1
        assert info.maxsize == 4

    def test_clear_cache_releases_loaded_networks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Cached Touchstone networks can be released after eager materialization.
        """
        rel_path = "raw/variation/simu_0.s2p"
        matrix = np.array([[0.1 + 0j, 0.2 + 0j], [0.5 + 0j, 0.3 + 0j]])
        _write_s2p(tmp_path / rel_path, [matrix])
        monkeypatch.setattr(touchstone_loader_module, "PROJECT_ROOT", tmp_path)
        loader = TouchstoneLoader(
            "scalar",
            _config(),
            cache_size=4,
        )

        loader(np.zeros(2), _metadata(rel_path))
        assert loader.cache_info().currsize == 1

        loader.clear_cache()

        info = loader.cache_info()
        assert info.currsize == 0
        assert info.hits == 0
        assert info.misses == 0
