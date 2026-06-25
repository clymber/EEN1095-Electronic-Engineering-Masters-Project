#!/usr/bin/env python3
"""
Tests for typed surrogate configuration resolution.
"""

import json
from pathlib import Path

from sparam_surrogate.config import basic_cfg
from sparam_surrogate.config.surrogate_config import SurrogateConfig


class TestSurrogateConfig:
    """
    Unit tests for ``SurrogateConfig``.
    """

    def test_from_csv_exposes_split_fractions_on_preprocessing(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """
        Split fractions belong to preprocessing, not training.
        """
        monkeypatch.setattr(basic_cfg, "PROJECT_ROOT", tmp_path)
        self._write_json(tmp_path / "configs" / "default.json", self._config_data())

        cfg = SurrogateConfig.from_csv()

        assert cfg.preprocessing.val_fraction == 0.2
        assert cfg.preprocessing.test_fraction == 0.1
        assert cfg.training.batch_size == 16
        assert cfg.training.epochs == 5
        assert not hasattr(cfg.training, "val_fraction")
        assert not hasattr(cfg.training, "test_fraction")

    def test_from_csv_rejects_invalid_dataset_nports(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """
        Dataset port count must be positive.
        """
        monkeypatch.setattr(basic_cfg, "PROJECT_ROOT", tmp_path)
        cfg_data = self._config_data()
        cfg_data["dataset"]["nports"] = 0
        self._write_json(tmp_path / "configs" / "default.json", cfg_data)

        try:
            SurrogateConfig.from_csv()
        except ValueError as exc:
            assert "dataset.nports must be positive" in str(exc)
        else:
            raise AssertionError("Expected invalid nports to raise ValueError.")

    def test_from_csv_rejects_invalid_dataset_port_pair(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """
        Dataset port pairs must use existing one-based ports.
        """
        monkeypatch.setattr(basic_cfg, "PROJECT_ROOT", tmp_path)
        cfg_data = self._config_data()
        cfg_data["dataset"]["ports"] = [[3, 1]]
        self._write_json(tmp_path / "configs" / "default.json", cfg_data)

        try:
            SurrogateConfig.from_csv()
        except ValueError as exc:
            assert "exceeds dataset.nports" in str(exc)
        else:
            raise AssertionError("Expected invalid port pair to raise ValueError.")

    def test_from_csv_rejects_wrong_length_dataset_port_pair(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """
        Dataset port pairs must contain exactly two ports.
        """
        monkeypatch.setattr(basic_cfg, "PROJECT_ROOT", tmp_path)
        cfg_data = self._config_data()
        cfg_data["dataset"]["ports"] = [[1, 2, 3]]
        self._write_json(tmp_path / "configs" / "default.json", cfg_data)

        try:
            SurrogateConfig.from_csv()
        except ValueError as exc:
            assert "must contain two ports" in str(exc)
        else:
            raise AssertionError("Expected malformed port pair to raise ValueError.")

    def _config_data(self) -> dict:
        """
        Return valid JSON-compatible test configuration data.
        """
        return {
            "project": {"name": "fake-project", "seed": 123},
            "paths": {
                "raw_data": "data/raw",
                "processed_data": "data/processed",
            },
            "dataset": {
                "name": "fake-dataset",
                "parameter_file": "parameter.csv",
                "nports": 2,
                "ports": [[1, 2]],
            },
            "preprocessing": {
                "cleaned_csv": "cleaned.csv",
                "val_fraction": 0.2,
                "test_fraction": 0.1,
            },
            "training": {
                "batch_size": 16,
                "epochs": 5,
            },
        }

    def _write_json(self, path: Path, data: dict) -> None:
        """
        Write JSON test configuration data.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
