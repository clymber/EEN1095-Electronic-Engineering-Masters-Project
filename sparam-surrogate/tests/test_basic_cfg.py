#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for basic configuration loading and path resolution.
"""
import json
from pathlib import Path
from sparam_surrogate.config import basic_cfg, load_config

class TestBasicConfig:
    """
    Unit tests for function: sparam_surrogate.config.basic_cfg.load_config
    """
    def _write_json(self, path: Path, data: dict) -> None:
        """
        Helper function to write a dictionary to a JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_load_default_cfg(self, tmp_path: Path, monkeypatch) -> None:
        """
        Test that load_config correctly loads the default configuration.
        """
        monkeypatch.setattr(basic_cfg, "PROJECT_ROOT", tmp_path)
        self._write_json(
            tmp_path / "configs" / "default.json",
            {
                "project": {"name": "fake"},
                "paths": {"data_root": "data"},
                "training": {"epochs": 10},
            }
        )

        cfg = load_config()
        assert isinstance(cfg, dict)
        assert cfg["project"]["name"] == "fake"
        assert cfg["training"]["epochs"] == 10
        assert cfg["paths"]["data_root"] == str((tmp_path / "data").resolve())

    def test_override_with_local_cfg(self, tmp_path: Path, monkeypatch) -> None:
        """
        If "configs/local.json" exists, it overrides the default configuration.
        """
        monkeypatch.setattr(basic_cfg, "PROJECT_ROOT", tmp_path)
        self._write_json(
            tmp_path / "configs" / "default.json",
            {
                "project": {"name": "fake"},
                "paths": {"data_root": "data"},
                "training": {"epochs": 10},
            }
        )
        self._write_json(
            tmp_path / "configs" / "local.json",
            {
                "project": {"name": "local_fake"},
                "paths": {"data_root": "local_data"},
                "training": {"epochs": 20},
            }
        )

        cfg = load_config()
        assert cfg["project"]["name"] == "local_fake"
        assert cfg["training"]["epochs"] == 20

    def test_override_with_extra_cfg(self, tmp_path: Path, monkeypatch) -> None:
        """
        If an extra config path is provided, it overrides both default and local configs.
        """
        monkeypatch.setattr(basic_cfg, "PROJECT_ROOT", tmp_path)
        self._write_json(
            tmp_path / "configs" / "default.json",
            {
                "project": {"name": "fake"},
                "paths": {"data_root": "data"},
                "training": {"epochs": 10},
            }
        )
        self._write_json(
            tmp_path / "configs" / "local.json",
            {
                "project": {"name": "local_fake"},
                "paths": {"data_root": "local_data"},
                "training": {"epochs": 20},
            }
        )
        extra_cfg_path = tmp_path / "extra_config.json"
        self._write_json(
            extra_cfg_path,
            {
                "project": {"name": "extra_fake"},
                "paths": {"data_root": "extra_data"},
                "training": {"epochs": 30},
                "other_setting": "value",
            }
        )

        cfg = load_config(extra_cfg_path)
        assert cfg["project"]["name"] == "extra_fake"
        assert cfg["training"]["epochs"] == 30
        assert cfg["other_setting"] == "value"
