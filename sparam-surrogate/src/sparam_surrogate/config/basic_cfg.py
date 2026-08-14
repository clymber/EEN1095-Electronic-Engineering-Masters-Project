"""
Basic runtime configurations.
"""

import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT

DEFAULT_OUTPUT_ROOT = "outputs"
DEFAULT_OUTPUT_CHILDREN = (
    "benchmarks",
    "logs",
    "figures",
    "models",
    "reports",
    "runs",
)


def _deep_update(base: MutableMapping[str, Any], override: dict) -> None:
    """
    Recursively update nested configuration dictionaries.
    """
    for key, value in override.items():
        base_value = base.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            _deep_update(base_value, value)
        else:
            base[key] = value


def _add_output_path_defaults(paths_cfg: MutableMapping[str, Any]) -> None:
    """
    Add default generated-output paths when configs omit them.
    """
    outputs = paths_cfg.setdefault("outputs", DEFAULT_OUTPUT_ROOT)
    outputs_path = Path(outputs)
    for child in DEFAULT_OUTPUT_CHILDREN:
        paths_cfg.setdefault(child, str(outputs_path / child))


def load_config(extra_cfg_path: str | Path | None = None) -> dict:
    """
    Load configuration from default, local, and optional extra JSON files.

    The configuration is loaded in the following order of precedence:
    1. Default configuration from "configs/default.json" in the project root.
    2. Local overrides from "configs/local.json" if it exists.
    3. Additional overrides from a user-specified config file if provided.
    """
    # Default to "configs/default.json" in the project root.
    default_cfg_path = PROJECT_ROOT / "configs/default.json"
    with default_cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # If "configs/local.json" exists, it will override the default config.
    local_cfg_path = PROJECT_ROOT / "configs/local.json"
    if local_cfg_path.is_file():
        with local_cfg_path.open("r", encoding="utf-8") as f:
            local_cfg = json.load(f)
        # Update the default config with local overrides.
        _deep_update(cfg, local_cfg)

    # If a specific config path is provided,
    # it will override both default and local configs.
    if extra_cfg_path is not None:
        extra_cfg_path = Path(extra_cfg_path)
        if not extra_cfg_path.is_absolute():
            extra_cfg_path = PROJECT_ROOT / extra_cfg_path

        with extra_cfg_path.open("r", encoding="utf-8") as f:
            extra_cfg = json.load(f)
            _deep_update(cfg, extra_cfg)

    _add_output_path_defaults(cfg.setdefault("paths", {}))

    # Resolve relative paths against project root
    for key, value in cfg.get("paths", {}).items():
        path = Path(value)
        if not path.is_absolute():
            cfg["paths"][key] = str((PROJECT_ROOT / path).resolve())

    return cfg
