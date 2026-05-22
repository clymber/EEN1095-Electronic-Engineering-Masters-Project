# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: meng
#     language: python
#     name: python3
# ---

# %%
"""
Basic runtime configurations.
"""
import json
from pathlib import Path
from .paths import PROJECT_ROOT


# %%
def load_config(extra_cfg_path: str | Path | None = None) -> dict:
    """
    Load basic configuration from a JSON file.

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
        cfg.update(local_cfg)

    # If a specific config path is provided, it will override both default and local configs.
    if extra_cfg_path is not None:
        extra_cfg_path = Path(extra_cfg_path)
        if not extra_cfg_path.is_absolute():
            extra_cfg_path = PROJECT_ROOT / extra_cfg_path

        with extra_cfg_path.open("r", encoding="utf-8") as f:
            extra_cfg = json.load(f)
            cfg.update(extra_cfg)

    # Resolve relative paths against project root
    for key, value in cfg.get("paths", {}).items():
        path = Path(value)
        if not path.is_absolute():
            cfg["paths"][key] = str((PROJECT_ROOT / path).resolve())

    return cfg
