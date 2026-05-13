"""
Configuration utilities for the sparam-surrogate project.
"""
from .paths import PROJECT_ROOT, CONFIG_DIR, OUTPUT_DIR, LOGGING_DIR, DATA_DIR
from .paths import find_project_root, LOG_CFG_PATH
from .mylogging import set_logging_cfg, get_md_logger, MarkdownLogger, MarkdownFormatter

__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "OUTPUT_DIR",
    "LOGGING_DIR",
    "LOG_CFG_PATH",
    "DATA_DIR",
    "find_project_root",
    "set_logging_cfg",
    "get_md_logger",
    "MarkdownLogger",
    "MarkdownFormatter",
]
