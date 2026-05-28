"""
Configuration utilities for the sparam-surrogate project.
"""
from .paths import PROJECT_ROOT
from .paths import NOTEBOOK_RESOURCE_DIR
from .paths import find_project_root
from .paths import notebook_resource_path
from .mylogging import set_logging_cfg, get_md_logger
from .mylogging import MarkdownLogger, MarkdownFormatter
from .basic_cfg import load_config

__all__ = [
    "PROJECT_ROOT",
    "NOTEBOOK_RESOURCE_DIR",
    "find_project_root",
    "notebook_resource_path",
    "set_logging_cfg",
    "get_md_logger",
    "MarkdownLogger",
    "MarkdownFormatter",
    "load_config",
]
