"""
Configuration utilities for the sparam-surrogate project.
"""
from .basic_cfg import load_config
from .mylogging import MarkdownFormatter, MarkdownLogger, get_md_logger, set_logging_cfg
from .paths import (
    NOTEBOOK_RESOURCE_DIR,
    PROJECT_ROOT,
    configure_stdio_relative_path,
    find_project_root,
    notebook_resource_path,
    relative_to_project_root,
)
from .surrogate_config import (
    ModelsConfig,
    PathsConfig,
    PreprocessingConfig,
    SurrogateConfig,
)

__all__ = [
    "PROJECT_ROOT",
    "NOTEBOOK_RESOURCE_DIR",
    "configure_stdio_relative_path",
    "find_project_root",
    "notebook_resource_path",
    "relative_to_project_root",
    "set_logging_cfg",
    "get_md_logger",
    "MarkdownLogger",
    "MarkdownFormatter",
    "load_config",
    "ModelsConfig",
    "PreprocessingConfig",
    "PathsConfig",
    "SurrogateConfig",
]
