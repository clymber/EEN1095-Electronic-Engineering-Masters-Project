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
Application runtime paths helper and config.
"""
from pathlib import Path


# %%
def find_project_root(start: Path | None = None) -> Path:
    """
    Find project root by looking for pyproject.toml file.
    """
    current = (start or Path.cwd()).resolve()

    for path in [current, *current.parents]:
        if (path / "pyproject.toml").is_file():
            return path

    raise RuntimeError("Project root not found")


# %% [markdown]
# Basic runtime directory configuration:

# %%
PROJECT_ROOT = find_project_root()
