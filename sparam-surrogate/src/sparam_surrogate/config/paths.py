# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: meng
#     language: python
#     name: python3
# ---

# %%
"""
Application runtime paths helper and config.
"""
from __future__ import annotations

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
        project_child = path / "sparam-surrogate"
        if (project_child / "pyproject.toml").is_file():
            return project_child

    raise RuntimeError("Project root not found")


def relative_to_project_root(
    path: Path | str,
    *,
    project_root: Path | str | None = None,
) -> str:
    """
    Return a stable POSIX-style path relative to the project root.

    Absolute paths under the project root are shortened for reproducible
    notebook and log output. Relative paths are interpreted from the project
    root. Paths outside the project root are rejected because they cannot be
    represented portably as project-relative paths.
    """
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate

    resolved = candidate.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Path is not inside project root: {resolved}"
        ) from exc


# %% [markdown]
# Basic runtime directory configuration:

# %%
PROJECT_ROOT = find_project_root()
NOTEBOOK_RESOURCE_DIR = PROJECT_ROOT / "notebooks" / "resources"

# %%
def notebook_resource_path(
    filename: Path | str,
    *,
    search_path: Path | str = NOTEBOOK_RESOURCE_DIR,
) -> Path:
    """
    Return a notebook resource path resolved from the notebook resources folder.

    Resolving from ``NOTEBOOK_RESOURCE_DIR`` keeps image paths stable when
    Jupytext notebooks are rendered from different working directories.
    """
    resource = Path(filename)
    if resource.is_absolute():
        if resource.is_file():
            return resource
        raise FileNotFoundError(f"Notebook resource not found: {resource}")

    candidate = Path(search_path) / resource
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Notebook resource not found: {resource}. Checked: {candidate}"
    )
