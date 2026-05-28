#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the find_project_root function in sparam_surrogate.config.
"""

import pytest
from pathlib import Path
from sparam_surrogate.config import find_project_root, notebook_resource_path

def test_find_project_root_from_project_root(tmp_path: Path):
    """
    Test that find_project_root correctly identifies the project root when given
    the path to the project root itself.
    """

    project_root = tmp_path / "fake_project"
    pyproject = project_root / "pyproject.toml"

    project_root.mkdir()
    pyproject.write_text("[project]\nname = 'fake-project'\n", encoding="utf-8")

    result = find_project_root(project_root)
    assert result == project_root.resolve()


def test_find_project_root_from_nested_directory(tmp_path: Path):
    """
    Test that find_project_root correctly identifies the project root when given
    a nested directory within the project structure.
    """

    project_root = tmp_path / "fake_project"
    nested_dir = project_root / "src" / "sparam_surrogate" / "data"
    pyproject = project_root / "pyproject.toml"

    nested_dir.mkdir(parents=True)
    pyproject.write_text("[project]\nname = 'fake-project'\n", encoding="utf-8")

    result = find_project_root(nested_dir)
    assert result == project_root.resolve()


def test_find_project_root_from_outer_workspace(tmp_path: Path):
    """
    Test that find_project_root recognises the repository when the current
    workspace is one directory above it.
    """

    workspace_root = tmp_path / "workspace"
    project_root = workspace_root / "sparam-surrogate"
    pyproject = project_root / "pyproject.toml"

    project_root.mkdir(parents=True)
    pyproject.write_text("[project]\nname = 'sparam-surrogate'\n", encoding="utf-8")

    result = find_project_root(workspace_root)
    assert result == project_root.resolve()


def test_find_project_root_raises_error_when_no_pyproject(tmp_path: Path):
    """
    Test that find_project_root raises a RuntimeError when no pyproject.toml
    file is found in the directory hierarchy.
    """

    some_dir = tmp_path / "not_a_project"
    some_dir.mkdir()

    with pytest.raises(RuntimeError, match="Project root not found"):
        find_project_root(some_dir)


def test_find_project_root_uses_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Test that find_project_root correctly identifies the project root when
    called without an explicit starting path, relying on the current working
    directory.
    """

    project_root = tmp_path / "fake_project"
    notebook_dir = project_root / "notebooks"
    pyproject = project_root / "pyproject.toml"

    notebook_dir.mkdir(parents=True)
    pyproject.write_text("[project]\nname = 'fake-project'\n", encoding="utf-8")

    monkeypatch.chdir(notebook_dir)
    result = find_project_root()
    assert result == project_root.resolve()


class TestNotebookResourcePath:
    """
    Tests for resolving notebook resources from the resource directory.
    """

    def test_relative_to_search_path(self, tmp_path: Path) -> None:
        """
        Resolve resource paths independently of the current working directory.
        """
        search_path = tmp_path / "resources"
        resource = search_path / "figure.png"
        search_path.mkdir()
        resource.touch()

        result = notebook_resource_path(
            "figure.png",
            search_path=search_path,
        )

        assert result == resource

    def test_absolute_resource_path(self, tmp_path: Path) -> None:
        """
        Leave absolute paths independent of the project root.
        """
        resource = tmp_path / "figure.png"
        resource.touch()

        result = notebook_resource_path(resource, search_path=tmp_path / "resources")

        assert result == resource

    def test_raises_when_resource_is_missing(self, tmp_path: Path) -> None:
        """
        Report missing resources with the requested path.
        """
        with pytest.raises(FileNotFoundError, match="missing.png"):
            notebook_resource_path("missing.png", search_path=tmp_path)
