"""
Tests for filesystem display utilities.
"""

from pathlib import Path

import pytest

from sparam_surrogate.utils.filesystem import directory_tree


class TestDirectoryTree:
    """
    Tests for rendering hierarchical directory previews.
    """

    @staticmethod
    def _make_directory_fixture(tmp_path: Path, file_paths: list[str]) -> Path:
        """
        Create a dataset directory containing files at the requested paths.
        """
        dataset = tmp_path / "dataset"
        for file_path in file_paths:
            path = dataset / file_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        return dataset

    def test_displays_directory_structure(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Keep all first-level archive entries visible while abbreviating large folders.
        """
        dataset = self._make_directory_fixture(
            tmp_path,
            [
                "description.pdf",
                "parameter.csv",
                "metadata/notes.txt",
                "variation/simu_0.s12p",
                "variation/simu_1.s12p",
                "variation/simu_2.s12p",
            ],
        )

        tree = directory_tree(dataset, max_depth=2, max_children=2)

        assert tree == "\n".join(
            [
                "dataset",
                "├── description.pdf",
                "├── parameter.csv",
                "├── metadata",
                "│   └── notes.txt",
                "└── variation",
                "    ├── simu_0.s12p",
                "    ├── simu_1.s12p",
                "    └── ... (1 more entry)",
            ]
        )

    def test_obeys_max_depth(self, tmp_path: Path) -> None:
        """
        Stop expansion at the requested nesting depth.
        """
        root = self._make_directory_fixture(
            tmp_path,
            ["variation/group/simu_0.s12p"],
        )

        assert directory_tree(root, max_depth=1) == "\n".join(
            [
                "dataset",
                "└── variation",
            ]
        )
        assert directory_tree(root, max_depth=2) == "\n".join(
            [
                "dataset",
                "└── variation",
                "    └── group",
            ]
        )

    def test_ignores_hidden_entries(self, tmp_path: Path) -> None:
        """
        Omit dot-prefixed files and directories at every displayed level.
        """
        dataset = self._make_directory_fixture(
            tmp_path,
            [
                ".DS_Store",
                ".cache/index",
                "parameter.csv",
                "variation/.metadata",
                "variation/simu_0.s12p",
            ],
        )

        assert directory_tree(dataset, max_depth=2) == "\n".join(
            [
                "dataset",
                "├── parameter.csv",
                "└── variation",
                "    └── simu_0.s12p",
            ]
        )

    def test_rejects_a_file_path(self, tmp_path: Path) -> None:
        """
        Reject paths that cannot be expanded as a directory tree.
        """
        file_path = tmp_path / "parameter.csv"
        file_path.touch()

        with pytest.raises(NotADirectoryError, match="Not a directory"):
            directory_tree(file_path)
