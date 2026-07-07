"""
Tests for JSON serialization utilities.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from sparam_surrogate.utils.json_io import json_ready, read_json, write_json


class TestJsonReady:
    """
    Tests for converting project values to JSON-friendly structures.
    """

    def test_converts_common_project_values(self, tmp_path: Path) -> None:
        """
        Convert mappings, sequences, paths, NumPy arrays, and NumPy scalars.
        """
        value = {
            1: tmp_path / "artifact.txt",
            "array": np.asarray([1.0, 2.0]),
            "nested": (np.int64(3), {"path": Path("outputs/runs")}),
        }

        assert json_ready(value) == {
            "1": (tmp_path / "artifact.txt").as_posix(),
            "array": [1.0, 2.0],
            "nested": [3, {"path": "outputs/runs"}],
        }


class TestWriteJson:
    """
    Tests for stable JSON writing.
    """

    def test_writes_sorted_indented_json_with_trailing_newline(
        self,
        tmp_path: Path,
    ) -> None:
        """
        JSON files are human-readable and reload to the expected data.
        """
        path = tmp_path / "metadata.json"

        written = write_json(path, {"z": np.int64(1), "a": Path("model.joblib")})

        assert path.read_text(encoding="utf-8") == (
            '{\n  "a": "model.joblib",\n  "z": 1\n}\n'
        )
        assert written == len(path.read_text(encoding="utf-8"))
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "a": "model.joblib",
            "z": 1,
        }
        assert read_json(path) == {
            "a": "model.joblib",
            "z": 1,
        }


class TestReadJson:
    """
    Tests for reading JSON objects.
    """

    def test_rejects_non_object_json(self, tmp_path: Path) -> None:
        """
        Files must contain JSON objects, not arrays or scalar values.
        """
        path = tmp_path / "values.json"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")

        with pytest.raises(TypeError, match="does not contain an object"):
            read_json(path)
