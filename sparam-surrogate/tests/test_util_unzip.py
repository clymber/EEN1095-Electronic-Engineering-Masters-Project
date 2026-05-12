#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the unzip function in sparam_surrogate.utils.unzip.
"""
import pytest
from sparam_surrogate.utils import unzip
from pathlib import Path
from zipfile import ZipFile

class TestUnzip:
    """
    Tests for the unzip function.
    """
    tmp_path: Path

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        self.tmp_path = tmp_path

    def _create_zip(self, dest_zip: Path, files: dict[str, bytes]) -> None:
        dest_zip.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(dest_zip, 'w') as archive:
            for filename, content in files.items():
                archive.writestr(filename, content)

    def test_derive_subdir_from_zip_name(self):
        """
        Test that the correct subdirectory is derived from the zip file name.
        """
        zip_path = self.tmp_path / "test.zip"
        out_dir = self.tmp_path
        self._create_zip(zip_path, {"file.txt": b"Hello, world!"})

        dst_dir = unzip.extract_zip(zip_path, out_dir, True)
        assert dst_dir == self.tmp_path / "test"
        assert (dst_dir / "file.txt").is_file()

    def test_not_derive_subdir_if_requested(self):
        """
        Explicitly requested not to derive a subdirectory from the zip file name.
        """
        zip_path = self.tmp_path / "test.zip"
        out_dir = self.tmp_path / "output"
        self._create_zip(zip_path, {"file.txt": b"Hello, world!"})

        dst_dir = unzip.extract_zip(zip_path, out_dir, False)
        assert dst_dir == out_dir
        assert (dst_dir / "file.txt").is_file()

    def test_subdir_specified_in_outdir(self):
        """
        If the name of subdirectory is the same with @outdir, subdirectory will
        not be derived from the zip file name.
        """
        zip_path = self.tmp_path / "test.zip"
        out_dir = self.tmp_path / "test"
        self._create_zip(zip_path, {"file.txt": b"Hello, world!"})

        dst_dir = unzip.extract_zip(zip_path, out_dir, True)
        assert dst_dir == out_dir
        assert (dst_dir / "file.txt").is_file()

    def test_rejects_path_traversal_attack(self):
        """
        Raises an ValueError when a ZIP file contains entries that attempt to
        traverse the filesystem.
        """
        zip_path = self.tmp_path / "malicious.zip"
        out_dir = self.tmp_path / "output"
        self._create_zip(zip_path, {"../evil.txt": b"Malicious content"})

        with pytest.raises(ValueError, match="Unsafe ZIP member path: ../evil.txt"):
            unzip.extract_zip(zip_path, out_dir, False)

