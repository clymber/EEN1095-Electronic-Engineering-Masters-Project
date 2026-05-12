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
Unzip utility for extracting ZIP archives.
"""
from pathlib import Path
from zipfile import ZipFile


# %%
def extract_zip(srcfile: Path, outdir: Path, keep_zipname: bool = True) -> Path:
    """
    Extract a ZIP archive to the output directory.

    - srcfile: Path to the ZIP file to extract.
    - outdir: Directory where the contents will be extracted.
    - keep_zipname: If True, creates a subdirectory named after the ZIP file
                    (without extension) inside the output directory. If False,
                    extracts directly into the output directory.
    Returns the path to the directory where the contents were extracted.
    """
    srcfile = srcfile.expanduser().resolve()
    outdir = outdir.expanduser().resolve()

    # Creates a subdirectory named after the ZIP file if needed.
    if keep_zipname and outdir.name != srcfile.stem:
        outdir = outdir / srcfile.stem

    # Creates the output directory if it doesn't exist.
    outdir.mkdir(parents=True, exist_ok=True)

    # Security check to prevent path traversal attacks.
    with ZipFile(srcfile) as archive:
        for member in archive.infolist():
            target_path = (outdir / member.filename).resolve()
            if outdir != target_path and outdir not in target_path.parents:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")

        archive.extractall(outdir)

    return outdir
