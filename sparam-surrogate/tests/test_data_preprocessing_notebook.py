"""
Smoke checks for the data preprocessing notebook source.
"""

from sparam_surrogate.config import PROJECT_ROOT


def _notebook_source() -> str:
    """
    Return the paired Jupytext source for the preprocessing notebook.
    """
    path = PROJECT_ROOT / "notebooks" / "data_preprocessing.py"
    return path.read_text(encoding="utf-8")


def test_data_preprocessing_notebook_source_compiles() -> None:
    """
    Compile the notebook source without executing the full preprocessing run.
    """
    source = _notebook_source()
    compile(source, "notebooks/data_preprocessing.py", "exec")


def test_data_preprocessing_notebook_uses_production_pipeline_components() -> None:
    """
    Verify the notebook orchestrates production classes instead of defining them.
    """
    source = _notebook_source()

    assert "SParameterDataset.from_touchstones" in source
    assert "DesignFrequencySplitter(" in source
    assert "PcbFeatureTransformer(" in source
    assert "MLDatasetBuilder(" in source
    assert "build_scalar_dataset(" in source
    assert "build_full_smatrix_dataset(" in source
    assert "class " not in source
