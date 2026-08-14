#!/usr/bin/env python3
"""
Tests for run-local figure persistence.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from sparam_surrogate.outputs.runs import ModelRunArtifactManager, save_run_figure
from sparam_surrogate.utils.json_io import read_json


def _manager(tmp_path: Path) -> ModelRunArtifactManager:
    """
    Return a deterministic run artifact manager for tests.
    """
    return ModelRunArtifactManager.create(
        tmp_path / "runs",
        "neural_mlp",
        timestamp="20260705T153000Z",
    )


def _figure():
    """
    Return a tiny matplotlib figure.
    """
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [0.5, 0.25, 0.1])
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    return fig


def test_save_figure_writes_non_empty_png(tmp_path: Path) -> None:
    """
    Figures save under the run-local figures directory.
    """
    manager = _manager(tmp_path)
    fig = _figure()

    try:
        path = save_run_figure(manager.run_dir, fig, "training_history.png")
    finally:
        plt.close(fig)

    assert path == manager.run_dir / "figures" / "training_history.png"
    assert path.is_file()
    assert path.stat().st_size > 0


def test_manager_save_figure_uses_run_directory(tmp_path: Path) -> None:
    """
    The artifact manager saves figures into its own run directory.
    """
    manager = _manager(tmp_path)
    fig = _figure()

    try:
        path = manager.save_figure(fig, "validation_curves")
    finally:
        plt.close(fig)

    assert path == manager.run_dir / "figures" / "validation_curves.png"
    assert path.is_file()
    assert path.stat().st_size > 0


def test_manifest_discovers_saved_figures(tmp_path: Path) -> None:
    """
    Manifest records figure roles from saved PNG filenames.
    """
    manager = _manager(tmp_path)
    fig = _figure()

    try:
        save_run_figure(manager.run_dir, fig, "training_history.png")
    finally:
        plt.close(fig)

    path = manager.save_manifest(completed_steps=("persist",))
    manifest = read_json(path)

    assert manifest["artifacts"]["figures"] == "figures"
    assert manifest["figures"] == {
        "training_history": "figures/training_history.png"
    }
