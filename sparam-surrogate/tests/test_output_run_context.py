#!/usr/bin/env python3
"""
Tests for run-local config and environment context files.
"""

from pathlib import Path

from sparam_surrogate.config.surrogate_config import (
    DatasetConfig,
    PathsConfig,
    PreprocessingConfig,
    ProjectConfig,
    SurrogateConfig,
)
from sparam_surrogate.outputs.runs import (
    ModelRunArtifactManager,
    build_environment_metadata,
    build_resolved_config,
    create_run_artifact_dirs,
    save_run_config,
    save_run_environment,
    save_run_split_summary,
)
from sparam_surrogate.utils.json_io import read_json


def _manager(tmp_path: Path) -> ModelRunArtifactManager:
    """
    Return a deterministic run artifact manager for tests.
    """
    return ModelRunArtifactManager.create(
        tmp_path / "runs",
        "scalar_ridge",
        timestamp="20260705T153000Z",
    )


def _config(tmp_path: Path) -> SurrogateConfig:
    """
    Return a small resolved surrogate configuration.
    """
    return SurrogateConfig(
        project=ProjectConfig(name="demo-project", seed=123),
        paths=PathsConfig(
            raw_data=tmp_path / "data" / "raw",
            processed_data=tmp_path / "data" / "processed",
            outputs=tmp_path / "outputs",
            benchmarks=tmp_path / "outputs" / "benchmarks",
            logs=tmp_path / "outputs" / "logs",
            figures=tmp_path / "outputs" / "figures",
            models=tmp_path / "outputs" / "models",
            reports=tmp_path / "outputs" / "reports",
            runs=tmp_path / "outputs" / "runs",
        ),
        dataset=DatasetConfig(
            name="demo-dataset",
            path=tmp_path / "data" / "raw" / "demo-dataset",
            parameter_csv=tmp_path / "data" / "raw" / "demo-dataset" / "params.csv",
            nports=2,
            ports=((1, 2),),
        ),
        preprocessing=PreprocessingConfig(
            cleaned_splits_csv=(
                tmp_path / "data" / "processed" / "cleaned_splits_parameter.csv"
            ),
            freq_expanded_csv=(
                tmp_path / "data" / "processed" / "frequency_expanded_dataset.csv"
            ),
            val_fraction=0.2,
            test_fraction=0.1,
        ),
    )


def test_save_config_writes_json_ready_resolved_config(tmp_path: Path) -> None:
    """
    Resolved config snapshots save as plain JSON values.
    """
    manager = _manager(tmp_path)

    path = save_run_config(manager.run_dir, _config(tmp_path))
    data = read_json(path)

    assert path == manager.run_dir / "config_resolved.json"
    assert data["schema_version"] == 1
    assert data["config"]["project"] == {"name": "demo-project", "seed": 123}
    assert data["config"]["paths"]["runs"] == "outputs/runs"
    assert data["config"]["dataset"]["parameter_csv"] == (
        "data/raw/demo-dataset/params.csv"
    )
    assert data["config"]["dataset"]["ports"] == [[1, 2]]
    assert data["config"]["preprocessing"]["val_fraction"] == 0.2


def test_build_resolved_config_writes_project_relative_paths(
    tmp_path: Path,
) -> None:
    """
    Resolved config snapshots shorten paths beneath the project root.
    """
    data = build_resolved_config(_config(tmp_path))
    config = data["config"]

    assert config["paths"] == {
        "benchmarks": "outputs/benchmarks",
        "figures": "outputs/figures",
        "logs": "outputs/logs",
        "models": "outputs/models",
        "outputs": "outputs",
        "processed_data": "data/processed",
        "raw_data": "data/raw",
        "reports": "outputs/reports",
        "runs": "outputs/runs",
    }
    assert config["dataset"]["path"] == "data/raw/demo-dataset"
    assert config["dataset"]["parameter_csv"] == (
        "data/raw/demo-dataset/params.csv"
    )
    assert config["preprocessing"]["cleaned_splits_csv"] == (
        "data/processed/cleaned_splits_parameter.csv"
    )
    assert config["preprocessing"]["freq_expanded_csv"] == (
        "data/processed/frequency_expanded_dataset.csv"
    )


def test_save_environment_writes_runtime_versions(tmp_path: Path) -> None:
    """
    Environment snapshots include Python, platform, and core package versions.
    """
    manager = _manager(tmp_path)

    path = save_run_environment(manager.run_dir)
    data = read_json(path)

    assert path == manager.run_dir / "environment.json"
    assert data["schema_version"] == 1
    assert data["python"]["version"]
    assert data["platform"]["platform"]
    assert data["packages"]["numpy"]
    assert data["packages"]["pandas"]
    assert data["packages"]["scikit-learn"]


def test_environment_metadata_skips_missing_packages() -> None:
    """
    Missing optional package versions are skipped.
    """
    data = build_environment_metadata(
        package_names={"missing": "definitely-not-installed-package-name"}
    )

    assert data["packages"] == {}


def test_save_split_summary_writes_csv(tmp_path: Path) -> None:
    """
    Split summary data saves as a simple CSV table.
    """
    manager = _manager(tmp_path)

    path = save_run_split_summary(
        manager.run_dir,
        {
            "train": {"rows": 80, "unique_designs": 40},
            "validation": {"rows": 10, "unique_designs": 10},
            "test": {"rows": 10, "unique_designs": 10},
        }
    )

    assert path == manager.run_dir / "split_summary.csv"
    assert path.read_text(encoding="utf-8").splitlines() == [
        "split,rows,unique_designs",
        "train,80,40",
        "validation,10,10",
        "test,10,10",
    ]


def test_create_run_artifact_dirs_creates_reserved_directories(
    tmp_path: Path,
) -> None:
    """
    Run-local figure and reserved prediction directories are created together.
    """
    manager = _manager(tmp_path)

    paths = create_run_artifact_dirs(manager.run_dir)

    assert paths == {
        "figures": manager.run_dir / "figures",
        "predictions": manager.run_dir / "predictions",
    }
    assert paths["figures"].is_dir()
    assert paths["predictions"].is_dir()
    assert not list(paths["predictions"].iterdir())


def test_manifest_references_context_files_and_reserved_dirs(
    tmp_path: Path,
) -> None:
    """
    Manifest artifact discovery includes run-context files and directories.
    """
    manager = _manager(tmp_path)
    save_run_config(manager.run_dir, _config(tmp_path))
    save_run_environment(manager.run_dir)
    save_run_split_summary(manager.run_dir, {"train": {"rows": 1}})
    create_run_artifact_dirs(manager.run_dir)

    path = manager.save_manifest(completed_steps=("persist",))
    manifest = read_json(path)

    assert manifest["artifacts"] == {
        "config": "config_resolved.json",
        "environment": "environment.json",
        "figures": "figures",
        "predictions": "predictions",
        "split_summary": "split_summary.csv",
    }
    assert (manager.run_dir / manifest["artifacts"]["split_summary"]).is_file()
    assert (manager.run_dir / manifest["artifacts"]["figures"]).is_dir()
    assert (manager.run_dir / manifest["artifacts"]["predictions"]).is_dir()
