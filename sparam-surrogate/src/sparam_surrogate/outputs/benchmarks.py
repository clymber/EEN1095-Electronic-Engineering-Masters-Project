"""
Benchmark summary CSV helpers derived from persisted model runs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from sparam_surrogate.outputs.models import ModelRegistry, ModelRegistryEntry
from sparam_surrogate.utils.filesystem import ensure_dir
from sparam_surrogate.utils.json_io import read_json

VECTOR_MAGNITUDE_DB = "vector_magnitude_db"
S7_1_MAGNITUDE_DB = "s7_1_magnitude_db"
PER_TARGET_MAGNITUDE_DB = "per_target_magnitude_db"
VECTOR_INSERTION_LOSS_DB = "vector_insertion_loss_db"
S7_1_INSERTION_LOSS_DB = "s7_1_insertion_loss_db"
PER_TARGET_INSERTION_LOSS_DB = "per_target_insertion_loss_db"
PER_TARGET_BENCHMARKS = {
    PER_TARGET_MAGNITUDE_DB,
    PER_TARGET_INSERTION_LOSS_DB,
}
METRIC_COLUMNS = ("val_mae_db", "val_rmse_db", "test_mae_db", "test_rmse_db")
DIAGNOSTIC_COLUMNS = (
    "deep_null_threshold_db",
    "high_frequency_threshold_ghz",
    "val_deep_null_mae_db",
    "test_deep_null_mae_db",
    "val_high_frequency_mae_db",
    "test_high_frequency_mae_db",
)


def refresh_benchmarks(
    benchmarks_root: Path | str,
    registry: ModelRegistry,
    model_name: str,
    *,
    selection: str = "latest",
    regenerate: bool = False,
) -> list[Path]:
    """
    Refresh compatible benchmark CSV rows for one registered model pointer.
    """
    if regenerate:
        return regenerate_benchmarks(
            benchmarks_root,
            registry,
            selections=(selection,),
        )

    if selection == "latest":
        entry = registry.latest(model_name)
    elif selection == "selected":
        entry = registry.selected(model_name)
    else:
        raise ValueError("selection must be 'latest' or 'selected'.")

    root = Path(benchmarks_root)
    written_paths: list[Path] = []
    for benchmark_name, rows in _benchmark_rows(registry, entry).items():
        path = root / f"{benchmark_name}_{selection}.csv"
        key_columns = (
            ("model_name", "target_name")
            if benchmark_name in PER_TARGET_BENCHMARKS
            else ("model_name",)
        )
        _upsert_rows(path, rows, key_columns=key_columns)
        written_paths.append(path)
    return written_paths


def regenerate_benchmarks(
    benchmarks_root: Path | str,
    registry: ModelRegistry,
    *,
    selections: tuple[str, ...] = ("latest", "selected"),
) -> list[Path]:
    """
    Rebuild benchmark CSVs from registry pointers and saved run metrics.
    """
    root = Path(benchmarks_root)
    written_paths: list[Path] = []
    for selection in selections:
        if selection not in {"latest", "selected"}:
            raise ValueError("selection must be 'latest' or 'selected'.")

        index_path = (
            registry.latest_path if selection == "latest" else registry.selected_path
        )
        if not index_path.is_file():
            continue

        index = read_json(index_path)
        models = index.get("models", {})
        if not isinstance(models, Mapping):
            continue

        rows_by_benchmark: dict[str, list[dict[str, Any]]] = {}
        entries = [
            ModelRegistryEntry.from_dict(entry)
            for entry in models.values()
            if isinstance(entry, Mapping)
        ]
        for entry in entries:
            for benchmark_name, rows in _benchmark_rows(registry, entry).items():
                rows_by_benchmark.setdefault(benchmark_name, []).extend(rows)

        for benchmark_name, rows in rows_by_benchmark.items():
            path = root / f"{benchmark_name}_{selection}.csv"
            ensure_dir(path.parent)
            _write_csv(pd.DataFrame(rows), path)
            written_paths.append(path)
    return written_paths


def _benchmark_rows(
    registry: ModelRegistry,
    entry: ModelRegistryEntry,
) -> dict[str, list[dict[str, Any]]]:
    """
    Return compatible benchmark rows for one registry entry.
    """
    metrics_path = registry.resolve_path(entry.metrics_path)
    if not metrics_path.is_file():
        return {}

    metrics = read_json(metrics_path).get("metrics", {})
    if not isinstance(metrics, Mapping):
        return {}

    metadata_path = registry.resolve_path(entry.metadata_path)
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    benchmark_contract = metrics.get("benchmark")
    if isinstance(benchmark_contract, Mapping):
        metrics = benchmark_contract
        metadata = {"data_interface": benchmark_contract}
    benchmark_names = _target_benchmark_names(metadata)
    if benchmark_names is None:
        return {}
    vector_benchmark, s7_1_benchmark, per_target_benchmark = benchmark_names

    rows_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    aggregate_row = _aggregate_row(entry, metrics)
    per_target_rows = _per_target_rows(entry, metrics)

    if aggregate_row is not None and _is_vector_benchmark(metadata):
        rows_by_benchmark[vector_benchmark] = [aggregate_row]

    s7_1_row = _s7_1_row(metadata, aggregate_row, per_target_rows)
    if s7_1_row is not None:
        rows_by_benchmark[s7_1_benchmark] = [s7_1_row]

    if per_target_rows:
        rows_by_benchmark[per_target_benchmark] = per_target_rows

    return rows_by_benchmark


def _target_benchmark_names(
    metadata: Mapping[str, Any],
) -> tuple[str, str, str] | None:
    """
    Return benchmark table names for a supported target representation.
    """
    data_interface = metadata.get("data_interface")
    if not isinstance(data_interface, Mapping):
        return None

    representation = str(
        data_interface.get("target_representation", "")
    ).lower()
    target_names = tuple(str(name) for name in data_interface.get("target_names", ()))
    insertion_loss_names = any(
        _normalise_target_name(name).startswith("il_s") for name in target_names
    )
    magnitude_names = any(
        _normalise_target_name(name).startswith("s") for name in target_names
    )

    if representation == "insertion_loss_db" or (
        not representation and insertion_loss_names
    ):
        return (
            VECTOR_INSERTION_LOSS_DB,
            S7_1_INSERTION_LOSS_DB,
            PER_TARGET_INSERTION_LOSS_DB,
        )
    if representation in {"magnitude_db", "log_magnitude_db"} or (
        not representation and magnitude_names
    ):
        return (
            VECTOR_MAGNITUDE_DB,
            S7_1_MAGNITUDE_DB,
            PER_TARGET_MAGNITUDE_DB,
        )
    return None


def _aggregate_row(
    entry: ModelRegistryEntry,
    metrics: Mapping[str, Any],
) -> dict[str, Any] | None:
    """
    Return one benchmark row from aggregate validation and test metrics.
    """
    values: dict[str, Any] = {
        "model_name": entry.model_name,
        "val_mae_db": _metric_value(metrics, "validation", "MAE"),
        "val_rmse_db": _metric_value(metrics, "validation", "RMSE"),
        "test_mae_db": _metric_value(metrics, "test", "MAE"),
        "test_rmse_db": _metric_value(metrics, "test", "RMSE"),
    }
    if any(values[key] is None for key in METRIC_COLUMNS):
        return None
    diagnostics = metrics.get("benchmark_diagnostics")
    if isinstance(diagnostics, Mapping):
        values.update(
            {
                column: float(diagnostics[column])
                for column in DIAGNOSTIC_COLUMNS
                if diagnostics.get(column) is not None
            }
        )
    values["run_id"] = entry.run_id
    return values


def _per_target_rows(
    entry: ModelRegistryEntry,
    metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Return per-target benchmark rows when per-target metrics are present.
    """
    per_target = metrics.get("per_target")
    if not isinstance(per_target, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for target_name, target_metrics in per_target.items():
        if not isinstance(target_metrics, Mapping):
            continue
        row = _aggregate_row(entry, target_metrics)
        if row is None:
            continue
        row["target_name"] = str(target_name)
        rows.append(row)
    return rows


def _s7_1_row(
    metadata: Mapping[str, Any],
    aggregate_row: dict[str, Any] | None,
    per_target_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Return one S7_1 benchmark row from per-target or scalar aggregate metrics.
    """
    for row in per_target_rows:
        if _is_s7_1_target(row.get("target_name")):
            return {
                key: value
                for key, value in row.items()
                if key != "target_name"
            }

    if aggregate_row is not None and _is_s7_1_benchmark(metadata):
        return aggregate_row
    return None


def _metric_value(
    metrics: Mapping[str, Any],
    split: str,
    metric_name: str,
) -> float | None:
    """
    Return a numeric metric value from a split metrics block.
    """
    split_metrics = metrics.get(split)
    if split == "validation" and split_metrics is None:
        split_metrics = metrics.get("val")
    if not isinstance(split_metrics, Mapping):
        return None

    value = split_metrics.get(metric_name)
    if value is None:
        value = split_metrics.get(metric_name.lower())
    return None if value is None else float(value)


def _is_vector_benchmark(metadata: Mapping[str, Any]) -> bool:
    """
    Return whether metadata describes a vector-target benchmark row.
    """
    data_interface = metadata.get("data_interface")
    if not isinstance(data_interface, Mapping):
        return False

    target_names = data_interface.get("target_names", ())
    target_scope = str(data_interface.get("target_scope", "")).lower()
    return target_scope == "vector" or len(target_names) > 1


def _is_s7_1_benchmark(metadata: Mapping[str, Any]) -> bool:
    """
    Return whether metadata describes an S7_1 scalar benchmark row.
    """
    data_interface = metadata.get("data_interface")
    if not isinstance(data_interface, Mapping):
        return False

    target_names = data_interface.get("target_names", ())
    target_scope = str(data_interface.get("target_scope", "")).lower()
    if target_scope == "vector" and len(target_names) > 1:
        return False
    return any(_is_s7_1_target(target_name) for target_name in target_names)


def _is_s7_1_target(target_name: Any) -> bool:
    """
    Return whether a target name refers to the S7_1 response in dB.
    """
    normalized = _normalise_target_name(target_name)
    return normalized in {
        "s7_1",
        "s7_1_db",
        "s7_1_magnitude_db",
        "il_s7_1_db",
        "s7_1_insertion_loss_db",
    }


def _normalise_target_name(target_name: Any) -> str:
    """
    Return a lowercase underscore-delimited target name.
    """
    return re.sub(r"[^a-z0-9]+", "_", str(target_name).lower()).strip("_")


def _upsert_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    key_columns: tuple[str, ...],
) -> None:
    """
    Insert or replace benchmark rows by key columns.
    """
    ensure_dir(path.parent)
    new_rows = pd.DataFrame(rows)
    table = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    columns = list(table.columns)
    columns.extend(column for column in new_rows.columns if column not in columns)

    for row in rows:
        if table.empty or not all(key in table.columns for key in key_columns):
            continue
        matches = pd.Series(True, index=table.index)
        for key in key_columns:
            matches &= table[key].astype(str) == str(row[key])
        table = table.loc[~matches]

    table = pd.concat([table, new_rows], ignore_index=True)
    table = table.reindex(columns=columns)
    _write_csv(table, path)


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    """
    Replace one CSV only after its temporary file is written completely.
    """
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        table.to_csv(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
