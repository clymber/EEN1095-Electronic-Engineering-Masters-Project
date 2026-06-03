# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (sparam-surrogate)
#     language: python
#     name: sparam-surrogate
# ---

# %% [markdown]
# # Data Preprocessing Pipeline
#
# This notebook turns one selected SI/PI-Database topology into two
# machine-learning-ready datasets:
#
# | Dataset | Input `X` | Target |
# | --- | --- | --- |
# | Scalar baseline | geometric/material parameters + frequency | one selected scalar S-parameter path |
# | Full S-matrix | geometric/material parameters + frequency | complete complex S-matrix at that frequency |
#
# The important design decision is that both datasets share the same input
# matrix:
#
# ```text
# X(n, k) = [u(n), f(k)]
# ```
#
# where `u(n)` is the PCB design parameter vector and `f(k)` is one Touchstone
# frequency point. The target changes between the scalar baseline and the full
# S-matrix dataset, but the rows, split labels, simulation indices, and
# frequency metadata must stay aligned.

# %% [markdown]
# ## 1. Setup
#
# The notebook uses production code from `src/sparam_surrogate/data`. It does
# not implement parsing, splitting, scaling, or target construction inside
# notebook cells. This keeps the workflow reproducible and testable.
#
# The constants below make the run explicit:
#
# - `DS_NAME` chooses the SI/PI-Database topology.
# - `REBUILD_RESPONSE_CACHE=False` reuses the compact parsed-response cache when
#   available.
# - `SCALAR_PAIR` chooses the first scalar baseline target. The default comes
#   from `configs/default.json`.

# %%
import json
from pathlib import Path

import numpy as np

from sparam_surrogate.config import (
    load_config,
    relative_to_project_root,
)
from sparam_surrogate.data import (
    DesignFrequencySplitter,
    MLDataset,
    MLDatasetBuilder,
    PcbFeatureTransformer,
    PcbParameters,
    RawData,
    SParameterDataset,
)

DS_NAME = "linkOn8CavityStackBetween10x10Array_19_08_2021"
REBUILD_RESPONSE_CACHE = False

# %%
cfg = load_config()

raw_data_dir = Path(cfg["paths"]["raw_data"]) / DS_NAME
interim_dir = Path(cfg["paths"]["interim_data"])
processed_dir = Path(cfg["paths"]["processed_data"])
processed_dir.mkdir(parents=True, exist_ok=True)

nports = int(cfg["dataset"]["nports"])
port_pairs = [tuple(pair) for pair in cfg["dataset"]["ports"]]
scalar_pair = port_pairs[0]

response_cache_path = interim_dir / f"{DS_NAME}_sparameters_full.npz"

print(f"Dataset: {DS_NAME}")
print(f"Raw data directory: {relative_to_project_root(raw_data_dir)}")
print(f"Response cache: {relative_to_project_root(response_cache_path)}")
print(f"Processed output directory: {relative_to_project_root(processed_dir)}")
print(f"Scalar baseline target: S{scalar_pair[0]}_{scalar_pair[1]}_DB")

# %% [markdown]
# ## 2. Load Raw Parameters And Align Touchstones
#
# `RawData` knows the raw folder structure. `PcbParameters` loads
# `parameter.csv`. `SParameterDataset` performs the expensive Touchstone
# parsing once, aligns files to `SIMU_INDEX`, validates the common frequency
# grid, and caches:
#
# - selected dB paths such as `S7_1_DB`
# - the complete complex S-matrix for each design and frequency
#
# The cache is intentionally stored in `data/interim`, because it is an
# intermediate representation derived from raw data. Reusing it avoids reparsing
# thousands of Touchstone files on every notebook run.

# %%
raw_data = RawData(raw_data_dir, nports=nports)
parameters = PcbParameters(raw_data.parameter_csv)

responses = SParameterDataset.from_touchstones(
    parameters,
    raw_data,
    port_pairs,
    cache_path=response_cache_path,
    rebuild_cache=REBUILD_RESPONSE_CACHE,
)

print(f"Parameter rows: {len(parameters.dataframe):,}")
print(f"Aligned response designs: {len(responses.simulation_indices):,}")
print(f"Frequency points: {len(responses.frequencies_ghz):,}")
print(f"Selected scalar paths: {responses.port_pairs}")
print(f"Full S-matrix shape: {responses.full_s_matrix.shape}")

# if responses.alignment_report is not None:
#     print("Alignment report:")
#     print(json.dumps(responses.alignment_report, indent=2))

# %% [markdown]
# ## 3. Define Split And Feature Rules
#
# Splitting must happen by design, not by design-frequency row. Otherwise the
# same physical PCB design could appear in both train and test sets at different
# frequencies, which would leak information and overstate model performance.
#
# The pipeline therefore:
#
# 1. splits `SIMU_INDEX` into train, validation, and test sets;
# 2. expands each design across the full frequency grid;
# 3. repeats the design-level split label for every frequency row;
# 4. fits feature scaling statistics using train rows only.

# %%
splitter = DesignFrequencySplitter(
    test_size=float(cfg["training"]["test_size"]),
    val_size=float(cfg["training"]["val_size"]),
    random_state=int(cfg["project"]["seed"]),
)

feature_transformer = PcbFeatureTransformer()

builder = MLDatasetBuilder(
    splitter=splitter,
    feature_transformer=feature_transformer,
    output_dir=processed_dir,
    metadata={
        "dataset_name": DS_NAME,
        "response_cache": str(response_cache_path),
        "random_seed": int(cfg["project"]["seed"]),
    },
)

split = splitter.split(responses.simulation_indices)
print(f"Train designs: {len(split.train_indices):,}")
print(f"Validation designs: {len(split.val_indices):,}")
print(f"Test designs: {len(split.test_indices):,}")

# %% [markdown]
# ## 4. Build The Scalar Baseline Dataset
#
# The scalar baseline is the smallest useful supervised-learning target. It
# checks that parsing, split handling, scaling, row ordering, and metrics work
# before training a high-dimensional S-matrix model.
#
# The generated dataset is saved as:
#
# ```text
# data/processed/scalar_baseline_dataset.npz
# ```

# %%
scalar_dataset = builder.build_scalar_dataset(
    parameters,
    responses,
    pair=scalar_pair,
    representation="db",
)

print("Scalar baseline dataset")
print(f"  X shape: {scalar_dataset.X.shape}")
print(f"  target shape: {scalar_dataset.target.shape}")
print(f"  feature names: {scalar_dataset.feature_names}")
print(f"  target names: {scalar_dataset.target_names}")
print(f"  split labels: {dict(zip(*np.unique(scalar_dataset.split_labels, return_counts=True)))}")

# %% [markdown]
# ## 5. Build The Full S-Matrix Dataset
#
# The full S-matrix dataset keeps the same `X` rows and replaces the scalar
# target with the complete complex S-matrix at each frequency.
#
# `TargetBuilder` flattens each parsed matrix in row-major order and keeps real
# and imaginary parts adjacent:
#
# ```text
# REAL_S1_1, IMAG_S1_1, REAL_S1_2, IMAG_S1_2, ...
# ```
#
# The generated dataset is saved as:
#
# ```text
# data/processed/full_smatrix_dataset.npz
# ```

# %%
full_smatrix_dataset = builder.build_full_smatrix_dataset(parameters, responses)

print("Full S-matrix dataset")
print(f"  X shape: {full_smatrix_dataset.X.shape}")
print(f"  target shape: {full_smatrix_dataset.target.shape}")
print(f"  first target names: {full_smatrix_dataset.target_names[:8]}")
print(f"  split labels: {dict(zip(*np.unique(full_smatrix_dataset.split_labels, return_counts=True)))}")

# %% [markdown]
# ## 6. Sanity Checks
#
# Both datasets must share exactly the same inputs and row metadata. This is the
# central guarantee that makes scalar baseline results comparable to the later
# full-S-matrix model.

# %%
np.testing.assert_allclose(scalar_dataset.X, full_smatrix_dataset.X)
np.testing.assert_array_equal(
    scalar_dataset.split_labels,
    full_smatrix_dataset.split_labels,
)
np.testing.assert_array_equal(
    scalar_dataset.simulation_indices,
    full_smatrix_dataset.simulation_indices,
)
np.testing.assert_allclose(
    scalar_dataset.frequencies_ghz,
    full_smatrix_dataset.frequencies_ghz,
)

for simulation_index in np.unique(scalar_dataset.simulation_indices):
    labels = set(
        scalar_dataset.split_labels[
            scalar_dataset.simulation_indices == simulation_index
        ].tolist()
    )
    assert len(labels) == 1, f"SIMU_INDEX {simulation_index} appears in {labels}"

print("Sanity checks passed: shared X, shared metadata, and no split leakage.")

# %% [markdown]
# ## 7. Save Human-Readable Metadata
#
# Each `.npz` file already contains its own arrays and JSON metadata. The files
# below are lightweight sidecars for quick inspection and reporting:
#
# - `splits.json`: design-level train/validation/test `SIMU_INDEX` membership
# - `preprocessing_metadata.json`: feature names, target names, shapes, scaling
#   metadata, and output paths

# %%
splits_json = {
    "train": split.train_indices.astype(int).tolist(),
    "val": split.val_indices.astype(int).tolist(),
    "test": split.test_indices.astype(int).tolist(),
}

preprocessing_metadata = {
    "dataset_name": DS_NAME,
    "scalar_dataset": str(processed_dir / MLDatasetBuilder.SCALAR_FILENAME),
    "full_smatrix_dataset": str(processed_dir / MLDatasetBuilder.FULL_SMATRIX_FILENAME),
    "response_cache": str(response_cache_path),
    "feature_names": list(scalar_dataset.feature_names),
    "scalar_target_names": list(scalar_dataset.target_names),
    "full_smatrix_target_names": list(full_smatrix_dataset.target_names),
    "scalar_shape": {
        "X": list(scalar_dataset.X.shape),
        "target": list(scalar_dataset.target.shape),
    },
    "full_smatrix_shape": {
        "X": list(full_smatrix_dataset.X.shape),
        "target": list(full_smatrix_dataset.target.shape),
    },
    "frequency_unit": "GHz",
    "frequencies_ghz": responses.frequencies_ghz.tolist(),
    "split_counts_designs": {
        "train": len(split.train_indices),
        "val": len(split.val_indices),
        "test": len(split.test_indices),
    },
    "feature_scaling": scalar_dataset.metadata.get("feature_scaling"),
    "feature_mean": scalar_dataset.metadata.get("feature_mean"),
    "feature_scale": scalar_dataset.metadata.get("feature_scale"),
}

(processed_dir / "splits.json").write_text(
    json.dumps(splits_json, indent=2),
    encoding="utf-8",
)
(processed_dir / "preprocessing_metadata.json").write_text(
    json.dumps(preprocessing_metadata, indent=2),
    encoding="utf-8",
)

print(f"Wrote {relative_to_project_root(processed_dir/'splits.json')}")
print(f"Wrote {relative_to_project_root(processed_dir/'preprocessing_metadata.json')}")

# %% [markdown]
# ## 8. Reload Check
#
# Finally, reload both saved `.npz` datasets through `MLDataset.load`. This
# checks that the saved artifacts are usable by future training notebooks and
# scripts without carrying notebook state.

# %%
loaded_scalar = MLDataset.load(processed_dir / MLDatasetBuilder.SCALAR_FILENAME)
loaded_full = MLDataset.load(processed_dir / MLDatasetBuilder.FULL_SMATRIX_FILENAME)

np.testing.assert_allclose(loaded_scalar.X, scalar_dataset.X)
np.testing.assert_allclose(loaded_scalar.target, scalar_dataset.target)
np.testing.assert_allclose(loaded_full.X, full_smatrix_dataset.X)
np.testing.assert_allclose(loaded_full.target, full_smatrix_dataset.target)

print("Reload checks passed.")
