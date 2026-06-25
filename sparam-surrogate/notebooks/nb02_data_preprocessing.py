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
# # Lazy Data Preprocessing Pipeline
#
# This report builds the lightweight preprocessing artifact used by both the
# scalar baseline model and the full S-matrix model:
#
# ```text
# data/processed/sipi_dataset_cleaned.csv
# ```
#
# The CSV stores design features, frequency, split labels, simulation indices,
# and `TOUCHSTONE_REL_PATH`. It deliberately does not store S-parameter targets.
# During training, targets are loaded lazily from Touchstone files through
# `tf.data.Dataset.map(...)`.


# %% tags=["remove-input"]
# Reloads all modules every time before executing code.
# %load_ext autoreload
# %autoreload 2
# %aimport -pathlib
# %aimport -numpy

# %% [markdown]
# ## 1. Setup
#
# The notebook uses production code from `src/sparam_surrogate/data`. The
# selected topology is fixed here so the rendered report records exactly which
# raw dataset was processed.

# %%
from typing import Any

import numpy as np
import pandas as pd

from sparam_surrogate.config import SurrogateConfig, configure_stdio_relative_path
from sparam_surrogate.data import MLDatasetBuilder, RawData, TouchstoneLoader
from sparam_surrogate.utils.filesystem import ensure_dir

REBUILD_CLEANED_CSV = False

# Display paths relative to project root or user home for consistent output.
configure_stdio_relative_path()

# %%
cfg = SurrogateConfig.from_csv()
print(f"Name of raw dataset: {cfg.dataset.name}")

raw_data_dir = cfg.dataset.path
print(f"Path of raw dataset: {raw_data_dir}")

processed_dir = ensure_dir(cfg.paths.processed_data)
print(f"Processed directory: {processed_dir}")
print(f"Preprocessing artifact: {cfg.preprocessing.processed_csv}")

# %% [markdown]
# ## 2. Raw Data Consistency
#
# `RawData` reports mismatches between `parameter.csv` records and Touchstone
# files. Parameter rows without Touchstone files are dropped during cleaning,
# while orphan Touchstone files are ignored.

# %%
raw_data = RawData(raw_data_dir, nports=cfg.dataset.nports)
report = raw_data.check_index_consistency()

print(f"Parameter rows: {report['parameter_count']:,}")
print(f"Touchstone files: {report['touchstone_count']:,}")
print(f"Parameter rows without Touchstones: {len(report['missing_touchstones']):,}")
print(f"Touchstones without parameters: {len(report['missing_parameter_records']):,}")

# %% [markdown]
# ## 3. Build The Cleaned CSV
#
# `MLDatasetBuilder.data_cleaning()` loads `parameter.csv`, keeps only designs
# with matching Touchstone files, reads a representative frequency grid, expands
# each design over frequency, and writes one cleaned CSV. Feature values remain
# unscaled; train-only scaling belongs in the training pipeline.

# %%
builder = MLDatasetBuilder(raw_data, processed_dir)
cleaned_before_split = builder.data_cleaning(force=REBUILD_CLEANED_CSV)

print(f"Cleaned rows before split: {len(cleaned_before_split):,}")
print(f"Cleaned CSV: {builder.cleaned_path}")
print("Columns:")
print(list(cleaned_before_split.columns))

# %% [markdown]
# ## 4. Assign Train/Validation/Test Splits
#
# Splitting is performed by `SIMU_INDEX` before frequency expansion labels are
# applied to rows. This prevents the same physical design from appearing in more
# than one split.

# %%
train_set, val_set, test_set = builder.split(
    val_fraction=cfg.preprocessing.val_fraction,
    test_fraction=cfg.preprocessing.test_fraction,
    seed=cfg.project.seed,
    force=REBUILD_CLEANED_CSV,
)
cleaned = pd.read_csv(builder.cleaned_path)

row_counts = cleaned["SPLIT_TYPE"].value_counts().sort_index()
design_counts = cleaned.groupby("SPLIT_TYPE")["SIMU_INDEX"].nunique().sort_index()

print("Row counts by split:", row_counts, sep="\n")
print("\nDesign counts by split:", design_counts, sep="\n")

# %% [markdown]
# ## 5. Sanity Checks
#
# These checks are intentionally lightweight. They validate the CSV contract and
# split behavior without reloading any large eager target arrays.

# %%
expected_columns = list(MLDatasetBuilder.CLEANED_COLUMNS)
assert list(cleaned.columns) == expected_columns

for simulation_index, group in cleaned.groupby("SIMU_INDEX"):
    labels = set(group["SPLIT_TYPE"].astype(str))
    assert len(labels) == 1, f"SIMU_INDEX {simulation_index} appears in {labels}"

feature_values = cleaned.loc[:, MLDatasetBuilder.PARAMETER_COLUMNS]
assert np.isfinite(feature_values.to_numpy(dtype=float)).all()
assert set(cleaned["SPLIT_TYPE"]) == {"train", "val", "test"}

print("Sanity checks passed: schema, paths, finite features, and no split leakage.")

# %% [markdown]
# ## 6. Lazy Target Loading Smoke Test
#
# The scalar baseline and full S-matrix model now differ by the map callable
# used during training. This smoke test loads one row from the train split, then
# extracts scalar and full S-matrix targets from the row's Touchstone file.

# %%
sample_features = train_set.features[0]
sample_metadata: dict[str, Any] = {
    str(key): value for key, value in train_set.row_metadata.iloc[0].to_dict().items()
}

scalar_loader = TouchstoneLoader(
    mode="scalar",
    config=cfg,
    representation="db",
)
full_loader = TouchstoneLoader(
    mode="smatrix",
    config=cfg,
    representation="real_imag",
)

scalar_target = scalar_loader(sample_features, sample_metadata)
full_target = full_loader(sample_features, sample_metadata)

print("Sample metadata:")
print(sample_metadata)
print(f"Scalar target shape: {scalar_target.shape}")
print(f"Scalar target names: {scalar_loader.target_names}")
print(f"Full S-matrix target shape: {full_target.shape}")
print(f"First full target names: {full_loader.target_names[:8]}")

# %% [markdown]
# ## 7. Output Summary
#
# The preprocessing output is now a compact CSV index plus raw Touchstone files.
# The previous eager array artifacts are no longer part of the normal pipeline.

# %%
print(f"Final cleaned CSV: {builder.cleaned_path}")
print(f"Total cleaned rows: {len(cleaned):,}")
print(f"Unique designs: {cleaned['SIMU_INDEX'].nunique():,}")
print(f"Unique frequencies: {cleaned['FREQ_GHZ'].nunique():,}")
print("Feature scaling: deferred to training with train-only statistics")
