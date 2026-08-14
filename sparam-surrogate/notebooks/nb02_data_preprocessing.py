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
# This report builds a shared design-level artifact, then derives the
# frequency-expanded preprocessing artifact used by point-wise models:
#
# ```text
# data/processed/cleaned_splits_parameter.csv
#         ↓
# data/processed/frequency_expanded_dataset.csv
# ```
#
# Both CSVs omit S-parameter targets. The cleaned split CSV is the authoritative
# one-row-per-design source for all later preprocessing, including whole-curve
# models. During training, targets are loaded lazily from Touchstone files.


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
from sparam_surrogate.data import (
    ParameterDatasetBuilder,
    PointwiseDataset,
    RawData,
    TouchstoneLoader,
)
from sparam_surrogate.utils.filesystem import ensure_dir

FORCE_REBUILD = False

# Display paths relative to project root or user home for consistent output.
configure_stdio_relative_path()

# %%
cfg = SurrogateConfig.from_config()
print(f"Name of raw dataset: {cfg.dataset.name}")

raw_data_dir = cfg.dataset.path
print(f"Path of raw dataset: {raw_data_dir}")

processed_dir = ensure_dir(cfg.paths.processed_data)
cleaned_splits_path = cfg.preprocessing.cleaned_splits_csv
frequency_expanded_path = cfg.preprocessing.freq_expanded_csv
print(f"Processed directory: {processed_dir}")
print(f"Cleaned split parameters: {cleaned_splits_path}")
print(f"Frequency-expanded dataset: {frequency_expanded_path}")

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
# ## 3. Data cleaning
#
# `ParameterDatasetBuilder.clean()` runs only when the cleaned CSV is missing or
# `FORCE_REBUILD` is `True`. It keeps only designs with matching Touchstone
# files and records one portable Touchstone path per design.

# %%
parameter_builder = ParameterDatasetBuilder(raw_data, cleaned_splits_path)
rebuild_cleaned_csv = FORCE_REBUILD or not cleaned_splits_path.exists()

if rebuild_cleaned_csv:
    cleaned_parameters = parameter_builder.clean()
    print(f"Valid designs after cleaning: {len(cleaned_parameters):,}")
    print("Columns:",list(cleaned_parameters.columns))
else:
    print(f"Reusing cleaned parameters: {cleaned_splits_path}")

# %% [markdown]
# ## 4. Dataset Splitting
#
# Splitting runs only after cleaning. Otherwise, the existing cleaned CSV is
# reused. This file is the authoritative source of split labels for all
# downstream point-wise and whole-curve datasets:
#
# ```text
# data/processed/cleaned_splits_parameter.csv
# ```

# %%
if rebuild_cleaned_csv:
    split_parameters = parameter_builder.split(
        cleaned_parameters,
        val_fraction=cfg.preprocessing.val_fraction,
        test_fraction=cfg.preprocessing.test_fraction,
        seed=cfg.project.seed,
    )
else:
    split_parameters = parameter_builder.load()

design_counts = split_parameters["SPLIT_TYPE"].value_counts().sort_index()

print("Design counts by split:", design_counts, sep="\n")
print(f"\nCleaned split parameters: {cleaned_splits_path}")

# %% [markdown]
# ## 5. Build the Frequency-Expanded Dataset
#
# `PointwiseDataset.build_frequency_expanded_csv()` reads only the cleaned and
# split parameter CSV. Each design row is expanded over the common Touchstone
# frequency grid. Expansion runs only when the processed CSV is missing, older
# than the cleaned split CSV, or explicitly forced.

# %%
pointwise_param = PointwiseDataset.build_frequency_expanded_csv(
    cleaned_splits_path,
    frequency_expanded_path,
    force=FORCE_REBUILD,
)
train_set = PointwiseDataset(pointwise_param, split_type="train")

row_counts = pointwise_param["SPLIT_TYPE"].value_counts().sort_index()
print("Frequency-expanded row counts by split:", row_counts, sep="\n")
print(f"\nFrequency-expanded CSV: {frequency_expanded_path}")

# %% [markdown]
# ## 6. Sanity Checks
#
# These checks validate both CSV contracts and confirm that design-level split
# labels remain unchanged after frequency expansion.

# %%
expected_parameter_columns = list(ParameterDatasetBuilder.SPLIT_COLUMNS)
assert list(split_parameters.columns) == expected_parameter_columns

expected_columns = list(PointwiseDataset.COLUMNS)
assert list(pointwise_param.columns) == expected_columns

for simulation_index, group in pointwise_param.groupby("SIMU_INDEX"):
    labels = set(group["SPLIT_TYPE"].astype(str))
    assert len(labels) == 1, f"SIMU_INDEX {simulation_index} appears in {labels}"

feature_values = pointwise_param.loc[:, PointwiseDataset.PARAMETER_COLUMNS]
assert np.isfinite(feature_values.to_numpy(dtype=float)).all()
assert set(pointwise_param["SPLIT_TYPE"]) == {"train", "val", "test"}

split_by_design = split_parameters.set_index("SIMU_INDEX")["SPLIT_TYPE"]
expanded_split_by_design = pointwise_param.groupby("SIMU_INDEX")["SPLIT_TYPE"].first()
pd.testing.assert_series_equal(
    expanded_split_by_design.sort_index(),
    split_by_design.sort_index(),
    check_names=False,
)

print("Sanity checks passed for design-level and frequency-expanded CSVs.")

# %% [markdown]
# ## 7. Lazy Target Loading Smoke Test
#
# The point-wise insertion-loss baseline and full S-matrix model differ by the
# map callable used during training. This smoke test loads one row from the train
# split, then extracts positive scalar IL and full S-matrix targets from the
# row's Touchstone file. The scalar convention is
# $IL_{ij,\mathrm{dB}}=-20\log_{10}|S_{ij}|$.

# %%
sample_features = train_set.features[0]
sample_metadata: dict[str, Any] = {
    str(key): value for key, value in train_set.row_metadata.iloc[0].to_dict().items()
}

scalar_il_loader = TouchstoneLoader(
    mode="scalar",
    config=cfg,
    representation="il",
)
full_loader = TouchstoneLoader(
    mode="smatrix",
    config=cfg,
    representation="real_imag",
)

scalar_il_target = scalar_il_loader(sample_features, sample_metadata)
full_target = full_loader(sample_features, sample_metadata)
assert np.all(scalar_il_target > 0.0)

print("Sample metadata:")
print(sample_metadata)
print(f"Scalar IL target shape: {scalar_il_target.shape}")
print(f"Scalar IL target names: {scalar_il_loader.target_names}")
print(f"Full S-matrix target shape: {full_target.shape}")
print(f"First full target names: {full_loader.target_names[:8]}")

# %% [markdown]
# ## 8. Output Summary
#
# The preprocessing outputs are a shared design-level split table and a
# point-wise frequency-expanded table. The previous eager array artifacts are
# not part of the normal pipeline.

# %%
print(f"Design-level split CSV: {cleaned_splits_path}")
print(f"Frequency-expanded CSV: {frequency_expanded_path}")
print(f"Total valid designs: {len(split_parameters):,}")
print(f"Total frequency-expanded rows: {len(pointwise_param):,}")
print(f"Unique designs: {pointwise_param['SIMU_INDEX'].nunique():,}")
print(f"Unique frequencies: {pointwise_param['FREQ_GHZ'].nunique():,}")
print("Feature scaling: deferred to training with train-only statistics")
