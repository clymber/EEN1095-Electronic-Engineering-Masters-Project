# sparam-surrogate

Machine learning surrogate models for predicting PCB interconnect
S-parameters and signal-integrity metrics.

## Current Preprocessing Design

The project uses a two-stage lazy preprocessing pipeline:

```text
data/processed/cleaned_splits_parameter.csv
        ↓
data/processed/frequency_expanded_dataset.csv
```

The cleaned split CSV contains one row per design and owns the fixed
train/validation/test assignment. It is the shared source for point-wise and
whole-curve preprocessing:

```csv
EPS,TAND,PITCH,TRACE_LEN,START,VIAR,ANTIPADR,TDIEL,DISTTL,TLWIDTH,SIMU_INDEX,TOUCHSTONE_REL_PATH,SPLIT_TYPE
```

The processed CSV expands those designs over frequency for point-wise models:

```csv
EPS,TAND,PITCH,TRACE_LEN,START,VIAR,ANTIPADR,TDIEL,DISTTL,TLWIDTH,FREQ_GHZ,SIMU_INDEX,TOUCHSTONE_REL_PATH,SPLIT_TYPE
```

S-parameter targets are not precomputed into large arrays. During training, a
lazy map callable loads the required Touchstone file for each row.

## Class Architecture

```text
RawData
  |
  +--> PcbParameters
  |
  +--> PcbDatasetEDA
  |
  +--> ParameterDatasetBuilder
         |
         +--> cleaned_splits_parameter.csv
                 |
                 +--> PointwiseDataset.build_frequency_expanded_csv(...)
                 |      |
                 |      +--> frequency_expanded_dataset.csv
                 |
                 +--> TouchstoneLoader.load_curve(...)
                        |
                        +--> nb05 whole-curve arrays
```

The frequency-expanded CSV continues through:

```text
frequency_expanded_dataset.csv
         +--> PointwiseDataset(train)
         +--> PointwiseDataset(val)
         +--> PointwiseDataset(test)
                |
                +--> features and cached TouchstoneLoader targets
```

| Component | Responsibility |
| --------- | -------------- |
| `RawData` | Locates one unzipped raw dataset, including `parameter.csv` and `variation/simu_<index>.sNp` files. |
| `PcbParameters` | Loads and validates PCB design/material parameters. |
| `PcbDatasetEDA` | Provides parameter and optional response exploration for reports. |
| `ParameterDatasetBuilder` | Cleans raw parameter rows, aligns Touchstones, assigns design-level splits, and writes the cleaned CSV. |
| `PointwiseDataset` | Represents one split from the frequency-expanded CSV and exposes NumPy features and targets. |
| `TouchstoneLoader` | Lazily loads S-parameter targets from `TOUCHSTONE_REL_PATH`; nb05 extends it with whole-curve loading. |

## Basic Usage

```python
from sparam_surrogate.config import SurrogateConfig
from sparam_surrogate.data import (
    PointwiseDataset,
    ParameterDatasetBuilder,
    RawData,
    TouchstoneLoader,
)

cfg = SurrogateConfig.from_config()

raw_data = RawData(
    cfg.dataset.path,
    nports=cfg.dataset.nports,
)
parameter_builder = ParameterDatasetBuilder(
    raw_data,
    cfg.preprocessing.cleaned_splits_csv,
)
parameter_builder.build(
    val_fraction=cfg.preprocessing.val_fraction,
    test_fraction=cfg.preprocessing.test_fraction,
    seed=cfg.project.seed,
    force=False,
)

PointwiseDataset.build_frequency_expanded_csv(
    cfg.preprocessing.cleaned_splits_csv,
    cfg.preprocessing.freq_expanded_csv,
)
train_set, val_set, test_set = PointwiseDataset.from_frequency_expanded_csv(
    cfg.preprocessing.freq_expanded_csv
)

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
```

`TouchstoneLoader` requires `cfg.dataset.nports`. Scalar and vector modes
also require `cfg.dataset.ports` so selected one-based port pairs stay in
the configuration file.

Feature scaling is intentionally not written into the frequency-expanded CSV. Training
code should fit scaling statistics on train rows only, then apply those
statistics to validation and test rows.

## Command Line

```bash
sparam-surrogate preprocess \
  --input-dir data/raw/linkOn8CavityStackBetween10x10Array_19_08_2021 \
  --output-dir data/processed \
  --nports 12
```

When split options are omitted, the command uses `configs/default.json`.
The command writes both preprocessing CSVs to the requested output directory.

An existing cleaned split CSV skips raw cleaning and split assignment. The
frequency-expanded CSV is rebuilt only when it is missing or older than the cleaned
split CSV. Pass `--force` to rebuild both artefacts.

## Setup

```bash
conda env create -f environment.yml
conda activate meng
pip install -e .
```

Install the optional ML dependencies before running the neural models:

```bash
pip install -e ".[ml]"
```

## Reports

Build the executed notebook reports with:

```bash
make webpdf
```

`notebooks/nb02_data_preprocessing.py` is the reproducible preprocessing report.
It builds both CSVs, checks split leakage, and performs a small lazy-loading
smoke test for scalar and full S-matrix targets.

`notebooks/nb01_dataset_exploration.py` remains the broader exploratory report for
raw parameters, geometry relationships, and small Touchstone inspections.
