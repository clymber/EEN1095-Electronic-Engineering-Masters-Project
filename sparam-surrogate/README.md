# sparam-surrogate

Machine learning surrogate models for predicting PCB interconnect
S-parameters and signal-integrity metrics.

## Current Preprocessing Design

The project now uses a lazy preprocessing pipeline. Preprocessing writes one
compact CSV index:

```text
data/processed/sipi_dataset_cleaned.csv
```

The CSV is shared by scalar baseline models and full S-matrix models. It stores
features and metadata only:

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
  +--> MLDatasetBuilder
         |
         +--> sipi_dataset_cleaned.csv
         |
         +--> DLDataset(train)
         +--> DLDataset(val)
         +--> DLDataset(test)
                |
                +--> tf.data.Dataset.map(TouchstoneLoader(...))
```

| Component | Responsibility |
| --------- | -------------- |
| `RawData` | Locates one unzipped raw dataset, including `parameter.csv` and `variation/simu_<index>.sNp` files. |
| `PcbParameters` | Loads and validates PCB design/material parameters. |
| `PcbDatasetEDA` | Provides parameter and optional response exploration for reports. |
| `MLDatasetBuilder` | Builds `sipi_dataset_cleaned.csv` and assigns split labels by `SIMU_INDEX`. |
| `DLDataset` | Represents one split from the cleaned CSV and builds `tf.data.Dataset` objects. |
| `TouchstoneLoader` | Lazily loads scalar or full S-matrix targets from `TOUCHSTONE_REL_PATH` using `dataset.nports` from configuration. |

## Basic Usage

```python
from sparam_surrogate.config import SurrogateConfig
from sparam_surrogate.data import MLDatasetBuilder, RawData, TouchstoneLoader

cfg = SurrogateConfig.from_config()

raw_data = RawData(
    cfg.dataset.path,
    nports=cfg.dataset.nports,
)
builder = MLDatasetBuilder(raw_data, cfg.paths.processed_data)

builder.data_cleaning()
train_set, val_set, test_set = builder.split(
    val_fraction=cfg.preprocessing.val_fraction,
    test_fraction=cfg.preprocessing.test_fraction,
    seed=cfg.project.seed,
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

Feature scaling is intentionally not written into the cleaned CSV. Training
code should fit scaling statistics on train rows only, then apply those
statistics to validation and test rows.

## TensorFlow Dataset Mapping

```python
train_ds = train_set.to_tf_dataset(
    map_func=scalar_loader,
    batch_size=cfg.models.neural_mlp.batch_size,
    shuffle=True,
)
```

For full S-matrix training, use `full_loader` as the map function. Full matrix
targets are flattened as all real components followed by all imaginary
components in row-major S-matrix order.

## Command Line

```bash
sparam-surrogate preprocess \
  --input-dir data/raw/linkOn8CavityStackBetween10x10Array_19_08_2021 \
  --output-dir data/processed \
  --nports 12
```

When split options are omitted, the command uses `configs/default.json`.

## Setup

```bash
conda env create -f environment.yml
conda activate meng
pip install -e .
```

Install the optional ML dependencies before using TensorFlow dataset mapping:

```bash
pip install -e ".[ml]"
```

## Reports

Build the executed notebook reports with:

```bash
make webpdf
```

`notebooks/data_preprocessing.py` is the reproducible preprocessing report. It
builds the cleaned CSV, checks split leakage, and performs a small lazy-loading
smoke test for scalar and full S-matrix targets.

`notebooks/dataset_exploration.py` remains the broader exploratory report for
raw parameters, geometry relationships, and small Touchstone inspections.
