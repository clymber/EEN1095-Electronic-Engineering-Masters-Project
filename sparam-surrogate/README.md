# sparam-surrogate

Machine learning surrogate models for predicting PCB interconnect S-parameters and signal integrity metrics.

## Class Architecture

The project separates source-data representation, exploratory analysis, and
ML-ready preprocessing. The central rule is that preprocessing should create
one shared design-frequency input and then attach different targets for
different modelling stages.

### Class Relationship

```text
RawData
  |
  +--> PcbParameters
  |
  +--> SParameterDataset
           ^
           |
      uses PcbParameters for aligned SIMU_INDEX records

PcbParameters + SParameterDataset
  |
  +--> PcbDatasetEDA
  |      exploration, plots, sanity checks
  |
  +--> MLDatasetBuilder
         |
         +--> DesignFrequencySplitter
         |      split by SIMU_INDEX before frequency expansion
         |
         +--> PcbFeatureTransformer
         |      build and scale X = [design parameters, frequency]
         |
         +--> TargetBuilder
         |      build scalar targets or full S-matrix targets
         |
         +--> MLDataset
                final training-ready arrays and metadata
```

`PcbDatasetEDA` is a side branch. It consumes the same data sources, but it
does not write the final model-training arrays.

### Existing Source And EDA Classes

| Class | Status | Responsibility |
| ----- | ------ | -------------- |
| `RawData` | Implemented | Locates the unzipped SI/PI dataset files, exposes `parameter.csv` and Touchstone paths, and reports index mismatches between parameter rows and response files. |
| `PcbParameters` | Implemented | Loads and validates the PCB geometric/material parameter table for the selected topology. |
| `SParameterDataset` | Implemented, to be extended | Aligns parameter records with Touchstone files and caches frequency-dependent S-parameter responses. The current implementation extracts selected through paths in dB; the full pipeline should extend this layer to preserve the complete complex S-matrix. It should not own train/validation/test splitting or feature scaling. |
| `PcbDatasetEDA` | Implemented | Provides exploratory summaries and plots for parameters and aligned response data. This class is for analysis, not for writing final training arrays. |

### Planned Preprocessing Classes

| Class | Responsibility |
| ----- | -------------- |
| `DesignFrequencySplitter` | Creates reproducible train/validation/test splits by `SIMU_INDEX`, before frequency expansion, so the same physical design cannot leak across splits. |
| `PcbFeatureTransformer` | Builds, expands, and scales the shared input matrix `X = [geometric/material parameters, frequency]` using train-split statistics only. Derived features can live here once they are needed for modelling rather than only EDA. |
| `TargetBuilder` | Builds the target array from `SParameterDataset`. It should support at least a scalar baseline target and a full S-matrix target, but both modes reuse the same `X`. |
| `MLDatasetBuilder` | Orchestrates splitting, feature transformation, target construction, metadata assembly, and saving processed datasets. |
| `MLDataset` | Stores model-ready `X`, target arrays, split labels, feature names, target names, frequency metadata, and simulation-index metadata. |

The two training datasets should therefore share the same input definition:

```text
X(n, k) = [u(n), f(k)]
```

where `u(n)` is the geometric/material parameter vector for design sample `n`
and `f(k)` is one Touchstone frequency point. The datasets differ in the target:

| Dataset | Input | Target |
| ------- | ----- | ------ |
| Scalar baseline dataset | `X = design + frequency` | One scalar S-parameter or insertion-loss value. |
| Full S-matrix dataset | `X = design + frequency` | Complete complex S-matrix at that frequency. |

### Notebook Roles

| Notebook | Purpose |
| -------- | ------- |
| `notebooks/dataset_exploration.ipynb` | Explore and sanity-check the selected topology, parameters, Touchstone structure, response distributions, and correlations. |
| `notebooks/data_preprocessing.ipynb` | Run the reproducible preprocessing pipeline that writes aligned caches, split metadata, and ML-ready scalar/full-S-matrix datasets. |

## Setup

```bash
conda env create -f environment.yml
conda activate meng
pip install -e .
```

## PDF reports

Build the executed notebook reports with:

```bash
make webpdf
```

The first WebPDF build may download Playwright's Chromium runtime into its
user cache for `nbconvert`; later builds reuse it. In an offline environment,
install it in advance with `playwright install chromium`.
