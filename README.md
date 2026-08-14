# Using Machine Learning to Predict Scattering Parameters for Signal Integrity Purposes

**EEN1095 project repository — Chunyu Long (A00049113), August 2026**

This repository accompanies the project portfolio and contains the implementation,
executed notebooks, tests, selected-run records, and machine-readable evidence used to
support the reported results. The Loop submission is named
`A00049113_project_repository.zip`.

## Project overview

The project asks whether a machine-learning surrogate can predict the complete
broadband scattering matrix of one selected multi-port printed circuit board (PCB)
topology with accuracy suitable for signal-integrity (SI) analysis, while enforcing
passivity and causality and reducing repeated electromagnetic-simulation cost.

The documented investigation follows this progression:

```text
data pipeline
  -> point-wise models (scalar -> multi-output -> nonlinear)
  -> whole-curve neural model
  -> reciprocal complete 12-port S-matrix neural model
  -> physical diagnostics
```

This is an exploration of progressively broader formulations, not a claim that model
accuracy improved at every stage.

The selected Signal Integrity/Power Integrity (SI/PI) Database topology is
*Link on 8 Cavity PCB with two 10×10 Via-Arrays*. After documented exclusions, the
dataset contains 7,030 aligned designs, ten geometric or material inputs, and a complex
12-port response at 200 frequencies
from 0.5 to 100 GHz. Designs are split 4,218/1,406/1,406 into
training/validation/test sets using seed 128; fitted transforms use training data only.

### Headline findings and boundaries

Mean absolute error (MAE) measures average error magnitude. Normalized
root-mean-square error (NRMSE) also emphasizes larger errors and scales them against
the simulated response. `IL(7,1)` is the insertion loss (IL), in decibels, derived from
the response from port 1 to port 7. The complete-matrix MAE and NRMSE are dimensionless.

| Evaluation | Final evidence | Interpretation |
| --- | --- | --- |
| Complete complex S-matrix | Test Complex MAE/NRMSE improved from 0.0804/0.8343 for the deterministic training-mean matrix to 0.0686/0.7889 for the selected neural model. | The model learned design-dependent information beyond the mean response. |
| Selected insertion-loss path | The whole-curve model achieved 7.3883 dB test MAE on `IL(7,1)`, the lowest value among the retained model formulations; the full-matrix model achieved 10.7903 dB on the same extracted path. | Selected-path accuracy and complete-matrix coverage are separate evaluation axes. |
| Physical diagnostics | Reciprocity residual was zero by construction. No passivity violation occurred in 281,200 sampled test design-frequency matrices; the maximum predicted singular value was 0.9141. The finite-band causality residual was 0.4812 for predictions and 0.3173 for simulated responses. | Reciprocity is guaranteed. Passivity was observed only on the sampled grid, and causality was only diagnosed; neither passivity nor causality was enforced. |

The approved question was therefore only partly answered. No downstream engineering
tolerance established accuracy sufficient for an SI decision. The test designs were
held out from fitting but their results were inspected during development, so the final
comparison is retrospective rather than untouched confirmation. Findings are limited
to one topology, sampled parameter ranges, and one selected training seed. The planned
controlled physical-parameter study and electromagnetic-simulation timing baseline
were not completed, so neither physical sensitivity nor reduced simulation cost was
demonstrated.

## Quick review without retraining

The retained evidence can be inspected without the raw dataset or model training:

1. Open the [compiled project portfolio](project_portfolio/build/A00049113_EEN1095_Full_Project_Portfolio.pdf).
   The six-page paper gives the concise argument; Appendices C–F contain planning,
   design, testing, and repository detail.
2. Read the rendered
   [NB07 selected-model evaluation](sparam-surrogate/reports/pdf/nb07_selected_models_evaluation_analysis.pdf)
   or its [executed notebook](sparam-surrogate/notebooks/nb07_selected_models_evaluation_analysis.ipynb).
3. Inspect the [evidence index](project_portfolio/evidence/README.md) and its CSV files
   for the exact table values, bootstrap comparisons, physical diagnostics, selected
   run provenance, and the exported saved-metric reproduction checks.
4. Open the [executed NB08 notebook](sparam-surrogate/notebooks/nb08_appendix_d_model_graphs.ipynb)
   and [exported model diagrams](project_portfolio/media/appendix_d/) for the retained
   model structures.
5. Trace each selected model from
   [`selected.json`](sparam-surrogate/outputs/models/selected.json) to its resolved
   configuration, metadata, metrics, manifest, environment record, and saved-run
   identifier.

The `.py` notebook files are readable Jupytext sources. Their same-stem `.ipynb` files
retain the executed outputs. NB01–NB08 are repository identifiers for those numbered
notebooks, not technical acronyms.

## Nine core items for detailed review

These items are the recommended assessment route and do not restrict review of other
files in the submitted repository.

| No. | Core item | Purpose |
| ---: | --- | --- |
| 1 | [`README.md`](README.md) | Main repository guide, evidence index, setup instructions, and scope statement. |
| 2 | [`project_portfolio/`](project_portfolio/) | Six-page paper and Appendices A–F, with references, figures, evidence exports, and build instructions. |
| 3 | [`sparam-surrogate/src/sparam_surrogate/`](sparam-surrogate/src/sparam_surrogate/) | Core Python package for data alignment and loading, surrogate models, reciprocal-matrix reconstruction, physical diagnostics, run records, and result utilities. |
| 4 | [`environment.yml`](sparam-surrogate/environment.yml), [`pyproject.toml`](sparam-surrogate/pyproject.toml), [`default.json`](sparam-surrogate/configs/default.json), [`env_setup.sh`](sparam-surrogate/env_setup.sh), and [`Makefile`](sparam-surrogate/Makefile) | Software environment and the dependency, dataset, split, model, and notebook-build configuration. |
| 5 | [`sparam-surrogate/notebooks/`](sparam-surrogate/notebooks/) | Paired Jupytext sources and executed notebooks NB01–NB08, covering the complete investigation. |
| 6 | [`selected.json`](sparam-surrogate/outputs/models/selected.json), [`selected_run_provenance.csv`](project_portfolio/evidence/selected_run_provenance.csv), and the referenced selected-run records under [`outputs/runs/`](sparam-surrogate/outputs/runs/) | Frozen model selection and provenance. The compact records include configurations, environments, manifests, metadata, metrics, training histories or validation-selection records where applicable, and key figures. |
| 7 | [`project_portfolio/evidence/`](project_portfolio/evidence/) | Machine-readable final results, provenance, bootstrap intervals, physical diagnostics, metric checks, and the Appendix D model inventory. |
| 8 | [`sparam-surrogate/tests/`](sparam-surrogate/tests/) | Unit and integration tests for data handling, model contracts, reciprocal reconstruction, run records, and evaluation utilities. |
| 9 | [`sparam-surrogate/reports/pdf/`](sparam-surrogate/reports/pdf/) | Core read-only PDF reports for NB01–NB07. A local NB08 PDF is supplementary; its core forms are the executed notebook and Appendix D diagram exports. |

## Notebook map

Every notebook has an executed `.ipynb` file and a same-stem Jupytext `.py` source.
The notebooks form an evidence trail; they are not a monotonic model leaderboard.

| Label | Executed notebook | Purpose |
| --- | --- | --- |
| NB01 | [`nb01_dataset_exploration.ipynb`](sparam-surrogate/notebooks/nb01_dataset_exploration.ipynb) | Explores the selected dataset, design parameters, geometry, and representative Touchstone responses. |
| NB02 | [`nb02_data_preprocessing.ipynb`](sparam-surrogate/notebooks/nb02_data_preprocessing.ipynb) | Aligns parameter and Touchstone records, creates the fixed design-level split, builds preprocessing artifacts, and checks split leakage. |
| NB03 | [`nb03_non_neural_modelling.ipynb`](sparam-surrogate/notebooks/nb03_non_neural_modelling.ipynb) | Trains and evaluates Scalar Ridge, Vector Ridge, Polynomial Ridge, and Random Forest baselines. |
| NB04 | [`nb04_neural_baseline.ipynb`](sparam-surrogate/notebooks/nb04_neural_baseline.ipynb) | Trains and evaluates the point-wise neural and polynomial-neural multilayer-perceptron baselines. |
| NB05 | [`nb05_curve_neural_model.ipynb`](sparam-surrogate/notebooks/nb05_curve_neural_model.ipynb) | Develops the whole-curve neural formulation and its validation-based decoder, frequency-feature, and loss comparisons. |
| NB06 | [`nb06_full_smatrix_physics.ipynb`](sparam-surrogate/notebooks/nb06_full_smatrix_physics.ipynb) | Trains the reciprocal complete complex S-matrix model and computes its accuracy and physical diagnostics. |
| NB07 | [`nb07_selected_models_evaluation_analysis.ipynb`](sparam-surrogate/notebooks/nb07_selected_models_evaluation_analysis.ipynb) | Reloads selected runs without retraining, reproduces metrics, compares models, and exports the evidence used by the final paper and Appendix E. |
| NB08 | [`nb08_appendix_d_model_graphs.ipynb`](sparam-surrogate/notebooks/nb08_appendix_d_model_graphs.ipynb) | Reloads retained estimators without retraining and exports model diagrams and the Appendix D neural-model inventory. |

NB01 is optional background. The recommended numerical regeneration order is
NB02 -> NB03 -> NB04 -> NB05 -> NB06 -> NB07 -> NB08.

## Repository map

```text
.
|-- README.md                       # assessor-facing repository index
|-- project_portfolio/              # paper, Appendices A-F, and frozen evidence
|-- sparam-surrogate/
|   |-- src/sparam_surrogate/       # implementation package
|   |-- configs/                    # fixed project and model configuration
|   |-- notebooks/                  # paired sources and executed NB01-NB08
|   |-- outputs/models/             # selected-model registry
|   |-- outputs/runs/               # selected records and local model artifacts
|   |-- reports/pdf/                # rendered notebook reports
|   `-- tests/                      # automated tests
|-- Datasets/                       # source notes; local raw archive is supplementary
|-- Documents/                      # supplementary planning and guidance material
|-- Meeting_Minutes/                # supplementary project-management record
|-- References/                     # supplementary literature collection
`-- Reviews_Research/               # supplementary review and development notes
```

## Environment and tests

The declared environment targets Apple Silicon and macOS because it pins
`tensorflow-macos` and `tensorflow-metal`. Other platforms require an appropriate
TensorFlow installation.

From the repository root:

```bash
conda env create -f sparam-surrogate/environment.yml
conda run -n meng python -m pip install -e ./sparam-surrogate
```

Register the notebook kernel if the named `sparam-surrogate` kernel is not already
available:

```bash
conda run -n meng python -m ipykernel install --user \
  --name sparam-surrogate --display-name "Python (sparam-surrogate)"
```

From the repository root, run the automated suite with:

```bash
cd sparam-surrogate
conda run -n meng python -c \
  "import numpy.typing, pytest; raise SystemExit(pytest.main())"
```

The explicit `numpy.typing` import is a compatibility step for the installed
NumPy/scikit-rf combination. On 14 August 2026, this command passed all 251 tests in
the declared `meng` environment.

## Data and execution

The raw SI/PI Database archive and processed arrays are not required to inspect the
executed notebooks, rendered reports, or exported evidence. They are supplementary and
are excluded from the Loop repository ZIP because of their size.

The configured dataset identifier is
`linkOn8CavityStackBetween10x10Array_19_08_2021`. Its extracted contents must be placed
at:

```text
sparam-surrogate/data/raw/
  linkOn8CavityStackBetween10x10Array_19_08_2021/
    parameter.csv
    variation/simu_<index>.s12p
```

Dataset provenance and source links are recorded in
[`Datasets/Readme.md`](Datasets/Readme.md).

If the supplementary archive is available in `Datasets/`, extract it from the
repository root with:

```bash
mkdir -p \
  sparam-surrogate/data/raw/linkOn8CavityStackBetween10x10Array_19_08_2021
unzip Datasets/linkOn8CavityStackBetween10x10Array_19_08_2021.zip \
  -d sparam-surrogate/data/raw/linkOn8CavityStackBetween10x10Array_19_08_2021
```

To create the documented preprocessing CSVs directly:

```bash
cd sparam-surrogate
conda run -n meng sparam-surrogate preprocess \
  --input-dir data/raw/linkOn8CavityStackBetween10x10Array_19_08_2021 \
  --output-dir data/processed \
  --nports 12
```

NB02 performs and documents the equivalent data preparation. Full notebook execution
is computationally expensive and should follow the recommended numerical order above.
The exact NB07/NB08 evidence-export commands and output descriptions are in the
[evidence README](project_portfolio/evidence/README.md).

From the repository root, check that any Jupytext pair agrees without executing it:

```bash
conda run -n meng jupytext --diff \
  sparam-surrogate/notebooks/<notebook>.py \
  sparam-surrogate/notebooks/<notebook>.ipynb
```

To render the outputs already stored in all executed notebooks without synchronising or
executing them:

```bash
cd sparam-surrogate
conda run -n meng make ipynb-to-pdf
```

The package-level [README](sparam-surrogate/README.md) documents the preprocessing API,
command-line interface, lazy Touchstone loading, and report targets in more detail.

## Building the portfolio

The portfolio uses a separate LaTeX environment. From the repository root:

```bash
conda env create -f project_portfolio/environment.yml
cd project_portfolio
conda run -n latex latexmk -pdf \
  -jobname=A00049113_EEN1095_Full_Project_Portfolio main.tex
```

The project-specific `latexmkrc` builds the six-page paper and Appendix E before
assembling the complete portfolio.

## Reproducibility, archive size, and supplementary material

The repository supports a documented, version-controlled evidence trail. The Loop
repository ZIP retains the compiled portfolio and research-paper PDFs, executed
notebooks, NB01–NB07 reports, compact selected-run records, smaller selected model
artifacts, and exported evidence. It excludes the raw data, generated caches, and the
6.1 GB selected Random Forest binary. Exact regeneration additionally depends on those
excluded artifacts:

- The raw dataset archive is approximately 2.5 GB, and its extracted form is much
  larger. Raw data and generated processed-data caches are supplementary.
- The excluded selected Random Forest binary alone is approximately 6.1 GB. The source,
  executed reports, compact run records, and final evidence remain inspectable, but
  complete NB07/NB08 regeneration is unavailable from the archive alone.
- Unselected runs, routine logs, temporary predictions, extra plots, archived reports,
  development notes, and LaTeX auxiliary files are supplementary unless a core item
  above explicitly includes them. The retained compiled PDFs are deliverables, not
  auxiliary files.
- Bootstrap intervals in the evidence resample the 1,406 fixed test designs. They do
  not include uncertainty from retraining, model search, the single selected seed, or
  prior exposure to test results.

Executed notebooks, rendered reports, data, and run outputs are intentionally ignored
by Git. The Loop ZIP must therefore be assembled from the working tree, not with
`git archive`. Before submission, unpack the ZIP in a temporary directory and verify
that all nine core items open from that copy. External links may supplement the archive
but do not replace evidence formally submitted through Loop.
