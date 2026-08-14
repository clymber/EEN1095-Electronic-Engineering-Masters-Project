# EEN1095 Project Repository

This repository supports the project portfolio *Using Machine Learning to Predict
Scattering Parameters for Signal Integrity Purposes* (student A00049113). The Loop
archive would be named `A00049113_project_repository.zip`.

Start with the compiled portfolio and Appendix F under `project_portfolio/`. Appendix F
identifies nine core items for assessment and maps notebook labels NB01--NB08 to their
exact filenames and purposes.

## Quick inspection

The final evidence can be reviewed without retraining the models:

1. Read the paper and Appendices D--F in `project_portfolio/`.
2. Open the executed NB07 notebook, or its PDF under `sparam-surrogate/reports/pdf/`,
   for the selected-model comparison.
3. Inspect the machine-readable tables in `project_portfolio/evidence/`.
4. Open NB08 and `project_portfolio/media/appendix_d/` for model structures.
5. Trace selected runs through `sparam-surrogate/outputs/models/selected.json` and their
   compact run records.

The `.py` notebook files are readable Jupytext sources. Their same-stem `.ipynb` files
contain the executed outputs. Labels NB01--NB08 are repository identifiers, not
technical acronyms.

## Setup and tests

The declared environment targets Apple Silicon and macOS:

```bash
conda env create -f sparam-surrogate/environment.yml
conda run -n meng pip install -e ./sparam-surrogate
cd sparam-surrogate
conda run -n meng python -c \
  "import numpy.typing,pytest;raise SystemExit(pytest.main())"
```

Other platforms require an appropriate TensorFlow build. NB08 additionally uses Graphviz
and `pydot`, both declared in `environment.yml`. The explicit `numpy.typing` import is a
compatibility step for the installed NumPy/scikit-rf combination. With that step, the
current suite passes all 251 tests.

The raw SI/PI-Database dataset is not part of the core archive because of its size. Its
identifier and expected path are recorded in `sparam-surrogate/configs/default.json`.
With the raw data and selected model artifacts available, the notebook order is optional
NB01, then NB02 -> NB03 -> NB04 -> NB05 -> NB06 -> NB07 -> NB08.

## Core and supplementary material

The nine core items are indexed in Appendix F. They comprise the portfolio,
implementation, configuration, notebook suite, compact selected-run records, evidence
exports, tests and rendered notebook reports.

Raw data, generated caches, unselected runs, logs, temporary predictions, large
saved-model binaries, extra plots and development records are supplementary unless
Appendix F explicitly includes them in a core item. The selected Random Forest binary is
approximately 6.1 GB. If it is omitted from the Loop ZIP, the executed reports and
compact evidence remain inspectable, but complete NB07 regeneration requires that
binary.

The Loop ZIP must be assembled from the working tree, not with `git archive`. Executed
notebooks, rendered reports, data and outputs are intentionally ignored by Git. Before
submission, unpack the ZIP in a temporary directory and open or check every core item.

## Project meeting records

The `Meeting_Minutes/` directory contains the dated supervisor-meeting records. These
are supplementary project-management evidence rather than core technical review items.
