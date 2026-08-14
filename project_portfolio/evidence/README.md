# Paper Evidence Exports

NB07 generates these files from the selected model registry without training,
promoting, or reselecting a model. They support the paper's page-4 tables,
response figure, and Appendix E.

The authoritative generator is the paired notebook source:

```text
sparam-surrogate/notebooks/nb07_selected_models_evaluation_analysis.py
```

The export uses the fixed project seed 128, the 4,218/1,406/1,406 design split,
and fixed test design `SIMU_INDEX=5491`. Run NB07 with
`SPARAM_REPORT_EXPORT_DIR` set to this directory to regenerate the files.

From `sparam-surrogate/`:

```bash
conda run -n meng jupytext --sync \
  notebooks/nb07_selected_models_evaluation_analysis.py
SPARAM_REPORT_EXPORT_DIR=../project_portfolio/evidence \
  conda run -n meng jupyter nbconvert \
  notebooks/nb07_selected_models_evaluation_analysis.ipynb \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=sparam-surrogate
```

- `page4_common_target.csv`: test S7 MAE and S7-only DeepNullMAE.
- `page4_full_matrix_comparison.csv`: same-task complex reference comparison.
- `page4_response_evidence.pdf`/`.png`: generated paper response figure.
- `response_design_5491.csv`: exact traces plotted in the response figure.
- `paired_bootstrap_transitions.csv`: all paired design-level comparisons.
- `full_matrix_physics_diagnostics.csv`: physical diagnostic values.
- `selected_run_provenance.csv`: selected run identifiers and scopes.
- `metric_reproduction_checks.csv`: all 80 persisted/recomputed comparisons.
- `metric_reproduction_summary.csv`: per-model pass summary.

## Appendix D model inventory

NB08 loads the selected Keras and scikit-learn artifacts without training or
changing model selection. It exports the neural topology diagrams and
`appendix_d_neural_model_inventory.csv`, which records the selected run,
input/output shapes and parameter count for each displayed model.
It also displays the selected scikit-learn estimator pipelines and a clearly
labelled, depth-limited tree from the Random Forest as supporting notebook
views; these are not presented as Keras graphs or separately exported figures.

The authoritative paired notebook source is:

```text
sparam-surrogate/notebooks/nb08_appendix_d_model_graphs.py
```

From `sparam-surrogate/`:

```bash
conda run -n meng jupytext --sync \
  notebooks/nb08_appendix_d_model_graphs.py
conda run -n meng jupyter nbconvert \
  notebooks/nb08_appendix_d_model_graphs.ipynb \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 \
  --ExecutePreprocessor.kernel_name=sparam-surrogate
```

The non-neural models are documented by equations and configuration tables in
Appendix D because scikit-learn estimators are not Keras computation graphs.

The paper copy of the vector figure is
`project_portfolio/media/page4_response_evidence.pdf`.
