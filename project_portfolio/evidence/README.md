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

The paper copy of the vector figure is
`project_portfolio/media/page4_response_evidence.pdf`.
