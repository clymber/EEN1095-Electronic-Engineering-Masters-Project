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
# # Selected Models: Evaluation, Analysis, and Comparison
#
# A full-wave electromagnetic simulation gives a detailed answer, but it is expensive
# to repeat for every new PCB design. The purpose of this project is to learn a
# surrogate: a faster model that maps design variables to the simulated response.
# The useful question is not simply, “Which model has the smallest error?” It is:
#
# > How did progressively richer models change what could be predicted, how
# > accurately, and at what computational and physical trade-off?
#
# I answer that question retrospectively. The models configured in
# `outputs/models/selected.json` were trained and selected during NB03–NB06. NB07
# loads those saved runs to demonstrate the exploration path and evaluate them under
# one common comparison protocol. It does not retrain a model, retune model settings,
# change which saved runs are selected, or use the test split for further model
# selection.
#
# | ID | Notebook file | Purpose |
# | --- | --- | --- |
# | **NB01** | `nb01_dataset_exploration.ipynb` | Dataset exploration |
# | **NB02** | `nb02_data_preprocessing.ipynb` | Data preprocessing |
# | **NB03** | `nb03_non_neural_modelling.ipynb` | Non-neural models |
# | **NB04** | `nb04_neural_baseline.ipynb` | Vector neural baseline |
# | **NB05** | `nb05_curve_neural_model.ipynb` | Whole-curve neural model |
# | **NB06** | `nb06_full_smatrix_physics.ipynb` | Full complex S-matrix model |
# | **NB07** | `nb07_selected_models_evaluation_analysis.ipynb` | Current notebook |
#
# *Note: These `.ipynb` notebooks are generated from their paired Jupytext `.py` source
# files.*
#
# The exploration follows four phases:
#
# 1. **Establish a baseline:** Scalar Ridge and Vector Ridge.
# 2. **Add nonlinear capacity:** Polynomial Ridge and Random Forest.
# 3. **Learn nonlinear features:** Neural MLP and Polynomial Neural MLP.
# 4. **Learn structured outputs:** Curve Neural and Full S-Matrix Neural.

# %% tags=["remove-input"]
"""
Evaluate and compare the selected surrogate-model artifacts from NB03 to NB06.
"""

# %load_ext autoreload
# %autoreload 2
# %aimport -pathlib
# %aimport -numpy

# ruff: noqa: E402 -- Configure filtered notebook output before remaining imports.
from sparam_surrogate.config import configure_stdio_relative_path

configure_stdio_relative_path()

# %% tags=["remove-input"]
import gc
import os
from time import perf_counter
from typing import Any

# Keep routine TensorFlow device and end-of-dataset messages out of stored cells.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import numpy.typing  # noqa: F401 -- Make NumPy typing visible to scikit-rf 2.0.1.
import pandas as pd
from IPython.display import Markdown, display
from nb07_support.presentation import (
    display_full_smatrix_diagnostics,
    display_headline_metrics,
    display_key_value_summary,
    display_model_choices,
    display_model_metrics,
    display_native_six_metrics,
    display_provenance_tables,
    display_reference_metrics,
    display_reproduction_summary,
    display_runtime_metrics,
    display_s7_diagnostics,
    display_selected_model_overview,
    display_split_summary,
    display_training_history,
    display_transition_result,
    display_transition_summary,
    display_validation_sweep,
    format_binary_size,
)

from sparam_surrogate.config import SurrogateConfig
from sparam_surrogate.data import CurveDataset, PointwiseDataset, TouchstoneLoader
from sparam_surrogate.outputs import ModelRegistry
from sparam_surrogate.utils.json_io import read_json

# %% [markdown]
# ## The microwave problem in plain language
#
# A 12-port scattering matrix, or **S-matrix**, describes how incident waves are
# reflected and transmitted between every pair of ports. $S_{7,1}$ is the complex
# wave measured at port 7 when port 1 is excited. This notebook uses the positive
# insertion-loss convention
#
# $$
# IL_{7,1,\mathrm{dB}}=-20\log_{10}|S_{7,1}|.
# $$
#
# A larger insertion-loss value means that less signal reaches the receiving port.
# When $|S_{7,1}|$ is extremely close to zero, the dB curve forms a high peak called
# a **transmission null**. All eight models can produce `IL_S7_1_DB`, so it is the
# common comparison target. It is an illustrative path, not a claim that one path
# represents the entire 12-port matrix.
#
# Each simulated design has ten inputs:
#
# | Variable | Engineering meaning |
# | --- | --- |
# | `EPS` | relative dielectric permittivity |
# | `TAND` | dielectric loss tangent |
# | `PITCH` | via-to-via pitch |
# | `TRACE_LEN` | transmission-line trace length |
# | `START` | global starting-position geometry parameter |
# | `VIAR` | via radius |
# | `ANTIPADR` | anti-pad radius around a via |
# | `TDIEL` | dielectric-layer thickness |
# | `DISTTL` | spacing between transmission lines |
# | `TLWIDTH` | transmission-line width |
#
# Every response is sampled at 200 frequencies from 0.5 to 100 GHz in 0.5 GHz
# steps. Point-wise models also receive frequency as an eleventh input. Curve models
# receive one ten-variable design and generate all 200 frequency positions together.

# %% [markdown]
# ### Randomness and reproducibility
#
# The project seed configures the randomized operations used in this notebook. Reading
# it from the shared configuration makes those operations reproducible when the
# notebook is rerun with the same settings.

# %%
cfg = SurrogateConfig.from_config()
random_seed = int(cfg.project.seed)
rng = np.random.default_rng(random_seed)

# %% [markdown]
# ## 1. Selected models and saved results
#
# The eight models listed in `outputs/models/selected.json` were trained and selected
# in NB03–NB06. They are loaded here in experiment order, from the first baseline to
# progressively more structured prediction tasks. The corresponding
# `*_selected.csv` tables contain the saved evaluation results. NB07 checks that these
# rows refer to the same saved runs before using them.

# %%
MODEL_ORDER = (
    "scalar_ridge",
    "vector_ridge",
    "polynomial_ridge",
    "random_forest",
    "neural_mlp",
    "polynomial_neural_mlp",
    "curve_neural",
    "full_smatrix_neural",
)
VECTOR_MODEL_NAMES = MODEL_ORDER[1:]
MODEL_LABELS = {
    "scalar_ridge": "Scalar Ridge",
    "vector_ridge": "Vector Ridge",
    "polynomial_ridge": "Polynomial Ridge",
    "random_forest": "Random Forest",
    "neural_mlp": "Neural MLP",
    "polynomial_neural_mlp": "Polynomial Neural MLP",
    "curve_neural": "Curve Neural",
    "full_smatrix_neural": "Full S-Matrix Neural",
    "global_mean": "Global training mean",
    "mean_curve": "Training mean curve",
}
registry = ModelRegistry(cfg.paths.models, project_root=cfg.paths.outputs.parent)
selected_entries = {name: registry.selected(name) for name in MODEL_ORDER}

print(f"Dataset: {cfg.dataset.name}")
print(f"Selected model list: {registry.selected_path}")
print(f"Evaluation seed: {random_seed}")

# %%
selected_metadata: dict[str, dict[str, Any]] = {}
selected_metrics: dict[str, dict[str, Any]] = {}
provenance_rows: list[dict[str, Any]] = []

for model_name in MODEL_ORDER:
    entry = selected_entries[model_name]
    run_dir = registry.resolve_path(entry.run_path)
    metadata_path = registry.resolve_path(entry.metadata_path)
    metrics_path = registry.resolve_path(entry.metrics_path)
    artifact_path = registry.resolve_path(entry.artifact_path)
    for required_path in (run_dir, metadata_path, metrics_path, artifact_path):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Selected artifact is missing for {model_name}: {required_path}"
            )

    metadata, metrics = read_json(metadata_path), read_json(metrics_path)
    selected_metadata[model_name] = metadata
    selected_metrics[model_name] = metrics

    artifact_bytes = 0
    for relative_path in metadata["artifacts"].values():
        artifact = run_dir / relative_path
        if not artifact.is_file():
            raise FileNotFoundError(f"Artifact not found: {artifact}")
        artifact_bytes += artifact.stat().st_size

    interface = metadata["data_interface"]
    summary = metrics.get("metrics", {}).get("model_summary", {})
    provenance_rows.append(
        {
            "model_name": model_name,
            "run_id": entry.run_id,
            "target_scope": interface["target_scope"],
            "artifact_size": format_binary_size(artifact_bytes),
            "parameter_count": summary.get("parameter_count", np.nan),
        }
    )

provenance = pd.DataFrame(provenance_rows)
display_selected_model_overview(provenance)

# %% [markdown]
# All eight selected runs are available. Their target scope grows from one scalar
# value to six paths, complete curves, and finally the full complex S-matrix. Artifact
# size does not grow in the same way: Random Forest occupies 6.11 GiB, whereas the
# neural model files range from hundreds of KiB to 1.52 MiB. This computational
# trade-off matters alongside accuracy.

# %%
benchmark_specs = {
    "s7_1_insertion_loss_db_selected.csv": set(MODEL_ORDER),
    "vector_insertion_loss_db_selected.csv": set(VECTOR_MODEL_NAMES),
    "per_target_insertion_loss_db_selected.csv": set(VECTOR_MODEL_NAMES),
}
tables: dict[str, pd.DataFrame] = {}

for filename, expected_models in benchmark_specs.items():
    benchmark_path = cfg.paths.benchmarks / filename
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Selected benchmark view is missing: {benchmark_path}")
    table = pd.read_csv(benchmark_path)
    key_columns = (
        ["model_name", "target_name"]
        if filename == "per_target_insertion_loss_db_selected.csv"
        else ["model_name"]
    )
    if table.duplicated(key_columns).any():
        raise RuntimeError(f"{filename} contains duplicate selected benchmark keys.")
    observed_models = set(table["model_name"])
    if observed_models != expected_models:
        raise RuntimeError(
            f"{filename} model scope is stale: expected {sorted(expected_models)}, "
            f"observed {sorted(observed_models)}."
        )
    for model_name in expected_models:
        run_ids = set(table.loc[table["model_name"] == model_name, "run_id"])
        expected_run_id = selected_entries[model_name].run_id
        if run_ids != {expected_run_id}:
            raise RuntimeError(
                f"{filename} is stale for {model_name}: {sorted(run_ids)} "
                f"!= {[expected_run_id]}."
            )
    tables[filename] = table

per_target_selected = tables["per_target_insertion_loss_db_selected.csv"]
targets = {f"IL_S{receiver}_{source}_DB" for receiver, source in cfg.dataset.ports}
target_sets = per_target_selected.groupby("model_name")["target_name"].apply(set)
if not all(model_targets == targets for model_targets in target_sets):
    raise RuntimeError("The selected per-target benchmark has a stale target scope.")

s7_table = tables["s7_1_insertion_loss_db_selected.csv"].set_index("model_name")
vector_table = tables["vector_insertion_loss_db_selected.csv"].set_index("model_name")

null_threshold_db = float(vector_table.loc["curve_neural", "deep_null_threshold_db"])
high_freq_quantile_ghz = float(
    vector_table.loc["curve_neural", "high_frequency_threshold_ghz"]
)

print("Saved benchmark rows match the selected model runs.")
print(f"Frozen pooled-training deep-null threshold: {null_threshold_db:.6f} dB")
print(f"Frozen upper-frequency quantile boundary: {high_freq_quantile_ghz:.3f} GHz")

# %% [markdown]
# The saved benchmark rows agree with `selected.json`. The deep-null threshold is
# frozen at 69.189752 dB from pooled training targets, and the upper-frequency
# quantile boundary is 75.125 GHz. These values are therefore fixed before the
# selected models are compared.

# %% [markdown]
# ## 2. Common data contract
#
# The **training split** is used to fit model coefficients or neural-network weights.
# The **validation split** was used in NB03–NB06 to choose settings such as Ridge
# regularization strength or an early-stopping checkpoint. The **test split** provides
# the held-out comparison reported here. The splits are fixed at design level, so no
# frequency row from one physical design can appear in more than one split.
#
# The training set contains 4,218 designs; validation and test contain 1,406 each. At
# 200 frequencies, those become 843,600, 281,200, and 281,200 point-wise rows.
#
# Two representations are aligned below. The curve-form dataset provides the canonical
# six-path truth. The frequency-expanded table provides the eleven point-wise inputs.
# I sort every point-wise split by `(SIMU_INDEX, FREQ_GHZ)` before reshaping. Using the
# incidental CSV row order could misalign predictions and truth.

# %%
vector_curve_loader = TouchstoneLoader("vector", cfg, "il", 8)
curve_train, curve_val, curve_test = CurveDataset.from_cleaned_splits_csv(
    cfg.preprocessing.cleaned_splits_csv,
    vector_curve_loader,
    cache=True,
    cache_dir=cfg.paths.processed_data,
)
vector_curve_loader.clear_cache()

curve_splits = {"train": curve_train, "validation": curve_val, "test": curve_test}
frequencies_ghz = np.asarray(curve_train.frequencies_ghz, dtype=float)
target_names = tuple(curve_train.target_names)
s7_target_name = "IL_S7_1_DB"
s7_target_index = target_names.index(s7_target_name)
n_frequencies = len(frequencies_ghz)

canonical_ids: dict[str, np.ndarray] = {}
canonical_design_features: dict[str, np.ndarray] = {}
truth_s7: dict[str, np.ndarray] = {}
truth_six: dict[str, np.ndarray] = {}

for split_name, dataset in curve_splits.items():
    order = np.argsort(dataset.simulation_indices)
    canonical_ids[split_name] = dataset.simulation_indices[order].copy()
    canonical_design_features[split_name] = dataset.features[order].copy()
    truth_s7[split_name] = dataset.targets[order, :, s7_target_index].copy()
    if split_name in {"validation", "test"}:
        truth_six[split_name] = dataset.targets[order].copy()

if n_frequencies != 200 or len(target_names) != 6:
    raise RuntimeError("Expected a 200-frequency, six-path curve target contract.")
np.testing.assert_allclose(np.diff(frequencies_ghz), 0.5, rtol=0.0, atol=1e-12)

assert set(canonical_ids["train"]).isdisjoint(canonical_ids["validation"])
assert set(canonical_ids["train"]).isdisjoint(canonical_ids["test"])
assert set(canonical_ids["validation"]).isdisjoint(canonical_ids["test"])

# The curve dataset objects retain the complete six-path training tensor. Only the
# compact canonical arrays above are needed until the selected curve model is loaded.
del curve_train, curve_val, curve_test, curve_splits
_ = gc.collect()

# %% [markdown]
# The canonical curves contain 200 frequency samples, six insertion-loss paths,
# 0.5 GHz spacing, and mutually disjoint training, validation, and test designs. The
# point-wise representation is aligned with the same curves below before any model is
# evaluated.

# %%
pointwise_loader = TouchstoneLoader("vector", cfg, "il", 8)
point_train, point_val, point_test = PointwiseDataset.from_frequency_expanded_csv(
    cfg.preprocessing.freq_expanded_csv,
    target_loader=pointwise_loader,
    cache=True,
)
point_splits = {"train": point_train, "validation": point_val, "test": point_test}
point_features: dict[str, np.ndarray] = {}

for split_name, dataset in point_splits.items():
    metadata = dataset.row_metadata
    simulation_ids = metadata["SIMU_INDEX"].to_numpy(dtype=np.int64)
    row_frequencies = metadata["FREQ_GHZ"].to_numpy(dtype=float)
    order = np.lexsort((row_frequencies, simulation_ids))

    sorted_ids = simulation_ids[order].reshape(-1, n_frequencies)
    sorted_frequencies = row_frequencies[order].reshape(-1, n_frequencies)
    if not np.all(sorted_ids == sorted_ids[:, :1]):
        raise RuntimeError(f"{split_name} point-wise rows mix simulation IDs.")
    if not np.array_equal(sorted_ids[:, 0], canonical_ids[split_name]):
        raise RuntimeError(f"{split_name} point-wise and curve IDs are misaligned.")
    np.testing.assert_allclose(
        sorted_frequencies,
        np.broadcast_to(frequencies_ghz, sorted_frequencies.shape),
        rtol=0.0,
        atol=pointwise_loader.FREQUENCY_TOLERANCE_GHZ,
    )

    point_features[split_name] = dataset.features[order]
    pointwise_truth = dataset.targets[order].reshape(
        len(canonical_ids[split_name]),
        n_frequencies,
        len(target_names),
    )
    if split_name in truth_six:
        np.testing.assert_allclose(
            pointwise_truth,
            truth_six[split_name],
            rtol=0.0,
            # Point-wise targets are float64; the compact curve cache is float32.
            atol=2e-5,
        )
    else:
        np.testing.assert_allclose(
            pointwise_truth[..., s7_target_index],
            truth_s7[split_name],
            rtol=0.0,
            atol=2e-5,
        )
    del pointwise_truth

pointwise_loader.clear_cache()
del point_train, point_val, point_test, point_splits
_ = gc.collect()

high_frequency_mask = frequencies_ghz >= high_freq_quantile_ghz
high_frequency_start_ghz = float(frequencies_ghz[high_frequency_mask][0])

example_rng = np.random.default_rng(random_seed)
example_id = int(example_rng.choice(np.sort(canonical_ids["test"])))
example_position = int(np.flatnonzero(canonical_ids["test"] == example_id)[0])
s7_display_context = {
    "frequencies_ghz": frequencies_ghz,
    "truth": truth_s7["test"],
    "example_position": example_position,
    "example_id": example_id,
    "high_frequency_start_ghz": high_frequency_start_ghz,
}

display_split_summary(
    canonical_ids,
    point_features,
    split_names=("train", "validation", "test"),
)
print(
    "Fixed example selected before inspecting model error: "
    f"SIMU_INDEX={example_id}, seed={random_seed}."
)

# %% [markdown]
# The point-wise and curve representations agree after sorting by simulation ID and
# frequency. Each design contributes exactly 200 rows. The fixed example is
# `SIMU_INDEX=5491`; it was chosen independently of observed model error.

# %% [markdown]
# ## 3. Metrics and comparison protocol
#
# A signed prediction residual is
#
# $$e_n=y_n-\hat y_n.$$
#
# Positive residuals mean that insertion loss was underpredicted; negative residuals
# mean it was overpredicted. **Mean absolute error** discards the sign and averages
# the error magnitude:
#
# $$MAE=\frac{1}{N}\sum_n|y_n-\hat y_n|.$$
#
# Its unit here is dB. An MAE of 7 dB means that predictions are about 7 dB from the
# truth on average. It is easy to read, but it can hide where errors occur and whether
# a few failures are severe. **Root mean squared error** is
#
# $$RMSE=\sqrt{\frac{1}{N}\sum_n(y_n-\hat y_n)^2}.$$
#
# RMSE is also in dB. Squaring gives large errors more influence, so a large gap
# between RMSE and MAE warns that tail failures matter. The median, 90th percentile,
# and 95th percentile of absolute error describe typical and tail behaviour without
# replacing the frequency-resolved plots.
#
# Aggregate error mixes the 200 frequency locations. Frequency-wise MAE instead asks
# where the model is reliable:
#
# $$MAE(f_k)=\frac{1}{N_{design}}\sum_i
# |IL_{i,k}-\widehat{IL}_{i,k}|.$$
#
# First-difference MAE compares adjacent steps,
# $\Delta_f y_{i,k}=y_{i,k+1}-y_{i,k}$. Its unit is dB per 0.5 GHz step. It measures
# local curve shape, but a vertically shifted curve can still score well.
# **Median-curve MAE** compares the median predicted response with the median true
# response at each frequency. **P10–P90 band-width MAE** compares the width containing
# the central 80% of designs. The first measures population centre; the second measures
# whether the model reproduces population spread.
#
# A deep-null score uses true $IL_{7,1}$ values at or above the frozen 69.19 dB
# threshold.
# That threshold is the 99th percentile of the **pooled six-path training tensor**,
# not an $IL_{7,1}$-only test-derived threshold. High-frequency MAE uses the
# upper-quarter training-grid boundary. The quantile is 75.125 GHz, so the sampled
# mask starts at 75.5 GHz.

# %%
PRACTICAL_MARGIN_DB = 0.10
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = random_seed + 7_007

result_rows: list[dict[str, Any]] = []
six_rows: list[dict[str, Any]] = []
reproduction_rows: list[dict[str, Any]] = []
runtime_rows: list[dict[str, Any]] = []
transition_rows: list[dict[str, Any]] = []
test_design_mae: dict[str, np.ndarray] = {}


def regression_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """
    Return MAE and RMSE for equally shaped arrays.
    """
    error = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    return {
        "MAE_dB": float(np.mean(np.abs(error))),
        "RMSE_dB": float(np.sqrt(np.mean(error**2))),
    }


def s7_diagnostics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """
    Return shared IL(7,1) aggregate, tail, shape, and distribution diagnostics.
    """
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError(
            "IL(7,1) diagnostics require aligned design-by-frequency arrays."
        )

    absolute_error = np.abs(prediction - truth)
    deep_mask = truth >= null_threshold_db
    if not np.any(deep_mask):
        raise RuntimeError("The frozen deep-null mask contains no samples.")

    true_width = np.quantile(truth, 0.9, axis=0) - np.quantile(truth, 0.1, axis=0)
    predicted_width = np.quantile(prediction, 0.9, axis=0) - np.quantile(
        prediction, 0.1, axis=0
    )
    metrics = regression_summary(truth, prediction)
    metrics.update(
        {
            "MeanResidual_dB": float(np.mean(truth - prediction)),
            "MedianAE_dB": float(np.median(absolute_error)),
            "P90AE_dB": float(np.quantile(absolute_error, 0.90)),
            "P95AE_dB": float(np.quantile(absolute_error, 0.95)),
            "FirstDifferenceMAE_dB_per_step": float(
                np.mean(np.abs(np.diff(prediction, axis=1) - np.diff(truth, axis=1)))
            ),
            "DeepNullMAE_dB": float(np.mean(absolute_error[deep_mask])),
            "HighFrequencyMAE_dB": float(
                np.mean(absolute_error[:, high_frequency_mask])
            ),
            "MedianCurveMAE_dB": float(
                np.mean(
                    np.abs(np.median(prediction, axis=0) - np.median(truth, axis=0))
                )
            ),
            "P10P90BandWidthMAE_dB": float(
                np.mean(np.abs(predicted_width - true_width))
            ),
        }
    )
    return metrics


def predict_in_batches(
    model: Any,
    features: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """
    Predict point-wise rows in bounded batches and concatenate the results.
    """
    batches = []
    for start in range(0, len(features), batch_size):
        batches.append(np.asarray(model.predict(features[start : start + batch_size])))
    return np.concatenate(batches, axis=0)


def pointwise_curves(
    row_predictions: np.ndarray,
    n_designs: int,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Reshape scalar or six-column point-wise predictions into ordered curves.
    """
    values = np.asarray(row_predictions)
    if values.ndim == 1:
        s7_values = values.reshape(n_designs, n_frequencies)
        return s7_values, None
    if values.ndim != 2 or values.shape[1] != len(target_names):
        raise ValueError(f"Unexpected point-wise prediction shape: {values.shape}")
    six_values = values.reshape(n_designs, n_frequencies, len(target_names))
    return six_values[..., target_index], six_values


def load_pointwise_predictions(
    model_name: str,
    split_names: tuple[str, ...],
) -> tuple[Any, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """
    Load one selected point-wise wrapper and predict requested splits.
    """
    entry = selected_entries[model_name]
    load_start = perf_counter()
    model = registry.load(entry)
    load_seconds = perf_counter() - load_start
    if model.name != model_name:
        raise RuntimeError(
            f"Loaded wrapper {model.name!r} does not match {model_name!r}."
        )

    if hasattr(model, "keras_model"):
        provenance.loc[
            provenance["model_name"] == model_name,
            "parameter_count",
        ] = int(model.keras_model.count_params())

    interface = selected_metadata[model_name]["data_interface"]
    if tuple(interface["input_features"]) != tuple(PointwiseDataset.FEATURE_COLUMNS):
        raise RuntimeError(f"Saved input feature order changed for {model_name}.")
    model_targets = tuple(interface["target_names"])
    expected = (s7_target_name,) if model_name == "scalar_ridge" else target_names
    if model_targets != expected:
        raise RuntimeError(f"Saved target order changed for {model_name}.")
    target_index = model_targets.index(s7_target_name)

    batch = 50_000 if model_name == "random_forest" else 100_000
    s7_predictions: dict[str, np.ndarray] = {}
    six_predictions: dict[str, np.ndarray] = {}
    prediction_seconds: dict[str, float] = {}
    for split_name in split_names:
        prediction_start = perf_counter()
        rows = predict_in_batches(model, point_features[split_name], batch_size=batch)
        prediction_seconds[split_name] = perf_counter() - prediction_start
        s7_curves, six_curves = pointwise_curves(
            rows,
            len(canonical_ids[split_name]),
            target_index,
        )
        s7_predictions[split_name] = s7_curves
        if six_curves is not None:
            six_predictions[split_name] = six_curves
        del rows

    timing = {"load_seconds": load_seconds, **prediction_seconds}
    return model, s7_predictions, six_predictions, timing


def record_result(
    model_name: str,
    s7_preds: dict[str, np.ndarray],
    six_preds: dict[str, np.ndarray] | None = None,
) -> None:
    """
    Record fresh metrics and verify saved selected IL(7,1) headline values.
    """
    for split_name, prediction in s7_preds.items():
        stats = s7_diagnostics(truth_s7[split_name], prediction)
        result_rows.append({"model_name": model_name, "split": split_name, **stats})
        if split_name == "test":
            test_design_mae[model_name] = np.mean(
                np.abs(prediction - truth_s7[split_name]), axis=1
            )

        if split_name in {"validation", "test"}:
            prefix = "val" if split_name == "validation" else "test"
            if model_name.endswith("ridge"):
                tolerance = 1e-7
            elif model_name == "random_forest":
                tolerance = 1e-8
            else:
                tolerance = 1e-4
            for metric_name, column in (
                ("MAE_dB", f"{prefix}_mae_db"),
                ("RMSE_dB", f"{prefix}_rmse_db"),
            ):
                persisted = float(s7_table.loc[model_name, column])
                recomputed = float(stats[metric_name])
                difference = recomputed - persisted
                reproduction_rows.append(
                    {
                        "model_name": model_name,
                        "split": split_name,
                        "metric": metric_name,
                        "persisted": persisted,
                        "recomputed": recomputed,
                        "difference": difference,
                        "tolerance": tolerance,
                    }
                )
                if not np.isclose(recomputed, persisted, rtol=0.0, atol=tolerance):
                    raise AssertionError(
                        f"{model_name} {split_name} {metric_name} differs from "
                        f"the selected benchmark by {difference:.6g}."
                    )

    if six_preds:
        for split_name in ("validation", "test"):
            metrics = regression_summary(truth_six[split_name], six_preds[split_name])
            six_rows.append({"model_name": model_name, "split": split_name, **metrics})
            prefix = "val" if split_name == "validation" else "test"
            for metric_name, column in (
                ("MAE_dB", f"{prefix}_mae_db"),
                ("RMSE_dB", f"{prefix}_rmse_db"),
            ):
                persisted = float(vector_table.loc[model_name, column])
                recomputed = float(metrics[metric_name])
                difference = recomputed - persisted
                reproduction_rows.append(
                    {
                        "model_name": model_name,
                        "split": split_name,
                        "metric": f"SixPath{metric_name}",
                        "persisted": persisted,
                        "recomputed": recomputed,
                        "difference": difference,
                        "tolerance": 1e-4,
                    }
                )
                if not np.isclose(recomputed, persisted, rtol=0.0, atol=1e-4):
                    raise AssertionError(
                        f"{model_name} {split_name} native six-path "
                        f"{metric_name} differs by {difference:.6g}."
                    )


METRIC_LABELS = {
    "MAE_dB": "MAE (dB)",
    "RMSE_dB": "RMSE (dB)",
    "MeanResidual_dB": "Mean residual (dB)",
    "FirstDifferenceMAE_dB_per_step": "First-difference MAE (dB/step)",
    "DeepNullMAE_dB": "Deep-null MAE (dB)",
    "HighFrequencyMAE_dB": "High-frequency MAE (dB)",
    "MedianCurveMAE_dB": "Median-curve MAE (dB)",
    "P10P90BandWidthMAE_dB": "P10–P90 width MAE (dB)",
}


def paired_bootstrap_comparison(
    predecessor: str,
    current: str,
) -> dict[str, Any]:
    """
    Compare test per-design MAE with a paired design-level bootstrap interval.
    """
    paired_difference = test_design_mae[current] - test_design_mae[predecessor]
    point_difference = float(np.mean(paired_difference))
    predecessor_mean = float(np.mean(test_design_mae[predecessor]))
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_means = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    chunk_size = 250
    for start in range(0, BOOTSTRAP_SAMPLES, chunk_size):
        stop = min(start + chunk_size, BOOTSTRAP_SAMPLES)
        indices = bootstrap_rng.integers(
            0,
            len(paired_difference),
            size=(stop - start, len(paired_difference)),
        )
        bootstrap_means[start:stop] = paired_difference[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])

    if lower >= -PRACTICAL_MARGIN_DB and upper <= PRACTICAL_MARGIN_DB:
        classification = "practically equivalent"
    elif upper < -PRACTICAL_MARGIN_DB:
        classification = "practically meaningful reduction"
    elif lower > PRACTICAL_MARGIN_DB:
        classification = "practically meaningful increase"
    elif upper < 0.0 or lower > 0.0:
        classification = "directionally supported; practical size unresolved"
    else:
        classification = "inconclusive"

    row = {
        "predecessor": predecessor,
        "current": current,
        "delta_MAE_dB": point_difference,
        "delta_percent": 100.0 * point_difference / predecessor_mean,
        "CI95_low_dB": float(lower),
        "CI95_high_dB": float(upper),
        "classification": classification,
    }
    transition_rows.append(row)
    return row


def display_transition(predecessor: str, current: str) -> None:
    """
    Compute and display one predeclared transition comparison.
    """
    row = paired_bootstrap_comparison(predecessor, current)
    display_transition_result(
        row,
        predecessor_label=MODEL_LABELS[predecessor],
        current_label=MODEL_LABELS[current],
        practical_margin_db=PRACTICAL_MARGIN_DB,
    )


def release_loaded_model(model: Any) -> None:
    """
    Release one loaded wrapper and clear Keras state when applicable.
    """
    family = getattr(model, "name", "")
    del model
    if family in {
        "neural_mlp",
        "polynomial_neural_mlp",
        "curve_neural",
        "full_smatrix_neural",
    }:
        import keras

        keras.backend.clear_session()
    gc.collect()


# %% [markdown]
# The transition statistic is `current - predecessor`. Negative values mean lower
# error. Models predict the same test designs, so the comparison repeatedly resamples
# complete designs with replacement. Each resampled design keeps all 200 correlated
# frequencies together. The central 95% of those resampled mean differences forms the
# bootstrap interval. An interval inside $[-0.10,0.10]$ dB is called practically
# equivalent. An interval entirely below or above that band is a practically
# meaningful reduction or increase.
#
# The 0.10 dB margin is a project interpretation threshold, not a universal microwave
# standard. The interval describes variation across these held-out designs conditional
# on the saved models. It does not include retraining seeds, hyperparameter-search
# uncertainty, repeated test inspection, or correction for multiple comparisons.

# %% [markdown]
# ### How to read the repeated plots
#
# Each model uses the same three $IL_{7,1}$ views. The **distribution plot** shows the
# median response and the central 10th–90th percentile band across test designs; it
# reveals whether the model captures both the typical curve and design-to-design
# spread. The **fixed-design plot** compares prediction and truth for one test design
# selected before model errors were inspected. The **MAE-by-frequency plot** averages
# absolute error across test designs at each frequency, showing where error is
# concentrated.

# %% [markdown]
# ## Train-only reference predictors
#
# Before asking whether Ridge is good, I need to ask whether it learned anything
# beyond central tendency. The global reference repeats one training-set mean at every
# frequency. The stronger mean-curve reference repeats one training-set average curve
# for every held-out design. Both use training targets only. They are reference lines,
# not selected models.

# %%
train_mean = float(np.mean(truth_s7["train"]))
train_mean_curve = np.mean(truth_s7["train"], axis=0)
reference_rows: list[dict[str, Any]] = []

for ref_name in ("global_mean", "mean_curve"):
    for split_name in ("validation", "test"):
        if ref_name == "global_mean":
            prediction = np.full_like(truth_s7[split_name], train_mean, dtype=float)
        else:
            prediction = np.broadcast_to(train_mean_curve, truth_s7[split_name].shape)
        metrics = regression_summary(truth_s7[split_name], prediction)
        reference_rows.append({"reference": ref_name, "split": split_name, **metrics})
        if split_name == "test":
            test_design_mae[ref_name] = np.mean(
                np.abs(prediction - truth_s7[split_name]), axis=1
            )

reference_table = pd.DataFrame(reference_rows)
display_reference_metrics(reference_table)

# %% [markdown]
# Frequency dependence is already a strong source of structure. On test designs,
# replacing one global mean with the training mean curve lowers MAE from 12.3535 to
# 7.4254 dB, about 39.9%. A design-conditioned surrogate should therefore be compared
# with the mean curve, not only with the weaker scalar mean.

# %% [markdown]
# # Phase 1 — Establishing a baseline
#
# The opening question is deliberately modest: how much can be learned before adding
# nonlinear capacity? I also want to know whether changing a scalar linear model into
# a six-output model changes its mathematics.

# %% [markdown]
# ## 4. Scalar Ridge: can a simple linear model learn anything useful?
#
# A linear regressor predicts a weighted sum of its inputs:
#
# $$\hat y=b+\mathbf w^T\mathbf x.$$
#
# The weights $\mathbf w$ and intercept $b$ are learned. The supplied features are
# fixed. “Linear” refers to the weights: the prediction is an affine plane in the
# supplied 11-dimensional feature space. Standardization first subtracts each
# training-feature mean and divides by its training standard deviation. Validation
# and test data reuse those frozen training statistics.
#
# Ridge regression minimizes squared error with an L2 weight penalty,
#
# $$\sum_n(y_n-\hat y_n)^2+\alpha\lVert\mathbf w\rVert_2^2.$$
#
# The hyperparameter $\alpha$ controls the penalty strength. Larger values shrink the
# coefficients more strongly. The penalty discourages unstable, very large
# coefficients, but it does not make the feature mapping nonlinear. The selected
# pipeline is:
#
# ```text
# 10 design variables + frequency (11)
#     → training-fitted StandardScaler
#     → Ridge(alpha=0.01)
#     → one scalar IL(7,1) value in dB (stored as IL_S7_1_DB)
# ```
#
# I began here because a low-capacity model is easy to interpret and difficult to
# hide behind. If it cannot beat a training-only reference, adding complexity needs a
# clearer justification than simply “neural networks are more powerful.”

# %%
scalar_model, scalar_s7, scalar_six, scalar_timing = load_pointwise_predictions(
    "scalar_ridge",
    ("train", "validation", "test"),
)
record_result("scalar_ridge", scalar_s7, scalar_six)
runtime_rows.append(
    {
        "model_name": "scalar_ridge",
        "load_seconds": scalar_timing["load_seconds"],
        "validation_prediction_seconds": scalar_timing["validation"],
        "test_prediction_seconds": scalar_timing["test"],
    }
)

scalar_pipeline = scalar_model.pipeline
scalar_regressor = scalar_pipeline.named_steps["model"]
scalar_scaler = scalar_pipeline.named_steps["scaler"]
scalar_equivalence_state = {
    "coef": np.asarray(scalar_regressor.coef_).copy(),
    "intercept": np.asarray(scalar_regressor.intercept_).copy(),
    "scaler_mean": np.asarray(scalar_scaler.mean_).copy(),
    "scaler_scale": np.asarray(scalar_scaler.scale_).copy(),
}

display_model_metrics(
    result_rows,
    model_name="scalar_ridge",
    metric_names=(
        "MAE_dB",
        "RMSE_dB",
        "MeanResidual_dB",
        "HighFrequencyMAE_dB",
    ),
    metric_labels=METRIC_LABELS,
)
display_s7_diagnostics(
    current_prediction=scalar_s7["test"],
    current_label=MODEL_LABELS["scalar_ridge"],
    **s7_display_context,
)
display_transition("mean_curve", "scalar_ridge")

previous_name = "scalar_ridge"
previous_test_s7 = scalar_s7["test"].copy()
release_loaded_model(scalar_model)
del scalar_model, scalar_s7, scalar_six
_ = gc.collect()

# %% [markdown]
# The mean curve is the important reference because it already captures the average
# frequency response. Scalar Ridge beats the global constant but does **not** beat the
# mean training curve. Its positive test mean residual also shows modest average
# underprediction. This defines the baseline's limitation: frequency is learnable, but
# this linear specification did not extract enough design-dependent information to
# improve on the average curve. It remains a transparent fitted reference for the
# models that follow.

# %% [markdown]
# ## 5. Vector Ridge: is one vector model more than six scalar models?
#
# Vector Ridge maps the same input to six insertion-loss values:
#
# $$\hat{\mathbf y}=\mathbf b+W^T\mathbf x.$$
#
# Each output column of $W$ is fitted independently under squared error. The outputs
# share an input matrix and the selected alpha, but the objective contains no term
# that couples one path to another. In this setting, multi-output Ridge is a stack of
# six scalar Ridge solutions packaged into one call.
#
# ```text
# same 11 inputs → same scaling → Ridge(alpha=0.01) → six IL values
# ```
#
# This experiment checks the claim directly. If the $IL_{7,1}$ coefficients and
# predictions are numerically identical, vector output is a packaging improvement
# rather than an $IL_{7,1}$ accuracy improvement. That would remove any reason to keep
# designing separate scalar architectures.

# %%
vector_model, vector_s7, vector_six, vector_timing = load_pointwise_predictions(
    "vector_ridge",
    ("train", "validation", "test"),
)
record_result("vector_ridge", vector_s7, vector_six)
runtime_rows.append(
    {
        "model_name": "vector_ridge",
        "load_seconds": vector_timing["load_seconds"],
        "validation_prediction_seconds": vector_timing["validation"],
        "test_prediction_seconds": vector_timing["test"],
    }
)

vector_pipeline = vector_model.pipeline
vector_regressor = vector_pipeline.named_steps["model"]
vector_scaler = vector_pipeline.named_steps["scaler"]
vector_s7_coefficient = np.asarray(vector_regressor.coef_)[s7_target_index]
vector_s7_intercept = np.asarray(vector_regressor.intercept_)[s7_target_index]

coefficient_difference = float(
    np.max(np.abs(vector_s7_coefficient - scalar_equivalence_state["coef"]))
)
intercept_difference = float(
    np.max(np.abs(vector_s7_intercept - scalar_equivalence_state["intercept"]))
)
test_prediction_difference = float(np.max(np.abs(vector_s7["test"] - previous_test_s7)))
np.testing.assert_allclose(
    vector_scaler.mean_, scalar_equivalence_state["scaler_mean"], atol=0.0
)
np.testing.assert_allclose(
    vector_scaler.scale_, scalar_equivalence_state["scaler_scale"], atol=0.0
)
np.testing.assert_allclose(
    vector_s7_coefficient,
    scalar_equivalence_state["coef"],
    rtol=0.0,
    atol=1e-7,
)
np.testing.assert_allclose(
    vector_s7_intercept,
    scalar_equivalence_state["intercept"],
    rtol=0.0,
    atol=1e-11,
)
np.testing.assert_allclose(vector_s7["test"], previous_test_s7, rtol=0.0, atol=2e-12)

ridge_equivalence = pd.DataFrame(
    [
        {
            "maximum coefficient difference": coefficient_difference,
            "intercept difference": intercept_difference,
            "maximum test-prediction difference (dB)": test_prediction_difference,
        }
    ]
)
display_key_value_summary(
    ridge_equivalence.iloc[0],
    key_label="Check",
    value_label="Difference",
    precision=3,
)
display_s7_diagnostics(
    current_prediction=vector_s7["test"],
    current_label=MODEL_LABELS["vector_ridge"],
    predecessor_prediction=previous_test_s7,
    predecessor_label=MODEL_LABELS[previous_name],
    **s7_display_context,
)
display_transition("scalar_ridge", "vector_ridge")

previous_name = "vector_ridge"
previous_test_s7 = vector_s7["test"].copy()
release_loaded_model(vector_model)
del vector_model, vector_s7, vector_six, scalar_equivalence_state
_ = gc.collect()

# %% [markdown]
# The numerical check makes the architecture lesson precise. Six-output Ridge is a
# convenient way to return six values, but it did not learn interactions among the six
# paths.
# From this point onward, separate scalar models add no experimental information.
# The remaining question is whether the linear feature map is simply too rigid.

# %% [markdown]
# # Phase 2 — Adding nonlinear capacity
#
# Phase 1 established a transparent baseline and simplified how outputs are represented.
# Phase 2 asks whether smooth feature curvature or local tree partitions can recover
# response structure that an affine plane misses.

# %% [markdown]
# ## 6. Polynomial Ridge: can simple nonlinear curvature help?
#
# The powers-only transformer expands each standardized feature independently:
#
# $$x_j\longrightarrow[x_j,x_j^2,x_j^3].$$
#
# With 11 inputs and degree 3, this produces 33 features. It deliberately omits cross
# terms such as $x_1x_2$. A second scaler standardizes the expanded columns before
# Ridge. The selected model is therefore nonlinear in the original inputs but remains
# linear in its learned coefficients:
#
# ```text
# 11 inputs
#     → scale
#     → [x, x², x³], no interactions (33 features)
#     → rescale
#     → Ridge(alpha=50)
#     → six IL values
# ```
#
# This is a controlled way to add smooth univariate curvature without creating a very
# large full-polynomial interaction space. Frequency curvature is the most obvious
# possible benefit, but the plots—not the architecture diagram—must show whether it
# changes the held-out response distribution or high-frequency error.

# %%
polynomial_model, polynomial_s7, polynomial_six, polynomial_timing = (
    load_pointwise_predictions(
        "polynomial_ridge",
        ("train", "validation", "test"),
    )
)
record_result("polynomial_ridge", polynomial_s7, polynomial_six)
runtime_rows.append(
    {
        "model_name": "polynomial_ridge",
        "load_seconds": polynomial_timing["load_seconds"],
        "validation_prediction_seconds": polynomial_timing["validation"],
        "test_prediction_seconds": polynomial_timing["test"],
    }
)
display_model_metrics(
    result_rows,
    model_name="polynomial_ridge",
    metric_names=(
        "MAE_dB",
        "RMSE_dB",
        "FirstDifferenceMAE_dB_per_step",
        "HighFrequencyMAE_dB",
    ),
    metric_labels=METRIC_LABELS,
)
display_s7_diagnostics(
    current_prediction=polynomial_s7["test"],
    current_label=MODEL_LABELS["polynomial_ridge"],
    predecessor_prediction=previous_test_s7,
    predecessor_label=MODEL_LABELS[previous_name],
    **s7_display_context,
)
display_transition("vector_ridge", "polynomial_ridge")

previous_name = "polynomial_ridge"
previous_test_s7 = polynomial_s7["test"].copy()
release_loaded_model(polynomial_model)
del polynomial_model, polynomial_s7, polynomial_six

# Train predictions are no longer required. Releasing them before the forest avoids
# carrying an unnecessary 843,600-row matrix beside a 6.1 GB model file.
point_features.pop("train")
truth_s7.pop("train")
canonical_design_features.pop("train")
_ = gc.collect()

# %% [markdown]
# The selected powers-only model changes test MAE only slightly.
# Its paired interval lies wholly inside the ±0.10 dB band, so Vector Ridge and
# Polynomial Ridge are practically equivalent on this $IL_{7,1}$ comparison. The
# experiment limits the value of adding univariate powers alone. It does not test
# interactions.

# %% [markdown]
# ## 7. Random Forest: does local nonlinear capacity recover curve structure?
#
# A regression tree repeatedly splits the feature space with rules such as
# $x_j<t$. Each terminal leaf predicts the mean target vector of its training rows.
# A Random Forest fits many such trees to bootstrap samples—new training sets formed
# by sampling rows with replacement—and averages their predictions. This resampling
# is part of fitting the forest and is separate from the later bootstrap interval used
# to compare models. Averaging reduces the variance of an individual tree. No feature
# scaling is needed because a threshold split depends on ordering, not units.
#
# ```text
# 11 raw inputs
#     → 128 bootstrap-fitted multi-output trees
#     → unlimited depth, min_samples_leaf=2, max_features=1.0
#     → average six-value prediction
# ```
#
# `min_samples_leaf=2` requires at least two training rows in every terminal leaf.
# `max_features=1.0` makes all eleven inputs available when a split is considered.
#
# The selected configuration uses all features as candidates at each split, so the
# relevant injected randomness comes from bootstrap sampling and tree construction.
# The hypothesis is that local partitions can represent bends and design regions that
# a global polynomial misses. The engineering cost matters too: this selected model
# saved model is about 6.1 GB.

# %%
forest_model, forest_s7, forest_six, forest_timing = load_pointwise_predictions(
    "random_forest",
    ("validation", "test"),
)
record_result("random_forest", forest_s7, forest_six)
runtime_rows.append(
    {
        "model_name": "random_forest",
        "load_seconds": forest_timing["load_seconds"],
        "validation_prediction_seconds": forest_timing["validation"],
        "test_prediction_seconds": forest_timing["test"],
    }
)
display_model_metrics(
    result_rows,
    model_name="random_forest",
    metric_names=(
        "MAE_dB",
        "RMSE_dB",
        "FirstDifferenceMAE_dB_per_step",
        "MedianCurveMAE_dB",
        "P10P90BandWidthMAE_dB",
    ),
    metric_labels=METRIC_LABELS,
)
display_s7_diagnostics(
    current_prediction=forest_s7["test"],
    current_label=MODEL_LABELS["random_forest"],
    predecessor_prediction=previous_test_s7,
    predecessor_label=MODEL_LABELS[previous_name],
    **s7_display_context,
)
display_transition("polynomial_ridge", "random_forest")

previous_name = "random_forest"
previous_test_s7 = forest_s7["test"].copy()
release_loaded_model(forest_model)
del forest_model, forest_s7, forest_six
_ = gc.collect()

# %% [markdown]
# NB07 deliberately avoids a full forest prediction over the training split. Without
# that calculation, it would be wrong to diagnose overfitting from a train-test gap.
# The selected forest is larger and has meaningfully worse held-out $IL_{7,1}$ MAE
# than Polynomial Ridge. Still, the distribution plot is useful: its central band is
# less compressed, and its band-width error is lower. First-difference error is higher,
# so wider population spread did not translate into better individual curve shape.
# That trade-off motivates a smoother nonlinear approximator with a much smaller saved
# model.

# %% [markdown]
# # Phase 3 — Learning nonlinear features
#
# The point-wise formulation stays fixed in this phase. The question is whether a
# compact neural network can learn useful nonlinear features from the raw inputs, and
# whether manually supplying powers adds anything to that capacity.

# %% [markdown]
# ## 8. Neural MLP: can learned nonlinear features generalize better?
#
# A **multilayer perceptron (MLP)** is a feed-forward neural network. Each artificial
# neuron forms a weighted sum of its inputs, adds a bias, and applies an activation
# function. A hidden layer is an intermediate set of such neurons; it learns features
# that are useful for the final prediction rather than producing a reported output.
#
# A dense layer first forms $\mathbf z=W\mathbf x+\mathbf b$. Its ReLU activation is
# $\max(0,z)$, applied element by element. Stacking dense/ReLU layers creates a
# piecewise-linear nonlinear mapping. He initialization sets an appropriate initial
# weight scale for ReLU layers; the output layer stays linear because insertion loss
# is a continuous regression target.
#
# ```text
# 11 raw inputs → train-only input scaling
#               → Dense(128, ReLU)
#               → Dense(128, ReLU)
#               → Dense(64, ReLU)
#               → Dense(6, linear)
#               → inverse target scaling to dB
# ```
#
# Targets are also standardized. Training minimizes mean squared error in those scaled
# units, so the history loss is not a dB MAE. Adam adapts updates from mini-batch
# gradients. Gradient clipping limits an unusually large update. Learning-rate
# reduction takes smaller steps after a plateau, while early stopping restores a
# validation-qualified state instead of training indefinitely. One complete pass
# through the training rows is an **epoch**.

# %%
neural_model, neural_s7, neural_six, neural_timing = load_pointwise_predictions(
    "neural_mlp",
    ("validation", "test"),
)
record_result("neural_mlp", neural_s7, neural_six)
runtime_rows.append(
    {
        "model_name": "neural_mlp",
        "load_seconds": neural_timing["load_seconds"],
        "validation_prediction_seconds": neural_timing["validation"],
        "test_prediction_seconds": neural_timing["test"],
    }
)
display_model_metrics(
    result_rows,
    model_name="neural_mlp",
    metric_names=("MAE_dB", "RMSE_dB", "DeepNullMAE_dB"),
    metric_labels=METRIC_LABELS,
)
display_s7_diagnostics(
    current_prediction=neural_s7["test"],
    current_label=MODEL_LABELS["neural_mlp"],
    predecessor_prediction=previous_test_s7,
    predecessor_label=MODEL_LABELS[previous_name],
    **s7_display_context,
)
display_transition("random_forest", "neural_mlp")

# The forest is the immediately preceding experiment, but Vector Ridge is the
# predeclared same-formulation reference for deciding whether measured test MAE differs.
display_transition("vector_ridge", "neural_mlp")

previous_name = "neural_mlp"
previous_test_s7 = neural_s7["test"].copy()
release_loaded_model(neural_model)
del neural_model, neural_s7, neural_six
_ = gc.collect()

# %% [markdown]
# The neural model recovers from the forest's worse held-out test MAE, but the
# direct Vector Ridge comparison is the important capacity test. The practical band
# classifies their small difference as practically equivalent. The selected result
# remains near the Ridge range, so nonlinear hidden layers alone did not break the
# point-wise error plateau.

# %% [markdown]
# ## 9. Polynomial Neural MLP: can the MLP learn its own nonlinear features?
#
# The polynomial neural model expands each scaled input to powers 1 through 5, still
# without interactions. Eleven inputs become 55 columns. Those columns are rescaled
# and passed to the same 128–128–64–6 hidden architecture:
#
# ```text
# 11 inputs → scale → [x, ..., x⁵], no interactions (55 features)
#           → rescale → 128–128–64 ReLU MLP → six dB values
# ```
#
# This is a focused feature-engineering comparison. The plain MLP must construct a
# useful nonlinear representation from raw inputs; the polynomial MLP receives one
# manually chosen family of nonlinear features. The comparison is not perfectly
# parameter matched because expanding the input makes the first dense layer larger.
# Only one selected training run is available for each architecture.

# %%
(
    polynomial_neural_model,
    polynomial_neural_s7,
    polynomial_neural_six,
    poly_mlp_timing,
) = load_pointwise_predictions(
    "polynomial_neural_mlp",
    ("validation", "test"),
)
record_result("polynomial_neural_mlp", polynomial_neural_s7, polynomial_neural_six)
runtime_rows.append(
    {
        "model_name": "polynomial_neural_mlp",
        "load_seconds": poly_mlp_timing["load_seconds"],
        "validation_prediction_seconds": poly_mlp_timing["validation"],
        "test_prediction_seconds": poly_mlp_timing["test"],
    }
)
display_model_metrics(
    result_rows,
    model_name="polynomial_neural_mlp",
    metric_names=(
        "MAE_dB",
        "RMSE_dB",
        "FirstDifferenceMAE_dB_per_step",
        "HighFrequencyMAE_dB",
    ),
    metric_labels=METRIC_LABELS,
)
display_s7_diagnostics(
    current_prediction=polynomial_neural_s7["test"],
    current_label=MODEL_LABELS["polynomial_neural_mlp"],
    predecessor_prediction=previous_test_s7,
    predecessor_label=MODEL_LABELS[previous_name],
    **s7_display_context,
)
display_transition("neural_mlp", "polynomial_neural_mlp")

previous_name = "polynomial_neural_mlp"
previous_test_s7 = polynomial_neural_s7["test"].copy()
release_loaded_model(polynomial_neural_model)
del polynomial_neural_model, polynomial_neural_s7, polynomial_neural_six
_ = gc.collect()

# %% [markdown]
# The complete paired interval lies inside ±0.10 dB. Plain Neural MLP and Polynomial
# Neural MLP are therefore practically equivalent under the predeclared rule. This
# demonstrates that explicit powers supplied no practically measurable held-out
# benefit in this setup. It is consistent with the raw-input MLP already having enough
# nonlinear feature-learning capacity for this task. It does **not** prove that hidden
# neurons recreated the same polynomial terms. Representation probing and repeated-
# seed ablations would be needed for that stronger claim.
#
# The portfolio lesson is deliberately bounded: this manual powers-only expansion was
# unnecessary for the selected MLP. A neural network does not automate data cleaning,
# target construction, split design, or every other kind of data engineering.
#
# Across the selected point-wise models, $IL_{7,1}$ error remains near the same level.
# The next phase changes the question: instead of treating every design-frequency row
# as independent, can one model generate a complete curve for each design?

# %% [markdown]
# # Phase 4 — Learning structured outputs
#
# Point-wise models see frequency as another input row. They are never asked to emit a
# coherent 200-point object. This phase represents frequency structure directly, then
# expands the prediction task from six loss curves to the complete complex S-matrix.

# %% [markdown]
# ## 10. Curve Neural: can whole-curve structure improve prediction?
#
# In plain terms, this model first creates a coarse frequency response, upsamples it to
# the full 200-point grid, and then refines neighbouring frequency points together.
# This gives the network an explicit notion of a curve rather than 200 unrelated rows.
#
# The curve model receives ten scaled design variables. A dense encoder creates a
# 32-value latent representation. A projection reshapes it to 25 positions with 32
# channels. Three stride-2 transposed-convolution stages learn to upsample 25 → 50 →
# 100 → 200 positions. A transposed convolution is a learned upsampling operation; it
# is not an inverse convolution.
#
# Fixed Fourier coordinates tell the decoder where it is on the frequency grid. With
# normalized coordinate $u$, order four supplies
#
# $$[u,\sin(\pi u),\cos(\pi u),\ldots,
# \sin(4\pi u),\cos(4\pi u)].$$
#
# These are positional features, not a Fourier transform of the response and not
# learned Fourier coefficients. They are concatenated with the decoded features. A
# kernel-5 convolution refines local frequency context, and a linear 1×1 convolution
# produces six channels:
#
# ```text
# 10 design variables → Dense latent encoder (32)
#                     → project and reshape (25 × 32)
#                     → Conv1DTranspose: 50×32 → 100×16 → 200×8
#                     → concatenate 9 fixed Fourier coordinates
#                     → kernel-5 refinement
#                     → linear 1×1 convolution → 200 × 6 IL curves
# ```
#
# The curve-aware loss adds a first-difference penalty to ordinary scaled MSE:
#
# $$L=MSE(\hat y,y)+\lambda MSE(\Delta_f\hat y,\Delta_f y),$$
#
# with selected $\lambda=11.626038$. NB05 used validation-only experiments to choose
# Fourier coordinates, a compact decoder, and this loss. The selected result therefore
# belongs to the complete curve-aware package. It is not a controlled proof that any
# one ingredient caused the change.

# %%
curve_entry = selected_entries["curve_neural"]
curve_load_start = perf_counter()
curve_model = registry.load(curve_entry)
curve_load_seconds = perf_counter() - curve_load_start
if curve_model.name != "curve_neural":
    raise RuntimeError("The selected Curve Neural wrapper has the wrong model name.")
provenance.loc[provenance["model_name"] == "curve_neural", "parameter_count"] = int(
    curve_model.keras_model.count_params()
)

curve_interface = selected_metadata["curve_neural"]["data_interface"]
if tuple(curve_interface["input_features"]) != tuple(CurveDataset.feature_columns):
    raise RuntimeError("The selected Curve Neural feature order is stale.")
np.testing.assert_allclose(
    curve_interface["frequencies_ghz"],
    frequencies_ghz,
    rtol=0.0,
    atol=1e-9,
)
if tuple(curve_interface["target_names"]) != target_names:
    raise RuntimeError("The selected Curve Neural target order is stale.")
for split_name in ("validation", "test"):
    ids = np.asarray(curve_interface["split_identifiers"][split_name], dtype=np.int64)
    if set(ids) != set(canonical_ids[split_name]):
        raise RuntimeError(
            f"Selected Curve Neural {split_name} simulation IDs are stale."
        )

curve_six: dict[str, np.ndarray] = {}
curve_prediction_seconds: dict[str, float] = {}
for split_name in ("validation", "test"):
    prediction_start = perf_counter()
    curve_six[split_name] = np.asarray(
        curve_model.predict(canonical_design_features[split_name])
    )
    curve_prediction_seconds[split_name] = perf_counter() - prediction_start
    if curve_six[split_name].shape != truth_six[split_name].shape:
        raise RuntimeError(
            f"Unexpected Curve Neural {split_name} shape: "
            f"{curve_six[split_name].shape}"
        )

curve_s7 = {
    split_name: prediction[..., s7_target_index]
    for split_name, prediction in curve_six.items()
}
record_result("curve_neural", curve_s7, curve_six)
runtime_rows.append(
    {
        "model_name": "curve_neural",
        "load_seconds": curve_load_seconds,
        "validation_prediction_seconds": curve_prediction_seconds["validation"],
        "test_prediction_seconds": curve_prediction_seconds["test"],
    }
)

deep_test_mask = truth_s7["test"] >= null_threshold_db
deep_null_prediction_summary = pd.DataFrame(
    [
        {
            "deep-null test points": int(np.sum(deep_test_mask)),
            "truth minimum (dB)": float(np.min(truth_s7["test"][deep_test_mask])),
            "truth median (dB)": float(np.median(truth_s7["test"][deep_test_mask])),
            "truth maximum (dB)": float(np.max(truth_s7["test"][deep_test_mask])),
            "prediction minimum (dB)": float(np.min(curve_s7["test"][deep_test_mask])),
            "prediction median (dB)": float(
                np.median(curve_s7["test"][deep_test_mask])
            ),
            "prediction maximum (dB)": float(np.max(curve_s7["test"][deep_test_mask])),
        }
    ]
)
display_key_value_summary(
    deep_null_prediction_summary.iloc[0],
    key_label="Quantity",
    value_label="Value",
    precision=3,
)
display_model_metrics(
    result_rows,
    model_name="curve_neural",
    metric_names=(
        "MAE_dB",
        "RMSE_dB",
        "FirstDifferenceMAE_dB_per_step",
        "DeepNullMAE_dB",
        "HighFrequencyMAE_dB",
    ),
    metric_labels=METRIC_LABELS,
)
display_s7_diagnostics(
    current_prediction=curve_s7["test"],
    current_label=MODEL_LABELS["curve_neural"],
    predecessor_prediction=previous_test_s7,
    predecessor_label=MODEL_LABELS[previous_name],
    **s7_display_context,
)
display_transition("polynomial_neural_mlp", "curve_neural")

previous_name = "curve_neural"
previous_test_s7 = curve_s7["test"].copy()
release_loaded_model(curve_model)
del curve_model, curve_s7, curve_six
_ = gc.collect()

# %% [markdown]
# Curve Neural model lowers test MAE by 0.1204 dB. Its paired interval
# excludes zero but overlaps the −0.10 dB practical boundary, so the reduction is
# directionally supported while its practical size remains unresolved. Median-curve
# error falls, but first-difference error rises slightly. The combined curve-aware
# package improved the centre of the response, not every aspect of local shape.
#
# The deep-null table exposes the sharper limitation. Conditional predictions top out
# below 40 dB while the corresponding truth reaches 267.63 dB. A lower average error
# does not mean that rare nulls are solved.
#
# Six IL curves also omit phase, reflections, most port pairs, and the complete complex
# response. They do not guarantee reciprocity. The final experiment therefore changes
# the task rather than merely enlarging the same output head.

# %% [markdown]
# ## 11. Full S-Matrix Neural: can the complete complex response be learned?
#
# A complex S-parameter is $S=a+jb$, where $a$ and $b$ are real and imaginary parts.
# The full model predicts finite real/imaginary values instead of directly regressing
# potentially extreme dB peaks. Each complex entry uses one RMS scale shared by its
# real and imaginary components.
#
# The model repeats the ten scaled design variables over the frequency grid. It adds
# the same nine Fourier coordinates used above and 32 localized Gaussian radial basis
# functions (RBFs):
#
# $$\phi_m(u)=\exp\left[-\frac{(u-c_m)^2}{2\sigma^2}\right].$$
#
# Fourier coordinates give global periodic position; Gaussian RBFs activate around
# local centers. Both are fixed coordinates. A residual block is approximately
# $\mathrm{ReLU}(\mathbf x+F(\mathbf x))$. Its skip connection helps optimize a deep
# correction to an identity path. This architectural residual is unrelated to the
# prediction residual $y-\hat y$ defined earlier.
#
# ```text
# repeated 10-variable design
#     + 9 Fourier coordinates
#     + 32 Gaussian RBF coordinates
#     → width-128 projection
#     → three width-128 residual blocks
#     → 78 unique upper-triangle complex entries
#     → mirror to a reciprocal 12 × 12 matrix
# ```
#
# Seventy-eight complex entries require 156 real output channels per frequency before
# mirroring. Predicting only the upper triangle guarantees $S_{ij}=S_{ji}$ by
# construction. The base loss is MSE over RMS-scaled real and imaginary channels. A
# weight-0.1 Huber term on stable natural-log magnitudes covers the default
# off-diagonal entries. Huber loss is squared for small errors and approximately
# absolute for large errors, making it less dominated by outliers than pure squared
# error. It is not the reported unscaled Complex NRMSE.
#
# The network applies the same frequency-conditioned residual MLP independently at
# each grid position. It has no convolution, recurrence, or attention between adjacent
# frequencies. Its frequency structure comes from shared weights and fixed coordinates.
# Passivity and causality were diagnostics, not enforced losses, in this selected run.

# %% [markdown]
# ### Complex and physical diagnostics
#
# Complex MAE averages the complex-plane error magnitude,
#
# $$ComplexMAE=\operatorname{mean}|\hat S-S|,$$
#
# and is dimensionless because S-parameters are amplitude ratios. Complex NRMSE is
#
# $$ComplexNRMSE=\sqrt{\frac{\sum|\hat S-S|^2}{\sum|S|^2}}.$$
#
# It normalizes total complex error by true signal energy. Neither complex metric is
# numerically comparable with dB-space MAE.
#
# Reciprocity means $S_{ij}=S_{ji}$ for this reciprocal interconnect. Passivity means
# the network cannot produce net energy; on each matrix it requires maximum singular
# value no greater than one. The finite-band Hilbert residual compares real and
# imaginary behaviour over the available positive-frequency band. It omits DC,
# negative frequencies, and all content above 100 GHz, so it is only a relative
# diagnostic—not proof of full-band causality.

# %%
from scipy.signal import hilbert

from sparam_surrogate.models.full_smatrix import (
    configured_insertion_loss_db,
    real_imag_channels_to_smatrix,
)


def empty_physics_accumulator() -> dict[str, float]:
    """
    Return primitive sums for exact design-batched physics aggregation.
    """
    return {
        "magnitude_min": np.inf,
        "magnitude_max": -np.inf,
        "reciprocity_numerator": 0.0,
        "reciprocity_denominator": 0.0,
        "passivity_count": 0.0,
        "passivity_violations": 0.0,
        "passivity_excess_sum": 0.0,
        "passivity_excess_squared_sum": 0.0,
        "maximum_singular_value": -np.inf,
        "causality_positive_sum": 0.0,
        "causality_negative_sum": 0.0,
        "causality_energy_sum": 0.0,
    }


def update_physics_accumulator(
    stats: dict[str, float],
    matrices: np.ndarray,
) -> None:
    """
    Add one complete-design batch to physics diagnostic primitive sums.
    """
    values = np.asarray(matrices)
    magnitude = np.abs(values)
    stats["magnitude_min"] = min(stats["magnitude_min"], float(np.min(magnitude)))
    stats["magnitude_max"] = max(stats["magnitude_max"], float(np.max(magnitude)))
    stats["reciprocity_numerator"] += float(
        np.sum(np.abs(values - np.swapaxes(values, -1, -2)) ** 2)
    )
    stats["reciprocity_denominator"] += float(np.sum(magnitude**2))

    max_singular_values = np.linalg.svd(values, compute_uv=False)[..., 0].reshape(-1)
    excess = np.maximum(max_singular_values - 1.0, 0.0)
    stats["passivity_count"] += float(len(excess))
    stats["passivity_violations"] += float(np.sum(excess > 0.0))
    stats["passivity_excess_sum"] += float(np.sum(excess))
    stats["passivity_excess_squared_sum"] += float(np.sum(excess**2))
    stats["maximum_singular_value"] = max(
        stats["maximum_singular_value"],
        float(np.max(max_singular_values)),
    )

    analytic_real = hilbert(values.real, axis=1).imag
    stats["causality_positive_sum"] += float(
        np.sum(np.abs(values.imag - analytic_real) ** 2)
    )
    stats["causality_negative_sum"] += float(
        np.sum(np.abs(values.imag + analytic_real) ** 2)
    )
    stats["causality_energy_sum"] += float(np.sum(magnitude**2))


def finalize_physics_accumulator(
    stats: dict[str, float],
) -> dict[str, float]:
    """
    Convert physics primitive sums into the reported aggregate diagnostics.
    """
    passivity_count = stats["passivity_count"]
    return {
        "MagnitudeMin": stats["magnitude_min"],
        "MagnitudeMax": stats["magnitude_max"],
        "ReciprocityResidual": float(
            np.sqrt(stats["reciprocity_numerator"] / stats["reciprocity_denominator"])
        ),
        "PassivityViolationFraction": stats["passivity_violations"] / passivity_count,
        "MeanPassivityExcess": stats["passivity_excess_sum"] / passivity_count,
        "PassivityPenalty": stats["passivity_excess_squared_sum"] / passivity_count,
        "MaximumSingularValue": stats["maximum_singular_value"],
        "BandLimitedCausalityResidual": float(
            np.sqrt(
                min(
                    stats["causality_positive_sum"],
                    stats["causality_negative_sum"],
                )
                / stats["causality_energy_sum"]
            )
        ),
    }


def evaluate_full_split(
    model: Any,
    features: np.ndarray,
    target_channels: np.ndarray,
    *,
    n_ports: int,
    entry_indices: tuple[int, ...],
    design_batch_size: int = 16,
) -> dict[str, Any]:
    """
    Stream full predictions while retaining only six-path curves and metric sums.
    """
    complex_absolute_error_sum = 0.0
    complex_squared_error_sum = 0.0
    complex_truth_energy_sum = 0.0
    complex_count = 0
    truth_physics = empty_physics_accumulator()
    prediction_physics = empty_physics_accumulator()
    true_path_batches: list[np.ndarray] = []
    predicted_path_batches: list[np.ndarray] = []

    for start in range(0, len(features), design_batch_size):
        stop = min(start + design_batch_size, len(features))
        true_channels = np.asarray(target_channels[start:stop])
        predicted_channels = np.asarray(model.predict(features[start:stop]))
        if predicted_channels.shape != true_channels.shape:
            raise RuntimeError(
                "Full S-matrix prediction shape does not match its truth batch."
            )

        true_matrices = real_imag_channels_to_smatrix(true_channels, n_ports)
        predicted_matrices = real_imag_channels_to_smatrix(predicted_channels, n_ports)
        complex_error = predicted_matrices - true_matrices
        complex_absolute_error_sum += float(np.sum(np.abs(complex_error)))
        complex_squared_error_sum += float(np.sum(np.abs(complex_error) ** 2))
        complex_truth_energy_sum += float(np.sum(np.abs(true_matrices) ** 2))
        complex_count += int(complex_error.size)

        update_physics_accumulator(truth_physics, true_matrices)
        update_physics_accumulator(prediction_physics, predicted_matrices)
        true_path_batches.append(
            configured_insertion_loss_db(true_channels, n_ports, entry_indices).astype(
                np.float32
            )
        )
        predicted_path_batches.append(
            configured_insertion_loss_db(
                predicted_channels, n_ports, entry_indices
            ).astype(np.float32)
        )
        del (
            true_channels,
            predicted_channels,
            true_matrices,
            predicted_matrices,
            complex_error,
        )

    return {
        "truth_six": np.concatenate(true_path_batches, axis=0),
        "prediction_six": np.concatenate(predicted_path_batches, axis=0),
        "complex": {
            "ComplexMAE": complex_absolute_error_sum / complex_count,
            "ComplexNRMSE": float(
                np.sqrt(complex_squared_error_sum / complex_truth_energy_sum)
            ),
        },
        "truth_physics": finalize_physics_accumulator(truth_physics),
        "prediction_physics": finalize_physics_accumulator(prediction_physics),
    }


def complex_reference_metrics(
    target_channels: np.ndarray,
    reference_channels: np.ndarray,
    *,
    design_batch_size: int = 64,
) -> dict[str, float]:
    """
    Evaluate one complex reference curve against a design split in batches.
    """
    complex_absolute_error_sum = 0.0
    complex_squared_error_sum = 0.0
    complex_truth_energy_sum = 0.0
    complex_count = 0
    reference = np.asarray(reference_channels, dtype=np.float32)
    n_entries = reference.shape[-1] // 2

    for start in range(0, len(target_channels), design_batch_size):
        stop = min(start + design_batch_size, len(target_channels))
        target_batch = np.asarray(target_channels[start:stop], dtype=np.float32)
        real = target_batch[..., :n_entries]
        imag = target_batch[..., n_entries:]
        real_error = real - reference[..., :n_entries]
        imag_error = imag - reference[..., n_entries:]
        squared_error = real_error**2 + imag_error**2
        complex_absolute_error_sum += float(
            np.sum(np.sqrt(squared_error), dtype=np.float64)
        )
        complex_squared_error_sum += float(
            np.sum(squared_error, dtype=np.float64)
        )
        complex_truth_energy_sum += float(
            np.sum(real**2 + imag**2, dtype=np.float64)
        )
        complex_count += int(real_error.size)

    return {
        "ComplexMAE": complex_absolute_error_sum / complex_count,
        "ComplexNRMSE": float(
            np.sqrt(complex_squared_error_sum / complex_truth_energy_sum)
        ),
    }


# %%
# Release every point-wise feature matrix and residual estimator reference before the
# 1.5 GB complete-complex cache is opened.
point_features.clear()
canonical_design_features.clear()
for optional_name in (
    "scalar_pipeline",
    "scalar_regressor",
    "scalar_scaler",
    "vector_pipeline",
    "vector_regressor",
    "vector_scaler",
):
    globals().pop(optional_name, None)
_ = gc.collect()

smatrix_loader = TouchstoneLoader("smatrix", cfg, "real_imag", 8)
full_train, full_val, full_test = CurveDataset.from_cleaned_splits_csv(
    cfg.preprocessing.cleaned_splits_csv,
    smatrix_loader,
    cache=True,
    cache_dir=cfg.paths.processed_data,
)
smatrix_loader.clear_cache()

n_ports = int(cfg.dataset.nports)
configured_entry_indices = tuple(
    (receiver - 1) * n_ports + source - 1 for receiver, source in cfg.dataset.ports
)

full_interface = selected_metadata["full_smatrix_neural"]["data_interface"]
if tuple(full_interface["input_features"]) != tuple(full_val.feature_columns):
    raise RuntimeError("The selected full-matrix input feature order is stale.")
if tuple(full_interface["target_names"]) != tuple(full_val.target_names):
    raise RuntimeError("The selected full-matrix target channel order is stale.")
np.testing.assert_allclose(
    full_interface["frequencies_ghz"],
    frequencies_ghz,
    rtol=0.0,
    atol=1e-9,
)
np.testing.assert_allclose(
    full_val.frequencies_ghz,
    frequencies_ghz,
    rtol=0.0,
    atol=1e-9,
)

for split_name, dataset in {"validation": full_val, "test": full_test}.items():
    ids = np.asarray(full_interface["split_identifiers"][split_name], dtype=np.int64)
    if set(dataset.simulation_indices) != set(ids):
        raise RuntimeError(
            f"Selected full-matrix {split_name} simulation IDs are stale."
        )
    if set(dataset.simulation_indices) != set(canonical_ids[split_name]):
        raise RuntimeError(
            f"Full-matrix and canonical {split_name} simulation IDs differ."
        )

# %% [markdown]
# ### Training-mean complete-matrix reference
#
# The earlier mean-curve reference predicts one training-average $IL_{7,1}$ curve for
# every design. The complete-matrix task needs its own same-task reference. At each
# frequency, I therefore average every real and imaginary S-matrix channel over
# training designs:
#
# $$
# \bar{S}_{\mathrm{train}}(f_k)
# =\frac{1}{N_{\mathrm{train}}}\sum_{n\in\mathrm{train}}S_n(f_k).
# $$
#
# Every validation or test design receives this same training-derived complex matrix.
# The reference uses no validation or test target during fitting. Beating it shows that
# the neural model learns design-conditioned information beyond an average frequency
# response; it does not establish industrial usefulness or replacement of simulation.

# %%
full_reference_channels = np.mean(
    full_train.targets,
    axis=0,
    dtype=np.float64,
).astype(np.float32)
full_reference_rows: list[dict[str, Any]] = []
frozen_full_reference_metrics = {
    "validation": {"ComplexMAE": 0.08117783, "ComplexNRMSE": 0.83700753},
    "test": {"ComplexMAE": 0.08039935, "ComplexNRMSE": 0.83429936},
}

for split_name, dataset in {"validation": full_val, "test": full_test}.items():
    metrics = complex_reference_metrics(dataset.targets, full_reference_channels)
    full_reference_rows.append({"split": split_name, **metrics})
    for metric_name, frozen in frozen_full_reference_metrics[split_name].items():
        recomputed = float(metrics[metric_name])
        difference = recomputed - frozen
        reproduction_rows.append(
            {
                "model_name": "full_smatrix_neural",
                "split": f"{split_name}_training_mean_reference",
                "metric": metric_name,
                "persisted": frozen,
                "recomputed": recomputed,
                "difference": difference,
                "tolerance": 1e-6,
            }
        )
        if not np.isclose(recomputed, frozen, rtol=0.0, atol=1e-6):
            raise AssertionError(
                f"Full-matrix {split_name} reference {metric_name} differs by "
                f"{difference:.6g}."
            )

# The training tensor alone is close to one gigabyte. Drop it before loading the
# selected neural artifact.
del full_train
_ = gc.collect()

full_entry = selected_entries["full_smatrix_neural"]
full_load_start = perf_counter()
full_model = registry.load(full_entry)
full_load_seconds = perf_counter() - full_load_start
if full_model.name != "full_smatrix_neural":
    raise RuntimeError("The selected full-matrix wrapper has the wrong model name.")
provenance.loc[provenance["model_name"] == "full_smatrix_neural", "parameter_count"] = (
    int(full_model.keras_model.count_params())
)

full_outputs: dict[str, dict[str, Any]] = {}
full_prediction_seconds: dict[str, float] = {}
for split_name, dataset in {"validation": full_val, "test": full_test}.items():
    prediction_start = perf_counter()
    output = evaluate_full_split(
        full_model,
        dataset.features,
        dataset.targets,
        n_ports=n_ports,
        entry_indices=configured_entry_indices,
    )
    full_prediction_seconds[split_name] = perf_counter() - prediction_start
    order = np.argsort(dataset.simulation_indices)
    output["truth_six"] = output["truth_six"][order]
    output["prediction_six"] = output["prediction_six"][order]
    np.testing.assert_allclose(
        output["truth_six"],
        truth_six[split_name],
        rtol=0.0,
        # Both sources are float32, but deriving dB from complex values adds rounding.
        atol=5e-5,
    )
    full_outputs[split_name] = output

full_six = {
    split_name: output["prediction_six"] for split_name, output in full_outputs.items()
}
full_s7 = {
    split_name: prediction[..., s7_target_index]
    for split_name, prediction in full_six.items()
}
record_result("full_smatrix_neural", full_s7, full_six)
runtime_rows.append(
    {
        "model_name": "full_smatrix_neural",
        "load_seconds": full_load_seconds,
        "validation_prediction_seconds": full_prediction_seconds["validation"],
        "test_prediction_seconds": full_prediction_seconds["test"],
    }
)

complex_rows: list[dict[str, Any]] = []
physics_rows: list[dict[str, Any]] = []
for split_name, output in full_outputs.items():
    complex_rows.append({"split": split_name, **output["complex"]})
    physics_rows.extend(
        [
            {
                "split": split_name,
                "matrix": "truth",
                **output["truth_physics"],
            },
            {
                "split": split_name,
                "matrix": "prediction",
                **output["prediction_physics"],
            },
        ]
    )

complex_table = pd.DataFrame(complex_rows)
physics_table = pd.DataFrame(physics_rows)

# %% [markdown]
# ### Same-task complex comparison
#
# The reference and neural rows below use the same target representation, split, and
# complex metrics. This comparison is therefore interpretable in a way that a direct
# numerical comparison between complex-domain error and dB insertion-loss error is not.

# %%
full_reference_table = pd.DataFrame(full_reference_rows)
full_reference_table.insert(0, "model", "Training-Mean Reference")
full_model_complex_table = complex_table.copy()
full_model_complex_table.insert(0, "model", MODEL_LABELS["full_smatrix_neural"])
full_complex_comparison_table = pd.concat(
    [full_reference_table, full_model_complex_table],
    ignore_index=True,
)
display(full_complex_comparison_table.set_index(["model", "split"]))

del full_reference_channels
_ = gc.collect()

# %%
persisted_full_metrics = selected_metrics["full_smatrix_neural"]["metrics"]
for row in complex_rows:
    split_name = row["split"]
    for metric_name in ("ComplexMAE", "ComplexNRMSE"):
        persisted = float(persisted_full_metrics[split_name][metric_name])
        recomputed = float(row[metric_name])
        difference = recomputed - persisted
        reproduction_rows.append(
            {
                "model_name": "full_smatrix_neural",
                "split": split_name,
                "metric": metric_name,
                "persisted": persisted,
                "recomputed": recomputed,
                "difference": difference,
                "tolerance": 1e-6,
            }
        )
        if not np.isclose(recomputed, persisted, rtol=0.0, atol=1e-6):
            raise AssertionError(
                f"Full-matrix {split_name} {metric_name} differs by "
                f"{difference:.6g}."
            )

persisted_test_physics = persisted_full_metrics["physics"]
for matrix_name, persisted_key in (
    ("truth", "test_truth"),
    ("prediction", "test_prediction"),
):
    recomputed_row = next(
        row
        for row in physics_rows
        if row["split"] == "test" and row["matrix"] == matrix_name
    )
    for metric_name, persisted in persisted_test_physics[persisted_key].items():
        recomputed = float(recomputed_row[metric_name])
        difference = recomputed - float(persisted)
        reproduction_rows.append(
            {
                "model_name": "full_smatrix_neural",
                "split": f"test_{matrix_name}",
                "metric": metric_name,
                "persisted": float(persisted),
                "recomputed": recomputed,
                "difference": difference,
                "tolerance": 2e-6,
            }
        )
        if not np.isclose(recomputed, persisted, rtol=0.0, atol=2e-6):
            raise AssertionError(
                f"Full-matrix test {matrix_name} {metric_name} differs by "
                f"{difference:.6g}."
            )

display_full_smatrix_diagnostics(
    complex_table,
    physics_table,
)

# %%
display_model_metrics(
    result_rows,
    model_name="full_smatrix_neural",
    metric_names=(
        "MAE_dB",
        "RMSE_dB",
        "MeanResidual_dB",
        "DeepNullMAE_dB",
        "HighFrequencyMAE_dB",
    ),
    metric_labels=METRIC_LABELS,
)

# %%
display_s7_diagnostics(
    current_prediction=full_s7["test"],
    current_label=MODEL_LABELS["full_smatrix_neural"],
    predecessor_prediction=previous_test_s7,
    predecessor_label=MODEL_LABELS[previous_name],
    **s7_display_context,
)

# %%
display_transition("curve_neural", "full_smatrix_neural")

# %%
release_loaded_model(full_model)
del (
    full_model,
    full_val,
    full_test,
    full_outputs,
    full_s7,
    full_six,
    previous_test_s7,
)
_ = gc.collect()

# %% [markdown]
# On the test split, the training-mean reference reaches Complex MAE 0.08040 and
# Complex NRMSE 0.83430. Full S-Matrix Neural improves these to 0.06857 and 0.78893,
# respectively. Reciprocal mirroring gives a zero prediction residual, and no
# passivity violation is observed on this finite grid. The predicted band-limited
# causality residual is 0.4812, compared with 0.3173 for the truth; neither value is
# proof of full-band causality.
#
# The conclusion must keep three axes separate. The full model predicts a much broader
# object, and reciprocal symmetry is exact by construction. Passivity is observed on
# these evaluated designs and frequencies, not guaranteed. Its $IL_{7,1}$ accuracy is
# much worse than Curve Neural: test MAE rises from 7.3883 to 10.7903 dB, a practically
# meaningful increase of 3.4020 dB. Its negative mean residual and pronounced
# low-frequency error are visible in the plot. Its high-frequency $IL_{7,1}$ MAE is
# slightly lower, but that isolated difference is not evidence of an overall
# improvement. The full model improves output scope and reciprocal consistency, but
# it is not the most accurate selected model for $IL_{7,1}$.

# %% [markdown]
# # 12. Synthesis: what changed, what did not, and why
#
# The common table below returns to one target and one unit. Validation explains the
# historical selection evidence; test is a retrospective held-out comparison. The
# test results have already been inspected during NB03–NB06, so this is not a pristine
# new confirmatory test set. Train columns appear only for the compact Ridge-family
# models evaluated cheaply; an em dash means that NB07 deliberately did not run that
# expensive diagnostic.

# %%
results = pd.DataFrame(result_rows)
model_rank = {name: rank for rank, name in enumerate(MODEL_ORDER)}
split_rank = {"train": 0, "validation": 1, "test": 2}
results["model_rank"] = results["model_name"].map(model_rank)
results["split_rank"] = results["split"].map(split_rank)
results = results.sort_values(["model_rank", "split_rank"]).drop(
    columns=["model_rank", "split_rank"]
)
display_headline_metrics(
    results,
    model_order=MODEL_ORDER,
    model_labels=MODEL_LABELS,
)

# %% [markdown]
# Scalar and Vector Ridge have the same $IL_{7,1}$ result, while Polynomial Ridge
# changes it only slightly. Random Forest is more flexible but has a higher test MAE.
# Curve Neural gives the lowest selected six-path $IL_{7,1}$ test MAE at 7.3883 dB.
# Full S-Matrix Neural reaches 10.7903 dB on this path while addressing a much broader
# complex prediction task. Because its architecture, target representation, scaling,
# and loss also differ, this comparison does not identify output scope as the cause of
# the $IL_{7,1}$ difference.

# %%
display_transition_summary(transition_rows, model_labels=MODEL_LABELS)

# %% [markdown]
# The two Ridge transitions are practically equivalent. Random Forest increases test
# MAE by 0.3074 dB, while Neural MLP recovers 0.2969 dB relative to the forest.
# Polynomial Neural MLP differs from the plain MLP by only 0.0014 dB. Curve Neural then
# lowers MAE by 0.1204 dB, with directional support but unresolved practical size. The
# final increase accompanies a different prediction task and model formulation; it is
# not a controlled output-scope ablation.
#
# A model can reduce average error while still missing rare nulls. It can also solve a
# broader output problem while scoring worse on $IL_{7,1}$. The next two tables
# therefore keep accuracy, saved-model size, output scope, and physical properties
# separate.

# %%
runtime_table = (
    pd.DataFrame(runtime_rows)
    .merge(
        provenance.loc[
            :,
            ["model_name", "artifact_size", "parameter_count"],
        ],
        on="model_name",
        how="left",
    )
    .set_index("model_name")
    .reindex(MODEL_ORDER)
    .reset_index()
)

test_mae = results.query("split == 'test'").set_index("model_name")["MAE_dB"]
best_six_model = min(VECTOR_MODEL_NAMES, key=lambda name: float(test_mae.loc[name]))
artifact_lookup = provenance.set_index("model_name")["artifact_size"]

model_choice_rows = [
    {
        "use case": "transparent one-path fitted reference",
        "choose": "scalar_ridge",
        "test IL(7,1) MAE (dB)": test_mae.loc["scalar_ridge"],
        "artifact size": artifact_lookup.loc["scalar_ridge"],
        "output scope": "one IL(7,1) value per row",
        "physics status": "none guaranteed",
        "give up": "nonlinear capacity and multi-path output",
    },
    {
        "use case": "lowest selected six-path IL(7,1) error",
        "choose": best_six_model,
        "test IL(7,1) MAE (dB)": test_mae.loc[best_six_model],
        "artifact size": artifact_lookup.loc[best_six_model],
        "output scope": "six complete insertion-loss curves",
        "physics status": "no complex phase or guaranteed reciprocity",
        "give up": "the rest of the complex 12-port response",
    },
    {
        "use case": "complete reciprocal complex response",
        "choose": "full_smatrix_neural",
        "test IL(7,1) MAE (dB)": test_mae.loc["full_smatrix_neural"],
        "artifact size": artifact_lookup.loc["full_smatrix_neural"],
        "output scope": "200 × 12 × 12 complex S-matrix",
        "physics status": "reciprocity guaranteed; passivity observed only",
        "give up": "the lower IL(7,1) error of the six-curve model",
    },
]
display_model_choices(
    pd.DataFrame(model_choice_rows),
    model_labels=MODEL_LABELS,
)

# %% [markdown]
# The appropriate choice depends on the job. Scalar Ridge is the compact, transparent
# one-path fitted reference. Curve Neural has the lowest selected six-path $IL_{7,1}$
# test MAE.
# Full S-Matrix Neural is the choice when phase, reflections, every port pair, and exact
# reciprocal construction matter more than minimum $IL_{7,1}$ error.

# %%
transitions = {(row["predecessor"], row["current"]): row for row in transition_rows}
mean_curve_test_mae = float(
    reference_table.query("reference == 'mean_curve' and split == 'test'")[
        "MAE_dB"
    ].iloc[0]
)
scalar_test_mae = float(test_mae.loc["scalar_ridge"])
polynomial_mlp_classification = transitions[("neural_mlp", "polynomial_neural_mlp")][
    "classification"
]
curve_class = transitions[("polynomial_neural_mlp", "curve_neural")]["classification"]
polynomial_mlp_transition = transitions[("neural_mlp", "polynomial_neural_mlp")]
curve_transition = transitions[("polynomial_neural_mlp", "curve_neural")]
test_diagnostics = results.query("split == 'test'").set_index("model_name")
deep_null_summary = deep_null_prediction_summary.iloc[0]
test_prediction_physics = physics_table.query(
    "split == 'test' and matrix == 'prediction'"
).iloc[0]

display(Markdown(f"""
### Evidence-led reading

- **The first linear model set a useful boundary.** The training mean curve reached
  `{mean_curve_test_mae:.4f} dB` test MAE, compared with
  `{scalar_test_mae:.4f} dB` for Scalar Ridge. Ridge learned a frequency-dependent
  fitted mapping, but this specification did not improve on the average curve's
  design-independent prediction.
- **Scalar and Vector Ridge are the same $IL_{7,1}$ solution.** Their coefficients,
  intercept, scaler, and test predictions agree at numerical precision. Vector output
  simplifies six-path prediction; it does not create cross-output learning.
- **More point-wise nonlinearity was not automatically better.** Powers-only Ridge
  changed the aggregate estimate marginally. The forest reduced central-band-width
  error from
  `{test_diagnostics.loc['polynomial_ridge', 'P10P90BandWidthMAE_dB']:.3f}` to
  `{test_diagnostics.loc['random_forest', 'P10P90BandWidthMAE_dB']:.3f} dB`, so it
  represented population spread better. Its first-difference error and held-out MAE
  were worse. The 6.1 GB saved model learned more variability, not more accurate
  individual curves.
- **Manual polynomial features added no detected advantage to the MLP.** The paired
  interval is
  `[{polynomial_mlp_transition['CI95_low_dB']:+.4f},
  {polynomial_mlp_transition['CI95_high_dB']:+.4f}] dB`, entirely inside the practical
  band. The result is **{polynomial_mlp_classification}**. This supports the bounded
  conclusion that the raw-input MLP already had sufficient nonlinear feature-learning
  capacity for this powers-only expansion. It does not prove identical hidden
  representations.
- **Changing the output formulation helped the central response, with limits.** Curve
  Neural changed test MAE by `{curve_transition['delta_MAE_dB']:+.4f} dB`; the result
  is **{curve_class}**. Median-curve MAE fell from
  `{test_diagnostics.loc['polynomial_neural_mlp', 'MedianCurveMAE_dB']:.3f}` to
  `{test_diagnostics.loc['curve_neural', 'MedianCurveMAE_dB']:.3f} dB`, while
  first-difference MAE rose slightly. On the deep-null subset, truth reaches
  `{deep_null_summary['truth maximum (dB)']:.2f} dB`, but prediction reaches only
  `{deep_null_summary['prediction maximum (dB)']:.2f} dB`.
- **The final model learned beyond the complete-matrix mean reference.** Its test
  Complex MAE is
  `{complex_table.query("split == 'test'")['ComplexMAE'].iloc[0]:.4f}` versus
  `{full_reference_table.query("split == 'test'")['ComplexMAE'].iloc[0]:.4f}` for the
  training-mean response. Full S-Matrix Neural also predicts complex phase,
  reflections, and every port pair with exact reciprocal construction. Its overall
  $IL_{7,1}$ error is higher, although its high-frequency $IL_{7,1}$ MAE is
  `{test_diagnostics.loc['full_smatrix_neural', 'HighFrequencyMAE_dB']:.3f}` versus
  `{test_diagnostics.loc['curve_neural', 'HighFrequencyMAE_dB']:.3f} dB` for Curve
  Neural. The observed maximum singular value is
  `{test_prediction_physics['MaximumSingularValue']:.3f}`, with no grid violations.
  Accuracy, output scope, and physical guarantees remain separate criteria.
"""))

# %% [markdown]
# ## 13. Limitations and next question
#
# These results come from simulated designs, not a measured manufacturing population.
# They support interpolation over the sampled design ranges; they do not establish
# extrapolation beyond those ranges. The fixed test designs were held out from model
# fitting, but NB03–NB06 already exposed their results during development. The
# retrospective confidence intervals therefore should not be read as a new preregistered
# confirmatory study.
#
# Deep transmission nulls remain difficult because a tiny magnitude change becomes a
# large dB change. The curve model's lower mean error does not remove that tail problem.
# The full model's passivity result is observed on this finite test grid, not guaranteed
# for every possible design or unsampled frequency. Its Hilbert residual is truncated
# by the 0.5–100 GHz positive-frequency window and cannot prove full-band causality.
#
# The next research question is multi-objective: can a complete complex model retain
# exact reciprocity, enforce passivity more directly, and give important transmission
# paths and deep-null regions enough weight without sacrificing global complex accuracy?

# %% [markdown]
# # Technical evidence appendix
#
# The main analysis used compact selection summaries. This appendix exposes full run
# provenance, validation sweeps, neural histories, and native metrics. These records
# explain how each selected model was obtained without confusing training loss with
# held-out dB performance.
#
# Presentation-only table formatting and figure construction are defined in
# `nb07_support/presentation.py`. The saved-model evaluation workflow, comparison
# logic, physics diagnostics, and reproducibility checks remain in this notebook.

# %% [markdown]
# ## A.1 Selected-run provenance and native metrics
#
# These tables report the exact selected runs, native six-path and complete-matrix
# scores, physics diagnostics, saved-model size, parameter count, and measured loading
# and prediction time. They support the main comparison rather than define a second
# model-selection exercise.

# %%
display_provenance_tables(provenance, model_labels=MODEL_LABELS)
display_native_six_metrics(six_rows, model_labels=MODEL_LABELS)
display_full_smatrix_diagnostics(
    complex_table,
    physics_table,
    complex_title="Complete-matrix complex accuracy",
)
display_runtime_metrics(runtime_table, model_labels=MODEL_LABELS)

# %% [markdown]
# Curve Neural has the lowest native six-path test MAE at 7.4598 dB. Full S-Matrix
# Neural reduces test Complex NRMSE from 0.8343 for the training-mean complete-matrix
# reference to 0.7889; its constructed prediction is reciprocal to numerical precision
# and shows no passivity violation on the evaluated grid. The cost comparison is
# equally clear: Random Forest occupies 6.11 GiB, whereas the neural model files range
# from hundreds of KiB to 1.52 MiB. Runtime values describe this execution environment,
# not universal deployment latency.
#
# ## A.2 Validation sweeps from model selection
#
# A **hyperparameter** is chosen using validation data rather than learned directly as
# a coefficient or neural-network weight. For Ridge, `alpha` controls regularization:
# larger values shrink the fitted coefficients more strongly. Polynomial degree and
# Random Forest `min_samples_leaf` are also hyperparameters. The tables below are the
# validation results saved by the earlier notebooks; NB07 does not refit candidates.

# %%
for model_name in (
    "scalar_ridge",
    "vector_ridge",
    "polynomial_ridge",
    "random_forest",
):
    run_dir = registry.resolve_path(selected_entries[model_name].run_path)
    sweep_path = run_dir / "validation_results.csv"
    if not sweep_path.is_file():
        raise FileNotFoundError(f"Validation sweep is missing: {sweep_path}")
    sweep = pd.read_csv(sweep_path)
    display_validation_sweep(
        sweep,
        model_label=MODEL_LABELS[model_name],
        plot=model_name != "random_forest",
        degree_column="degree" if model_name == "polynomial_ridge" else None,
    )

# %% [markdown]
# Scalar and Vector Ridge both select `alpha=0.01`, and their validation curves are
# nearly flat over the tested range. Polynomial Ridge selects degree 3 with
# `alpha=50`; degrees 4 and 5 do not improve validation MAE. This limits the case for
# adding still more powers. The forest record contains only the selected
# `min_samples_leaf=2` configuration, so no broader tuning conclusion is drawn.
#
# ## A.3 Neural training histories
#
# An **epoch** is one complete pass through the training data. Falling training and
# validation curves indicate that the model is learning. If training error keeps
# falling while validation error rises, the model is beginning to overfit the training
# designs.
#
# The two point-wise MLP histories contain MSE in scaled target units. Their saved
# files do not record which epoch's weights were restored, so the plot marks only the
# **minimum recorded validation-loss epoch**. It must not be relabelled as the restored
# epoch. Curve Neural and Full S-Matrix Neural persist their selected epochs, which may
# differ from the numerical minimum recorded later because early stopping used a
# nonzero minimum-change rule and model-specific selection metric.

# %%
for model_name in (
    "neural_mlp",
    "polynomial_neural_mlp",
    "curve_neural",
    "full_smatrix_neural",
):
    run_dir = registry.resolve_path(selected_entries[model_name].run_path)
    history_path = run_dir / "training_history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"Training history is missing: {history_path}")
    history = pd.read_csv(history_path)

    if model_name in {"neural_mlp", "polynomial_neural_mlp"}:
        marked_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
        marker_label = f"minimum recorded validation-loss epoch {marked_epoch}"
        display_training_history(
            history,
            model_label=MODEL_LABELS[model_name],
            marker_epoch=marked_epoch,
            marker_label=marker_label,
            primary_ylabel="MSE (scaled target units)",
        )
    else:
        selected_epoch = int(
            selected_metrics[model_name]["metrics"]["model_summary"]["selected_epoch"]
        )
        if model_name == "curve_neural":
            secondary_train_column = "mae_db"
            secondary_validation_column = "val_mae_db"
            secondary_ylabel = "Six-path MAE (dB)"
        else:
            secondary_train_column = "complex_nrmse"
            secondary_validation_column = "val_complex_nrmse"
            secondary_ylabel = "Complex NRMSE"
        display_training_history(
            history,
            model_label=MODEL_LABELS[model_name],
            marker_epoch=selected_epoch,
            marker_label=f"persisted selected epoch {selected_epoch}",
            primary_ylabel="Composite loss",
            secondary_train_column=secondary_train_column,
            secondary_validation_column=secondary_validation_column,
            secondary_ylabel=secondary_ylabel,
        )

# %% [markdown]
# Neural MLP and Polynomial Neural MLP reach their minimum recorded validation MSE at
# epochs 1 and 2, then follow similar plateaus. This agrees with their practically
# equivalent held-out errors. Curve Neural retains epoch 38, close to its minimum
# recorded validation-MAE epoch. Full S-Matrix Neural retains epoch 21 even though
# later plotted changes are small; the line records the saved decision under its
# minimum-change rule, not the numerical minimum of every curve.
#
# ## A.4 Final reproducibility audit
#
# This final cell checks the evidence supporting the conclusions. It does not prove
# that the models are universally valid. It does prove that this report loaded exactly
# the selected model files, used aligned splits, reproduced saved metrics within
# documented numerical tolerances, and completed every required comparison. The table
# summarizes all checks by model; the individual comparisons remain in memory and are
# still enforced separately.

# %%
reproduction_table = pd.DataFrame(reproduction_rows)
reproduction_table["tolerance_fraction"] = (
    reproduction_table["difference"].abs() / reproduction_table["tolerance"]
)
reproduction_summary = (
    reproduction_table.groupby("model_name", sort=False)
    .agg(
        checks=("metric", "size"),
        largest_tolerance_fraction=("tolerance_fraction", "max"),
    )
    .reindex(MODEL_ORDER)
    .reset_index()
)
reproduction_summary["status"] = np.where(
    reproduction_summary["largest_tolerance_fraction"] <= 1.0,
    "pass",
    "fail",
)
display_reproduction_summary(reproduction_summary, model_labels=MODEL_LABELS)

evaluated_splits = results.groupby("model_name")["split"].apply(set)
for model_name in MODEL_ORDER:
    if not {"validation", "test"}.issubset(evaluated_splits[model_name]):
        raise AssertionError(
            f"Missing validation/test IL(7,1) metrics for {model_name}."
        )
if set(test_design_mae) != {"global_mean", "mean_curve", *MODEL_ORDER}:
    raise AssertionError("The paired comparison store has a missing model/reference.")
if not np.isfinite(results.select_dtypes(include=[np.number])).all().all():
    raise AssertionError("A recomputed IL(7,1) metric is NaN or infinite.")

test_prediction_physics = physics_table.query(
    "split == 'test' and matrix == 'prediction'"
).iloc[0]
if test_prediction_physics["ReciprocityResidual"] >= 1e-7:
    raise AssertionError("Full-matrix reciprocal construction is not numerical zero.")

print("Final NB07 audit passed.")
print(f"Eight selected runs evaluated: {', '.join(MODEL_ORDER)}")
print(f"Fixed example SIMU_INDEX: {example_id}")
print("No model was trained, registered, promoted, or used from latest.json.")

# %% [markdown]
# All 80 metric-reproduction checks pass. The largest difference uses only about 29%
# of its permitted numerical tolerance, every selected model has finite held-out
# $IL_{7,1}$ metrics, and reciprocal symmetry remains below the audit threshold. NB07
# therefore reproduces the saved evidence for all eight selected runs without training
# or promoting a model. This is a reproducibility result, not proof of performance
# beyond the evaluated data.
