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

# %% tags=["remove-input"]
"""
Train and evaluate the selected whole-curve neural model.
"""

# %load_ext autoreload
# %autoreload 2
# %aimport -pathlib
# %aimport -numpy

# ruff: noqa: E402 -- Configure filtered notebook output before remaining imports.
from sparam_surrogate.config import configure_stdio_relative_path

configure_stdio_relative_path()

# %%
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.figure import Figure

from sparam_surrogate.config import (
    CurveNeuralModelConfig,
    SurrogateConfig,
    relative_to_project_root,
)
from sparam_surrogate.data import CurveDataset, TouchstoneLoader
from sparam_surrogate.models.curve_neural import CurveNeuralModel
from sparam_surrogate.outputs import ModelRunRunner, refresh_benchmarks
from sparam_surrogate.utils.json_io import read_json
from sparam_surrogate.utils.non_neural_modelling_utils import (
    per_target_metrics,
    plot_model_mae_comparison_by_frequency,
    regression_metrics,
)

# %% [markdown]
# # S-TCNN-Style Whole-Curve Neural Model
#
# NB05 predicts all 200 frequency points for the six configured transmission
# paths in one forward pass. It keeps the useful reasoning and evidence from
# model development and executes each validation-only selection stage.
#
# ```text
# Input: 10 design parameters
#              ↓
#        compact curve decoder ← Fourier frequency coordinates
#              ↓
# Target:  200 frequencies × 6 positive insertion-loss channels
# ```
#
# Here, **Fourier** means that the decoder receives sine/cosine features of
# each normalized frequency; it does not perform an inverse Fourier transform.
# These explicit coordinates help it associate each output position with both
# smooth trends and oscillatory changes over frequency.
#
# Targets use formula: 
# $IL_{ij,\mathrm{dB}}=-20\log_{10}|S_{ij}|$.
#
# The model-development path is reproduced as executable validation evidence:
#
# 1. compare no, linear, and Fourier frequency coordinates;
# 2. compare the current-size and compact decoders;
# 3. compare point-wise and curve-aware losses; and
# 4. freeze the winner before inspecting the test targets.
#
# One shared candidate runner removes the former orchestration duplication. The
# Fourier control is reused for capacity selection, and the compact point-wise
# run is reused for loss selection, so each distinct candidate is trained once.

# %% [markdown]
# ## 1. Configuration
#
# Whole-curve data start directly from the authoritative one-row-per-design
# cleaned split CSV. The frequency-expanded CSV used by the point-wise NB03 and
# NB04 models is not an input to NB05.

# %%
cfg = SurrogateConfig.from_config()
curve_config = cfg.models.curve_neural
random_seed = cfg.project.seed
keras.utils.set_random_seed(random_seed)

print(f"Dataset: {cfg.dataset.name}")
print(f"Cleaned design-level splits: {cfg.preprocessing.cleaned_splits_csv}")
print(f"Processed data directory: {cfg.paths.processed_data}")
print("Configured IL port pairs:", *cfg.dataset.ports)
print(f"Notebook display seed: {random_seed}")

# %% [markdown]
# ## 2. Cached Whole-Curve Data
#
# `CurveDataset` converts each design into one feature vector and one complete
# response tensor:
#
# | features | targets |
# | -------- | ------- |
# | (number of designs, 10) | (number of designs, 200, 6) |
#
# A single consolidated `vector_il_curve_dataset.npz` stores the three raw,
# unscaled splits and their shared metadata. Consolidating the cache removes the
# former train/validation/test cache plumbing without changing the data
# contract. A cache is reused only when its source signature, simulation
# indices, split labels, feature order, target order, frequency grid, and array
# shapes all remain compatible. Otherwise `CurveDataset` rebuilds it from the
# cleaned CSV and `TouchstoneLoader.load_curve(...)`.

# %%
curve_loader = TouchstoneLoader(
    mode="vector",
    config=cfg,
    representation="il",
    cache_size=8,
)
train_set, val_set, test_set = CurveDataset.from_cleaned_splits_csv(
    cfg.preprocessing.cleaned_splits_csv,
    curve_loader,
    cache=True,
    cache_dir=cfg.paths.processed_data,
)

feature_names = tuple(train_set.feature_columns)
target_names = tuple(train_set.target_names)
frequencies_ghz = train_set.frequencies_ghz

# pylint: disable=invalid-name
X_train, y_train = train_set.features, train_set.targets
X_val, y_val = val_set.features, val_set.targets
X_test, y_test = test_set.features, test_set.targets
# pylint: enable=invalid-name

# %%
for dataset in (train_set, val_set, test_set):
    np.testing.assert_allclose(
        dataset.frequencies_ghz,
        frequencies_ghz,
        rtol=0.0,
        atol=curve_loader.FREQUENCY_TOLERANCE_GHZ,
    )
    assert tuple(dataset.target_names) == target_names
    print(
        f"{dataset.split_type:>5}: X={dataset.features.shape}, "
        f"y={dataset.targets.shape}, cache={dataset.cache_status}"
    )

n_frequencies, n_targets = y_train.shape[1:]
assert len(feature_names) == 10
assert (n_frequencies, n_targets) == (200, 6)

print(f"Shared curve cache: {train_set.cache_path}")
print(f"Design features: {feature_names}")
print(f"Target channels: {target_names}")
print(
    f"Frequency grid: {frequencies_ghz[0]:g}–"
    f"{frequencies_ghz[-1]:g} GHz"
)

# %% [markdown]
# ### Array-alignment verification
#
# The loader already validates finite values and within-split consistency. The
# checks below make the cross-split assumptions explicit at the point where the
# arrays enter the model. In particular, all splits must use exactly the same
# feature and target order and the same 200-point frequency grid.

# %%
expected_shapes = {
    "train": (len(X_train), len(feature_names), n_frequencies, n_targets),
    "validation": (len(X_val), len(feature_names), n_frequencies, n_targets),
    "test": (len(X_test), len(feature_names), n_frequencies, n_targets),
}
for split_name, dataset in {
    "train": train_set,
    "validation": val_set,
    "test": test_set,
}.items():
    n_designs, n_features, split_frequencies, split_targets = expected_shapes[
        split_name
    ]
    assert dataset.features.shape == (n_designs, n_features)
    assert dataset.targets.shape == (
        n_designs,
        split_frequencies,
        split_targets,
    )
    assert tuple(dataset.feature_columns) == feature_names
    assert tuple(dataset.target_names) == target_names
    assert np.isfinite(dataset.features).all()
    assert np.isfinite(dataset.targets).all()

assert set(train_set.simulation_indices).isdisjoint(
    val_set.simulation_indices
)
assert set(train_set.simulation_indices).isdisjoint(
    test_set.simulation_indices
)
assert set(val_set.simulation_indices).isdisjoint(
    test_set.simulation_indices
)

array_summary = pd.DataFrame(
    [
        {
            "split": dataset.split_type,
            "designs": len(dataset.features),
            "feature_shape": str(dataset.features.shape),
            "target_shape": str(dataset.targets.shape),
            "simulation_id_min": int(dataset.simulation_indices.min()),
            "simulation_id_max": int(dataset.simulation_indices.max()),
            "cache_status": dataset.cache_status,
        }
        for dataset in (train_set, val_set, test_set)
    ]
)
display(array_summary.style.hide(axis="index"))
print("Cross-split ordering, grid alignment, and split isolation passed.")

# %%
print(f"Touchstone cache after curve loading: {curve_loader.cache_info()}")
curve_loader.clear_cache()
print(f"Touchstone cache after clearing: {curve_loader.cache_info()}")

# %% [markdown]
# ## 3. Wrapper-Owned Train-Only Scaling
#
# `CurveNeuralModel` owns both preprocessors, so the common model interface
# accepts raw `(N, 10)` designs and returns unscaled `(N, 200, 6)` predictions
# in dB:
#
# ```text
# raw design ── input StandardScaler ── decoder ── target inverse transform
# ```
#
# The input scaler is fitted only on `X_train`. The target scaler is fitted
# channel-wise on `y_train.reshape(-1, 6)`, which gives one location and scale
# per insertion-loss channel while retaining frequency structure. Validation
# and test arrays are transformed with those fitted statistics; neither can
# influence them.
#
# Scaling is deliberately inside the serializable wrapper. Callers should never
# need to reproduce preprocessing in notebook cells, and a reloaded artifact
# receives the same raw interface as the fitted model. After final training,
# NB05 verifies the scaler statistics directly against the training arrays.

# %% [markdown]
# ## 4. S-TCNN-Style Decoder
#
# The decoder is inspired by H. M. Torun *et al.*, “A spectral convolutional
# net for co-optimization of integrated voltage regulators and embedded
# inductors,” ICCAD 2019,
# [doi:10.1109/ICCAD45719.2019.8942109](https://doi.org/10.1109/ICCAD45719.2019.8942109).
# It is described as **S-TCNN-style** because it keeps the central
# design-to-latent-to-transposed-convolution idea rather than claiming an exact
# reproduction.
#
# ```text
# scaled design (10)
#     │
#     ├─ Dense encoder (32)
#     └─ Dense projection and reshape (25 × 32)
#             │
#             ├─ Conv1DTranspose, stride 2:  50 × 32
#             ├─ Conv1DTranspose, stride 2: 100 × 16
#             └─ Conv1DTranspose, stride 2: 200 × 8
#                         │
# Fourier frequency grid ─┤
#                         └─ Conv1D refinement → linear 6-channel output
# ```
#
# The Fourier grid represents each normalized frequency $f$ as $f$ plus four
# sine/cosine pairs, $\sin(k\pi f)$ and $\cos(k\pi f)$ for $k=1,\ldots,4$.
# Concatenating these fixed features with the decoded curve gives every output
# point an explicit frequency location and several smooth periodic bases. This
# helps the network learn broad trends as well as resonant or oscillatory
# variation without requiring it to infer position from the convolutional
# output alone. A validation-only ablation retained this encoding because it
# achieved a lower validation MAE than no coordinate or a linear coordinate.
#
# Material deviations from the source architecture are:
#
# - six positive insertion-loss curves replace the published inductor response;
# - latent width, channels, kernels, and three stride-two stages are adapted to
#   the `(200, 6)` output tensor;
# - a final ordinary `Conv1D` refinement and linear output are used;
# - targets are standardized channel-wise;
# - the retained loss combines point-wise and first-difference MSE; and
# - explicit frequency coordinates were evaluated as a separate ablation.

# %% [markdown]
# ## 5. Executable Validation-Only Selection
#
# Every value in the following tables is produced in this execution. One shared
# helper owns the repeated train, validate, persist, and reload lifecycle; the
# ablation cells only declare which typed configuration changes. Test metrics
# are not requested by any candidate runner.

# %%
curve_data_interface_base = {
    "dataset_name": cfg.dataset.name,
    "input_features": feature_names,
    "target_names": target_names,
    "target_scope": "curve",
    "target_units": "dB",
    "target_representation": "insertion_loss_db",
    "input_shape": [len(feature_names)],
    "output_shape": [n_frequencies, n_targets],
    "frequency_units": "GHz",
    "frequencies_ghz": frequencies_ghz,
}


def run_curve_candidate(
    candidate_config: CurveNeuralModelConfig,
) -> dict[str, Any]:
    """
    Train, validate, persist, and reload one validation-only candidate.
    """
    runner = ModelRunRunner(
        cfg,
        CurveNeuralModel.from_config(
            candidate_config,
            frequencies_ghz=frequencies_ghz,
        ),
    )
    model = runner.train(X_train, y_train, X_val, y_val)
    metrics = runner.validate(X_val, y_val)
    prediction = model.predict(X_val)
    point_error = np.abs(prediction - y_val)
    difference_error = np.abs(
        np.diff(prediction, axis=1) - np.diff(y_val, axis=1)
    )
    runner.persist(
        data_interface=curve_data_interface_base,
        extra_metrics={
            "validation_shape": {
                "FirstDifferenceMAE": float(difference_error.mean()),
            }
        },
        metric_units={
            "MAE": "dB",
            "RMSE": "dB",
            "FirstDifferenceMAE": "dB per frequency-grid step",
        },
        refresh_benchmarks=False,
    )
    reloaded_model = runner.manager.load_model()
    np.testing.assert_allclose(
        reloaded_model.predict(X_val),
        prediction,
        rtol=1e-5,
        atol=1e-5,
    )

    history = model.history
    selected_epoch = model.selected_epoch_
    assert history is not None
    assert selected_epoch is not None
    np.testing.assert_allclose(
        history.history["val_mae_db"][selected_epoch - 1],
        metrics["MAE"],
        rtol=1e-5,
        atol=1e-5,
    )
    return {
        "model": model,
        "history": history,
        "prediction": prediction,
        "metrics": metrics,
        "parameter_count": model.keras_model.count_params(),
        "first_difference_mae_db": float(difference_error.mean()),
        "mae_by_frequency_db": point_error.mean(axis=(0, 2)),
        "difference_mae_by_frequency_db": difference_error.mean(
            axis=(0, 2)
        ),
        "run_id": runner.manager.run_id,
    }


def candidate_summary(result: dict[str, Any]) -> dict[str, Any]:
    """
    Return callback-aligned validation statistics for one candidate.
    """
    model = result["model"]
    history = result["history"]
    selected_index = model.selected_epoch_ - 1
    train_mse = float(history.history["loss"][selected_index])
    validation_mse = float(history.history["val_loss"][selected_index])
    return {
        "parameters": result["parameter_count"],
        "restored_epoch": model.selected_epoch_,
        "stopped_epoch": len(history.epoch),
        "train_scaled_mse": train_mse,
        "validation_scaled_mse": validation_mse,
        "validation_train_mse_gap": validation_mse - train_mse,
        "validation_mae_db": result["metrics"]["MAE"],
        "validation_rmse_db": result["metrics"]["RMSE"],
        "first_difference_mae_db": result["first_difference_mae_db"],
        "run_id": result["run_id"],
    }


def plot_candidate_diagnostics(
    results: dict[str, dict[str, Any]],
    title: str,
    *,
    difference: bool = False,
) -> Figure:
    """
    Plot validation history and frequency-resolved errors for candidates.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    error_key = (
        "difference_mae_by_frequency_db"
        if difference
        else "mae_by_frequency_db"
    )
    error_grid = frequencies_ghz[1:] if difference else frequencies_ghz
    for name, result in results.items():
        history = result["history"]
        epochs = np.arange(1, len(history.epoch) + 1)
        axes[0].plot(epochs, history.history["val_mae_db"], label=name)
        axes[0].axvline(
            result["model"].selected_epoch_,
            color=axes[0].lines[-1].get_color(),
            linestyle="--",
            alpha=0.5,
        )
        axes[1].plot(error_grid, result[error_key], label=name)
    axes[0].set(
        xlabel="Epoch",
        ylabel="Validation MAE (dB)",
        title="Selection history",
    )
    axes[1].set(
        xlabel="Frequency (GHz)",
        ylabel=(
            "First-difference MAE (dB)"
            if difference
            else "Validation MAE (dB)"
        ),
        title="Frequency-resolved error",
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# %% [markdown]
# ### 5.1 Explicit Frequency Coordinates
#
# The current-size point-loss control is trained three times with only the
# frequency-coordinate interface changed. Fourier uses the normalized linear
# coordinate plus four sine/cosine pairs. The lowest inverse-scaled validation
# MAE determines the retained interface.

# %%
pointwise_config = replace(curve_config, derivative_loss_weight=0.0)
current_size_config = replace(
    pointwise_config,
    latent_dim=64,
    decoder_channels=(64, 32, 16),
)
frequency_results = {
    encoding: run_curve_candidate(
        replace(current_size_config, frequency_encoding=encoding)
    )
    for encoding in ("none", "linear", "fourier")
}
frequency_ablation_results = pd.DataFrame(
    [
        {
            "frequency_encoding": encoding,
            **candidate_summary(result),
        }
        for encoding, result in frequency_results.items()
    ]
).sort_values("validation_mae_db", ignore_index=True)
display(frequency_ablation_results.style.hide(axis="index").format(precision=6))

selected_frequency_encoding = str(
    frequency_ablation_results.loc[0, "frequency_encoding"]
)
assert selected_frequency_encoding == "fourier"
display(
    plot_candidate_diagnostics(
        frequency_results,
        "Frequency-coordinate ablation",
    )
)

# %% [markdown]
# ### 5.2 Capacity and Callback-Aligned Retention
#
# The selected Fourier interface is reused. The compact decoder is retained
# when its validation MAE is within `0.05 dB` of the best candidate and its
# selected-epoch validation–training MSE gap is no more than `0.01` worse than
# the current-size control. A weight-decay run is conditional on compact-model
# rejection plus post-selection history divergence.

# %%
capacity_results = {
    "current-size": frequency_results[selected_frequency_encoding],
    "compact": run_curve_candidate(pointwise_config),
}


def select_capacity(results: dict[str, dict[str, Any]]) -> str:
    """
    Apply the predeclared validation-MAE and compact-parsimony rule.
    """
    summaries = {
        name: candidate_summary(result) for name, result in results.items()
    }
    best_name = min(
        summaries,
        key=lambda name: summaries[name]["validation_mae_db"],
    )
    compact = summaries["compact"]
    control = summaries["current-size"]
    if (
        compact["validation_mae_db"]
        <= summaries[best_name]["validation_mae_db"] + 0.05
        and compact["validation_train_mse_gap"]
        <= control["validation_train_mse_gap"] + 0.01
    ):
        return "compact"
    return best_name


def diverged_after_selection(result: dict[str, Any]) -> bool:
    """
    Return whether final train and validation losses diverged after selection.
    """
    model = result["model"]
    history = result["history"]
    selected_index = model.selected_epoch_ - 1
    training = np.asarray(history.history["loss"], dtype=float)
    validation = np.asarray(history.history["val_loss"], dtype=float)
    return bool(
        selected_index < len(history.epoch) - 1
        and training[-1] < training[selected_index]
        and validation[-1] > validation[selected_index]
    )


selected_capacity = select_capacity(capacity_results)
if (
    selected_capacity != "compact"
    and diverged_after_selection(capacity_results["compact"])
):
    capacity_results["weight-decay"] = run_curve_candidate(
        replace(current_size_config, weight_decay=1e-4)
    )
    selected_capacity = select_capacity(capacity_results)

capacity_table = pd.DataFrame(
    [
        {"candidate": name, **candidate_summary(result)}
        for name, result in capacity_results.items()
    ]
).sort_values("validation_mae_db", ignore_index=True)
display(capacity_table.style.hide(axis="index").format(precision=6))
assert selected_capacity == "compact"
display(
    plot_candidate_diagnostics(
        capacity_results,
        "Capacity and regularization ablation",
    )
)

# %% [markdown]
# ### 5.3 Curve-Aware Loss
#
# Point-wise MSE does not directly penalize local curve-shape error. On the
# uniform grid, the candidate loss adds first-difference MSE in standardized
# target units:
#
# $$
# L = \operatorname{MSE}(\hat{y},y)
# + \lambda\operatorname{MSE}(\Delta_f\hat{y},\Delta_f y).
# $$
#
# The weight is computed from this run's training targets so the reference
# derivative contribution is 10% of point-wise MSE. It is retained only when
# first-difference validation MAE improves and point-wise MAE increases by no
# more than `0.05 dB`.

# %%
selected_point_result = capacity_results[selected_capacity]
selected_point_model = selected_point_result["model"]
# Accumulate reference moments in float64 even when cache targets are float32.
scaled_train_targets = selected_point_model.y_scaler.transform(
    y_train.reshape(-1, n_targets)
).reshape(y_train.shape).astype(np.float64, copy=False)
point_reference_mse = float(np.mean(np.square(scaled_train_targets)))
derivative_reference_mse = float(
    np.mean(np.square(np.diff(scaled_train_targets, axis=1)))
)
curve_aware_weight = 0.10 * point_reference_mse / derivative_reference_mse

loss_results = {
    "point-wise": selected_point_result,
    "curve-aware": run_curve_candidate(
        replace(
            pointwise_config,
            derivative_loss_weight=curve_aware_weight,
        )
    ),
}
point_mae = loss_results["point-wise"]["metrics"]["MAE"]
point_difference_mae = loss_results["point-wise"][
    "first_difference_mae_db"
]
loss_ablation_results = pd.DataFrame(
    [
        {
            "candidate": name,
            "derivative_loss_weight": result[
                "model"
            ].derivative_loss_weight,
            **candidate_summary(result),
            "point_mae_change_db": result["metrics"]["MAE"] - point_mae,
            "difference_mae_change_db": (
                result["first_difference_mae_db"] - point_difference_mae
            ),
        }
        for name, result in loss_results.items()
    ]
)
display(loss_ablation_results.style.hide(axis="index").format(precision=6))

curve_aware_summary = candidate_summary(loss_results["curve-aware"])
retain_curve_aware_loss = bool(
    curve_aware_summary["first_difference_mae_db"] < point_difference_mae
    and curve_aware_summary["validation_mae_db"] <= point_mae + 0.05
)
assert retain_curve_aware_loss
display(
    plot_candidate_diagnostics(
        loss_results,
        "Curve-aware-loss ablation",
        difference=True,
    )
)

# %% [markdown]
# ### 5.4 Frozen Selection Summary
#
# This decision ledger is generated from the three experiment stages and is
# persisted with the final model. It is not a manually maintained result table.

# %%
validation_selection = pd.DataFrame(
    [
        {
            "selection_stage": "frequency coordinates",
            "retained_candidate": selected_frequency_encoding,
            **candidate_summary(
                frequency_results[selected_frequency_encoding]
            ),
        },
        {
            "selection_stage": "decoder capacity",
            "retained_candidate": selected_capacity,
            **candidate_summary(capacity_results[selected_capacity]),
        },
        {
            "selection_stage": "curve objective",
            "retained_candidate": "curve-aware",
            **curve_aware_summary,
        },
    ]
)
display(validation_selection.style.hide(axis="index").format(precision=6))

frozen_controls = {
    "latent_dim": 32,
    "decoder_channels": (32, 16, 8),
    "kernel_size": 5,
    "frequency_encoding": "fourier",
    "fourier_order": 4,
    "weight_decay": 0.0,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "early_stopping_patience": 8,
    "reduce_lr_patience": 3,
}
for control_name, expected_value in frozen_controls.items():
    assert getattr(curve_config, control_name) == expected_value
np.testing.assert_allclose(
    curve_config.derivative_loss_weight,
    curve_aware_weight,
    rtol=1e-6,
)

print("Frozen selected configuration:")
display(pd.Series(vars(curve_config), name="curve_neural"))

# %% [markdown]
# ## 6. Fresh Final Model Run
#
# The wrapper owns train-only input and channel-wise target scaling, Fourier
# features, the serializable curve-aware loss, callbacks, and inverse
# transformation. The runner performs the complete lifecycle in order.

# %%
curve_data_interface = {
    **curve_data_interface_base,
    "split_identifiers": {
        "train": train_set.simulation_indices,
        "validation": val_set.simulation_indices,
        "test": test_set.simulation_indices,
    },
}

curve_runner = ModelRunRunner(
    cfg,
    CurveNeuralModel.from_config(
        curve_config,
        frequencies_ghz=frequencies_ghz,
    ),
)
curve_model = curve_runner.train(X_train, y_train, X_val, y_val)
validation_metrics = curve_runner.validate(X_val, y_val)
test_metrics = curve_runner.test(X_test, y_test)
validation_prediction = curve_model.predict(X_val)
test_prediction = curve_model.predict(X_test)

print("Validation:", validation_metrics)
print("Test:", test_metrics)
print(
    f"Selected epoch {curve_model.selected_epoch_}; "
    f"{curve_model.keras_model.count_params():,} parameters"
)

# %% [markdown]
# ### 6.1 Scaling, architecture, and training verification
#
# These checks turn the wrapper contract into executable evidence. Scaler
# locations and scales must equal statistics computed from training data only.
# The curve-aware weight is also reconstructed from the fitted standardized
# training targets rather than accepted only as a configuration literal.

# %%
flat_train_targets = y_train.reshape(-1, n_targets)
input_mean = curve_model.x_scaler.mean_
input_scale = curve_model.x_scaler.scale_
target_mean = curve_model.y_scaler.mean_
target_scale = curve_model.y_scaler.scale_
assert input_mean is not None
assert input_scale is not None
assert target_mean is not None
assert target_scale is not None

np.testing.assert_allclose(input_mean, X_train.mean(axis=0))
np.testing.assert_allclose(input_scale, X_train.std(axis=0))
np.testing.assert_allclose(
    target_mean,
    flat_train_targets.mean(axis=0, dtype=np.float64),
)
np.testing.assert_allclose(
    target_scale,
    flat_train_targets.std(axis=0, dtype=np.float64),
)

# Accumulate reference moments in float64 even when cache targets are float32.
scaled_train_targets = curve_model.y_scaler.transform(
    flat_train_targets
).reshape(y_train.shape).astype(np.float64, copy=False)
point_reference_mse = float(np.mean(np.square(scaled_train_targets)))
derivative_reference_mse = float(
    np.mean(np.square(np.diff(scaled_train_targets, axis=1)))
)
reconstructed_derivative_weight = (
    0.10 * point_reference_mse / derivative_reference_mse
)
np.testing.assert_allclose(
    reconstructed_derivative_weight,
    curve_config.derivative_loss_weight,
    rtol=1e-6,
)

scaling_verification = pd.Series(
    {
        "input statistics source": "training designs only",
        "target statistics source": "training curve points only",
        "point reference MSE": point_reference_mse,
        "first-difference reference MSE": derivative_reference_mse,
        "reconstructed derivative weight": (
            reconstructed_derivative_weight
        ),
    },
    name="verified value",
)
display(scaling_verification)
print("Train-only scaler and curve-aware-weight verification passed.")

# %%
curve_model.keras_model.summary()

# %%
fig_training_history = curve_model.plot_training_history()
curve_runner.manager.save_figure(
    fig_training_history,
    "final_training_history.png",
)
plt.show()

# %% [markdown]
# ## 7. Compatible NB04 Baseline
#
# The comparison reloads the latest positive-insertion-loss `neural_mlp`.
# A compact audit checks its identity, data interface, resolved dataset paths,
# configured port pairs, and freshness before prediction.

# %%


def audit_nb04_candidate() -> tuple[Any, dict[str, Any]]:
    """
    Validate and load the latest compatible point-wise NB04 model.
    """
    entry = curve_runner.registry.latest("neural_mlp")
    run_path = curve_runner.registry.resolve_path(entry.run_path)
    metadata_path = curve_runner.registry.resolve_path(entry.metadata_path)
    config_path = run_path / "config_resolved.json"
    metadata = read_json(metadata_path)
    saved_config = read_json(config_path)["config"]
    interface = metadata["data_interface"]
    project_root = cfg.paths.outputs.parent

    expected_paths = {
        "dataset_path": relative_to_project_root(
            cfg.dataset.path,
            project_root=project_root,
        ),
        "parameter_csv": relative_to_project_root(
            cfg.dataset.parameter_csv,
            project_root=project_root,
        ),
        "cleaned_splits_csv": relative_to_project_root(
            cfg.preprocessing.cleaned_splits_csv,
            project_root=project_root,
        ),
        "freq_expanded_csv": relative_to_project_root(
            cfg.preprocessing.freq_expanded_csv,
            project_root=project_root,
        ),
    }
    candidate_created_at = datetime.fromisoformat(
        str(entry.created_at).replace("Z", "+00:00")
    )
    cleaned_modified_at = datetime.fromtimestamp(
        cfg.preprocessing.cleaned_splits_csv.stat().st_mtime,
        tz=timezone.utc,
    )
    checks = {
        "model name": (
            entry.model_name == metadata["model"]["name"] == "neural_mlp"
        ),
        "dataset name": (
            interface["dataset_name"]
            == saved_config["dataset"]["name"]
            == cfg.dataset.name
        ),
        "positive insertion loss": (
            interface["target_representation"] == "insertion_loss_db"
        ),
        "input feature order": (
            tuple(interface["input_features"])
            == (*feature_names, "FREQ_GHZ")
        ),
        "target order": tuple(interface["target_names"]) == target_names,
        "port configuration": (
            int(saved_config["dataset"]["nports"]) == cfg.dataset.nports
            and tuple(
                tuple(pair) for pair in saved_config["dataset"]["ports"]
            )
            == cfg.dataset.ports
        ),
        "dataset paths": (
            saved_config["dataset"]["path"] == expected_paths["dataset_path"]
            and saved_config["dataset"]["parameter_csv"]
            == expected_paths["parameter_csv"]
        ),
        "preprocessing paths": (
            saved_config["preprocessing"]["cleaned_splits_csv"]
            == expected_paths["cleaned_splits_csv"]
            and saved_config["preprocessing"]["freq_expanded_csv"]
            == expected_paths["freq_expanded_csv"]
        ),
        "newer than cleaned splits": candidate_created_at > cleaned_modified_at,
        "model artifact exists": curve_runner.registry.resolve_path(
            entry.artifact_path
        ).is_file(),
    }
    display(
        pd.Series(checks, name="passed")
        .rename_axis("compatibility check")
        .to_frame()
        .style
    )
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Incompatible NB04 artifact: {failed}.")

    audit = {
        "status": "compatible",
        "checks": checks,
        "run_id": entry.run_id,
        "metadata_path": relative_to_project_root(
            metadata_path,
            project_root=project_root,
        ),
        "resolved_config_path": relative_to_project_root(
            config_path,
            project_root=project_root,
        ),
        "cleaned_split_sha256": sha256(
            cfg.preprocessing.cleaned_splits_csv.read_bytes()
        ).hexdigest(),
    }
    return curve_runner.registry.load(entry), audit


def pointwise_arrays(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten curve data in design-major, frequency-minor order for NB04.
    """
    pointwise_features = np.column_stack(
        [
            np.repeat(features, n_frequencies, axis=0),
            np.tile(frequencies_ghz, len(features)),
        ]
    )
    pointwise_targets = targets.reshape(-1, n_targets)
    assert pointwise_features.shape == (
        len(features) * n_frequencies,
        len(feature_names) + 1,
    )
    np.testing.assert_allclose(
        pointwise_features[:n_frequencies, -1],
        frequencies_ghz,
    )
    return pointwise_features, pointwise_targets


nb04_model, nb04_audit = audit_nb04_candidate()
X_val_pointwise, y_val_pointwise = pointwise_arrays(X_val, y_val)
X_test_pointwise, y_test_pointwise = pointwise_arrays(X_test, y_test)
nb04_validation_prediction = nb04_model.predict(
    X_val_pointwise
).reshape(y_val.shape)
nb04_test_prediction = nb04_model.predict(
    X_test_pointwise
).reshape(y_test.shape)

# %% [markdown]
# ## 8. Metrics and Diagnostics
#
# NB04 predicts one frequency row at a time; NB05 predicts a complete design
# tensor. The arrays are flattened only for common aggregate metrics. Shape
# diagnostics retain the original design, frequency, and target axes.

# %%


def flattened_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """
    Return aggregate metrics after flattening design and frequency axes.
    """
    return regression_metrics(
        targets.reshape(-1, n_targets),
        predictions.reshape(-1, n_targets),
    )


comparison_predictions = {
    "nb04 point-wise MLP": {
        "validation": nb04_validation_prediction,
        "test": nb04_test_prediction,
    },
    "nb05 curve-aware": {
        "validation": validation_prediction,
        "test": test_prediction,
    },
}
split_targets = {
    "validation": y_val,
    "test": y_test,
}
comparison_metrics = {
    model_name: {
        split_name: flattened_metrics(
            split_targets[split_name],
            prediction,
        )
        for split_name, prediction in split_predictions.items()
    }
    for model_name, split_predictions in comparison_predictions.items()
}
comparison_table = pd.DataFrame(
    [
        {
            "model": model_name,
            "validation_mae_db": metrics["validation"]["MAE"],
            "validation_rmse_db": metrics["validation"]["RMSE"],
            "test_mae_db": metrics["test"]["MAE"],
            "test_rmse_db": metrics["test"]["RMSE"],
        }
        for model_name, metrics in comparison_metrics.items()
    ]
)
display(comparison_table.style.hide(axis="index").format(precision=6))


deep_null_threshold_db = float(np.quantile(y_train, 0.99))
high_frequency_threshold_ghz = float(np.quantile(frequencies_ghz, 0.75))
high_frequency_mask = frequencies_ghz >= high_frequency_threshold_ghz


def benchmark_slice_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """
    Return MAE in the train-defined deep-null and upper-frequency regions.
    """
    absolute_error = np.abs(predictions - targets)
    return {
        "deep_null_mae_db": float(
            absolute_error[targets >= deep_null_threshold_db].mean()
        ),
        "high_frequency_mae_db": float(
            absolute_error[:, high_frequency_mask].mean()
        ),
    }


validation_slice_metrics = benchmark_slice_metrics(
    y_val,
    validation_prediction,
)
test_slice_metrics = benchmark_slice_metrics(y_test, test_prediction)
benchmark_diagnostics = {
    "deep_null_threshold_db": deep_null_threshold_db,
    "high_frequency_threshold_ghz": high_frequency_threshold_ghz,
    "val_deep_null_mae_db": validation_slice_metrics["deep_null_mae_db"],
    "test_deep_null_mae_db": test_slice_metrics["deep_null_mae_db"],
    "val_high_frequency_mae_db": (
        validation_slice_metrics["high_frequency_mae_db"]
    ),
    "test_high_frequency_mae_db": test_slice_metrics["high_frequency_mae_db"],
}
display(pd.Series(benchmark_diagnostics, name="NB05 benchmark diagnostics"))


def per_target_split_metrics() -> dict[str, dict[str, dict[str, float]]]:
    """
    Return benchmark-compatible NB05 validation and test metrics by target.
    """
    validation = per_target_metrics(
        y_val.reshape(-1, n_targets),
        validation_prediction.reshape(-1, n_targets),
        target_names,
    ).set_index("target")
    test = per_target_metrics(
        y_test.reshape(-1, n_targets),
        test_prediction.reshape(-1, n_targets),
        target_names,
    ).set_index("target")
    return {
        target_name: {
            "validation": validation.loc[target_name].to_dict(),
            "test": test.loc[target_name].to_dict(),
        }
        for target_name in target_names
    }


final_per_target_metrics = per_target_split_metrics()
display(
    pd.DataFrame(
        [
            {
                "target": target_name,
                "validation_mae_db": metrics["validation"]["MAE"],
                "validation_rmse_db": metrics["validation"]["RMSE"],
                "test_mae_db": metrics["test"]["MAE"],
                "test_rmse_db": metrics["test"]["RMSE"],
            }
            for target_name, metrics in final_per_target_metrics.items()
        ]
    )
    .style.hide(axis="index")
    .format(precision=6)
)

shape_metrics = {
    split_name: {
        "FirstDifferenceMAE": float(
            np.mean(
                np.abs(
                    np.diff(prediction, axis=1)
                    - np.diff(targets, axis=1)
                )
            )
        )
    }
    for split_name, targets, prediction in (
        ("validation", y_val, validation_prediction),
        ("test", y_test, test_prediction),
    )
}
prediction_ranges = [
    {
        "target": target_name,
        "test_true_min_db": float(y_test[:, :, index].min()),
        "test_true_max_db": float(y_test[:, :, index].max()),
        "test_predicted_min_db": float(test_prediction[:, :, index].min()),
        "test_predicted_max_db": float(test_prediction[:, :, index].max()),
    }
    for index, target_name in enumerate(target_names)
]
print("First-difference MAE:", shape_metrics)
display(pd.DataFrame(prediction_ranges).style.hide(axis="index").format(precision=3))

# %%
frequency_error_figures = {}
for split_name, targets in split_targets.items():
    frequency_frame = pd.DataFrame(
        {"FREQ_GHZ": np.tile(frequencies_ghz, len(targets))}
    )
    figure = plot_model_mae_comparison_by_frequency(
        frequency_frame,
        targets.reshape(-1, n_targets),
        {
            model_name: predictions[split_name].reshape(-1, n_targets)
            for model_name, predictions in comparison_predictions.items()
        },
        target_names,
    )
    figure.axes[0].set_title(
        f"{split_name.title()} MAE Comparison by Frequency"
    )
    curve_runner.manager.save_figure(
        figure,
        f"final_{split_name}_error_by_frequency.png",
    )
    frequency_error_figures[split_name] = figure
    plt.show()


def plot_representative_test_curves(design_position: int) -> Figure:
    """
    Plot all target curves for one deterministic held-out design.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for target_index, axis in enumerate(axes.ravel()):
        axis.plot(
            frequencies_ghz,
            y_test[design_position, :, target_index],
            color="black",
            linewidth=2,
            label="Touchstone target",
        )
        for model_name, predictions in comparison_predictions.items():
            axis.plot(
                frequencies_ghz,
                predictions["test"][design_position, :, target_index],
                label=model_name,
                alpha=0.85,
            )
        axis.set_title(target_names[target_index])
        axis.grid(True, alpha=0.3)
    for axis in axes[-1, :]:
        axis.set_xlabel("Frequency (GHz)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Insertion loss (dB)")
    axes[0, -1].legend(fontsize="small")
    fig.suptitle(
        "Held-out design "
        f"{test_set.simulation_indices[design_position]} — NB04 versus NB05"
    )
    fig.tight_layout()
    return fig


test_plot_position = int(
    np.random.default_rng(random_seed).integers(len(X_test))
)
fig_test_curves = plot_representative_test_curves(test_plot_position)
curve_runner.manager.save_figure(
    fig_test_curves,
    "final_test_design_curves.png",
)
plt.show()

# %% [markdown]
# ## 9. Persist, Reload, and Promote
#
# Promotion occurs only after the saved wrapper reproduces its validation and
# test predictions.

# %%
training_history = curve_model.history
assert training_history is not None

extra_metrics = {
    "benchmark_diagnostics": benchmark_diagnostics,
    "per_target": final_per_target_metrics,
    "shape": shape_metrics,
    "prediction_range": prediction_ranges,
    "model_summary": {
        "parameter_count": curve_model.keras_model.count_params(),
        "selected_epoch": curve_model.selected_epoch_,
        "stopped_epoch": len(training_history.epoch),
    },
    "nb04_comparison": {
        **nb04_audit,
        "metrics": comparison_metrics,
    },
    "validation_selection": validation_selection.to_dict(
        orient="records"
    ),
}
artifact_paths = curve_runner.persist(
    data_interface=curve_data_interface,
    extra_metrics=extra_metrics,
    metric_units={
        "MAE": "dB",
        "RMSE": "dB",
        "FirstDifferenceMAE": "dB per 0.5 GHz frequency-grid step",
    },
    refresh_benchmarks=True,
)

reloaded_model = curve_runner.manager.load_model()
np.testing.assert_allclose(
    reloaded_model.predict(X_val),
    validation_prediction,
    rtol=1e-5,
    atol=1e-5,
)
np.testing.assert_allclose(
    reloaded_model.predict(X_test),
    test_prediction,
    rtol=1e-5,
    atol=1e-5,
)

promoted_entry = curve_runner.registry.promote(
    curve_model.name,
    curve_runner.manager.run_id,
)
selected_benchmark_paths = refresh_benchmarks(
    cfg.paths.benchmarks,
    curve_runner.registry,
    curve_model.name,
    selection="selected",
)
assert promoted_entry.run_id == curve_runner.manager.run_id

project_root = cfg.paths.outputs.parent
print(f"Final run: {curve_runner.manager.run_id}")
print("Completed steps:", curve_runner.completed_steps)
print("Reload verification passed; final run promoted.")
display(
    pd.Series(
        {
            name: relative_to_project_root(path, project_root=project_root)
            for name, path in artifact_paths.items()
        },
        name="artifact path",
    )
)
print(
    "Selected benchmarks:",
    [
        relative_to_project_root(path, project_root=project_root)
        for path in selected_benchmark_paths
    ],
)

# %% [markdown]
# The selected NB05 artifact contains the model, scalers, exact frequency grid,
# split identifiers, training history, metrics, nb04 audit, and diagnostic
# figures. The curve-aware decoder improves aggregate accuracy over nb04 but
# still underestimates extreme resonant-loss peaks.
