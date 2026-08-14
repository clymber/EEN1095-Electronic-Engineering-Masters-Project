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
Train and evaluate a physics-aware complete complex S-matrix surrogate.
"""

# %load_ext autoreload
# %autoreload 2
# %aimport -pathlib
# %aimport -numpy

# ruff: noqa: E402 -- Configure filtered notebook output before remaining imports.
import os

from sparam_surrogate.config import configure_stdio_relative_path

# Keep routine TensorFlow device and end-of-dataset messages out of stored cells.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

configure_stdio_relative_path()

# %%
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from sparam_surrogate.config import SurrogateConfig, relative_to_project_root
from sparam_surrogate.data import CurveDataset, TouchstoneLoader
from sparam_surrogate.models.full_smatrix import (
    FullSMatrixModel,
    band_limited_causality_residual,
    configured_insertion_loss_db,
    physics_diagnostics,
    real_imag_channels_to_smatrix,
)
from sparam_surrogate.outputs import (
    ModelRunRunner,
    refresh_benchmarks,
    regenerate_benchmarks,
)
from sparam_surrogate.utils.non_neural_modelling_utils import regression_metrics

# %% [markdown]
# # Physics-Aware Full Complex S-Matrix Surrogate
#
# This notebook builds NB06 entirely from the configured cleaned data. All model
# weights and scaling statistics are fitted during this execution. Run NB05 first
# so its selected benchmark is available for the evaluation-only comparison.
#
# Unlike NB05, which predicts six insertion-loss curves directly, NB06 predicts
# the complete frequency-dependent complex 12-by-12 S-matrix. It uses finite real
# and imaginary targets rather than ill-conditioned dB values, shares one RMS
# scale between both components of each complex entry, and guarantees reciprocity
# by predicting only the unique upper triangle.
#
# The residual MLP receives both global Fourier and localized Gaussian RBF
# frequency features. This combination represents broad trends and local
# frequency behavior without an upsampling decoder.
#
# ```text
# scaled design parameters ─┐
# Fourier + Gaussian RBFs ──┼─> residual MLP ─> 78 complex entries
#                           └──────────────────> reciprocal 12×12 S-matrix
# ```

# %% [markdown]
# ## 1. Full-S-Matrix Data Contract
#
# A consolidated `smatrix_real_imag_curve_dataset.npz` stores the unscaled
# train, validation, and test tensors with their frequency grid, simulation
# indices, split labels, and target names. `CurveDataset` rebuilds the cache when
# the cleaned split source or stored metadata is incompatible; otherwise it avoids
# re-reading all 7,030 Touchstone files.

# %%
cfg = SurrogateConfig.from_config()
curve_config = cfg.models.curve_neural
keras.utils.set_random_seed(cfg.project.seed)

smatrix_loader = TouchstoneLoader(
    mode="smatrix",
    config=cfg,
    representation="real_imag",
    cache_size=8,
)
train_set, val_set, test_set = CurveDataset.from_cleaned_splits_csv(
    cfg.preprocessing.cleaned_splits_csv,
    smatrix_loader,
    cache=True,
    cache_dir=cfg.paths.processed_data,
)
smatrix_loader.clear_cache()

feature_names = tuple(train_set.feature_columns)
target_names = tuple(train_set.target_names)
frequencies_ghz = train_set.frequencies_ghz.astype(np.float32)

# pylint: disable=invalid-name
X_train, y_train = train_set.features, train_set.targets
X_val, y_val = val_set.features, val_set.targets
X_test, y_test = test_set.features, test_set.targets
# pylint: enable=invalid-name

n_ports = cfg.dataset.nports
n_frequencies = len(frequencies_ghz)
n_targets = 2 * n_ports**2
configured_entry_indices = tuple(
    (receiver - 1) * n_ports + source - 1 for receiver, source in cfg.dataset.ports
)
configured_target_names = tuple(
    f"IL_S{receiver}_{source}_DB" for receiver, source in cfg.dataset.ports
)

for dataset in (train_set, val_set, test_set):
    np.testing.assert_allclose(dataset.frequencies_ghz, frequencies_ghz)
    assert dataset.targets.dtype == np.float32
    assert dataset.targets.shape[1:] == (n_frequencies, n_targets)
    assert tuple(dataset.target_names) == target_names

assert len(feature_names) == 10
assert (n_frequencies, n_ports) == (200, 12)
assert set(train_set.simulation_indices).isdisjoint(val_set.simulation_indices)
assert set(train_set.simulation_indices).isdisjoint(test_set.simulation_indices)
assert set(val_set.simulation_indices).isdisjoint(test_set.simulation_indices)

display(
    pd.DataFrame(
        {
            "split": [dataset.split_type for dataset in (train_set, val_set, test_set)],
            "designs": [len(dataset) for dataset in (train_set, val_set, test_set)],
            "target shape": [
                str(dataset.targets.shape) for dataset in (train_set, val_set, test_set)
            ],
            "cache": [
                dataset.cache_status for dataset in (train_set, val_set, test_set)
            ],
        }
    ).style.hide(axis="index")
)
print(f"Shared full-S-matrix cache: {train_set.cache_path}")

# %% [markdown]
# ## 2. Metrics Fixed Before Training
#
# Six-path insertion-loss metrics are uncapped. The deepest 1% region is defined
# once from the NB05 training targets, so validation and test labels cannot move
# the threshold. Reading the selected benchmark also keeps the NB05 comparison
# synchronized with the latest NB05 notebook run.

# %%
nb05_benchmark = (
    pd.read_csv(
        cfg.paths.benchmarks / "vector_insertion_loss_db_selected.csv"
    )
    .set_index("model_name")
    .loc["curve_neural"]
)
deep_null_threshold_db = float(nb05_benchmark["deep_null_threshold_db"])
high_frequency_threshold_ghz = float(
    nb05_benchmark["high_frequency_threshold_ghz"]
)


def path_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """
    Return aggregate six-path insertion-loss MAE and RMSE.
    """
    truth = configured_insertion_loss_db(
        targets,
        n_ports,
        configured_entry_indices,
    )
    estimate = configured_insertion_loss_db(
        predictions,
        n_ports,
        configured_entry_indices,
    )
    return regression_metrics(
        truth.reshape(-1, len(configured_entry_indices)),
        estimate.reshape(-1, len(configured_entry_indices)),
    )


def deep_null_mae(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """
    Return six-path MAE in the train-defined top-one-percent null region.
    """
    truth = configured_insertion_loss_db(
        targets,
        n_ports,
        configured_entry_indices,
    )
    estimate = configured_insertion_loss_db(
        predictions,
        n_ports,
        configured_entry_indices,
    )
    return float(np.mean(np.abs(estimate - truth)[truth >= deep_null_threshold_db]))


def high_frequency_path_mae(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """
    Return six-path MAE above the NB05 high-frequency threshold.
    """
    truth = configured_insertion_loss_db(
        targets,
        n_ports,
        configured_entry_indices,
    )
    estimate = configured_insertion_loss_db(
        predictions,
        n_ports,
        configured_entry_indices,
    )
    mask = frequencies_ghz >= high_frequency_threshold_ghz
    return float(np.mean(np.abs(estimate - truth)[:, mask]))


def per_path_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Return benchmark metrics for each configured insertion-loss path.
    """
    truth = configured_insertion_loss_db(
        targets,
        n_ports,
        configured_entry_indices,
    )
    estimate = configured_insertion_loss_db(
        predictions,
        n_ports,
        configured_entry_indices,
    )
    return {
        target_name: regression_metrics(
            truth[..., target_index].reshape(-1),
            estimate[..., target_index].reshape(-1),
        )
        for target_index, target_name in enumerate(configured_target_names)
    }


def path_peak_table(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Return true and predicted maximum insertion loss for each configured path.
    """
    truth = configured_insertion_loss_db(
        targets,
        n_ports,
        configured_entry_indices,
    )
    estimate = configured_insertion_loss_db(
        predictions,
        n_ports,
        configured_entry_indices,
    )
    return pd.DataFrame(
        {
            "path": configured_target_names,
            "true_max_db": np.max(truth, axis=(0, 1)),
            "predicted_max_db": np.max(estimate, axis=(0, 1)),
        }
    )


print(f"Training-defined deep-null threshold: {deep_null_threshold_db:.3f} dB")

# %% [markdown]
# ## 3. Frequency-Conditioned Reciprocal Model
#
# At every frequency, the model concatenates ten scaled design parameters with 41
# fixed frequency features: normalized frequency, four sine/cosine pairs, and 32
# evenly spaced Gaussian RBFs. Three width-128 residual blocks predict the real and
# imaginary components of 78 unique complex entries. Mirroring those entries gives
# the complete reciprocal 12-by-12 output.
#
# The loss combines normalized complex MSE with a stable off-diagonal
# log-magnitude term of weight 0.1. Validation complex NRMSE selects the restored
# epoch.

# %%
runner = ModelRunRunner(
    cfg,
    FullSMatrixModel(
        frequencies_ghz=frequencies_ghz,
        n_ports=n_ports,
        hidden_width=128,
        residual_blocks=3,
        fourier_order=curve_config.fourier_order,
        frequency_rbf_count=32,
        reciprocal=True,
        log_magnitude_weight=0.1,
        log_magnitude_floor=1e-6,
        batch_size=curve_config.batch_size,
        epochs=curve_config.epochs,
        prediction_batch_size=curve_config.batch_size,
        learning_rate=curve_config.learning_rate,
        gradient_clip_norm=curve_config.gradient_clip_norm,
        early_stopping_patience=curve_config.early_stopping_patience,
        reduce_lr_patience=curve_config.reduce_lr_patience,
        reduce_lr_factor=curve_config.reduce_lr_factor,
        min_learning_rate=curve_config.min_learning_rate,
        random_state=cfg.project.seed,
    ),
)
model = runner.train(X_train, y_train, X_val, y_val)

print("Final localized full-S-matrix model summary")
model.keras_model.summary()
training_figure = model.plot_training_history()
print(f"Restored epoch: {model.selected_epoch_}")

# %% [markdown]
# ## 4. Validation Evaluation and Curve Inspection

# %%
validation_complex = runner.validate(X_val, y_val)
validation_prediction = model.predict(X_val)
validation_paths = path_metrics(y_val, validation_prediction)
validation_deep_null = deep_null_mae(y_val, validation_prediction)
validation_high_frequency_mae = high_frequency_path_mae(
    y_val,
    validation_prediction,
)

display(
    pd.Series(
        {
            **validation_complex,
            "SixPathMAE_dB": validation_paths["MAE"],
            "SixPathRMSE_dB": validation_paths["RMSE"],
            "DeepNullMAE_dB": validation_deep_null,
        },
        name="final NB06 validation",
    )
)

true_validation_il = configured_insertion_loss_db(
    y_val,
    n_ports,
    configured_entry_indices,
)
predicted_validation_il = configured_insertion_loss_db(
    validation_prediction,
    n_ports,
    configured_entry_indices,
)
validation_absolute_error = np.abs(predicted_validation_il - true_validation_il)
validation_design_mae = validation_absolute_error.mean(axis=(1, 2))
representative_design = int(
    np.argmin(np.abs(validation_design_mae - np.median(validation_design_mae)))
)
deepest_design = int(
    np.unravel_index(
        np.nanargmax(true_validation_il),
        true_validation_il.shape,
    )[0]
)
other_designs = np.flatnonzero(np.arange(len(validation_design_mae)) != deepest_design)
worst_design = int(other_designs[np.argmax(validation_design_mae[other_designs])])


def plot_design_paths(design_index: int, label: str) -> plt.Figure:
    """
    Plot true and predicted insertion loss for all six configured paths.
    """
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for path_index, axis in enumerate(axes.ravel()):
        axis.plot(
            frequencies_ghz,
            true_validation_il[design_index, :, path_index],
            color="black",
            linewidth=1.5,
            label="truth",
        )
        axis.plot(
            frequencies_ghz,
            predicted_validation_il[design_index, :, path_index],
            color="tab:green",
            linestyle="--",
            label="NB06 RBF model",
        )
        axis.set(
            title=configured_target_names[path_index],
            xlabel="Frequency (GHz)",
            ylabel="Insertion loss (dB)",
        )
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend()
    figure.suptitle(
        f"{label}: validation simulation {val_set.simulation_indices[design_index]}"
    )
    figure.tight_layout()
    return figure


representative_figure = plot_design_paths(
    representative_design,
    "Representative validation error",
)
deepest_figure = plot_design_paths(
    deepest_design,
    "Design containing the deepest true null",
)
worst_figure = plot_design_paths(
    worst_design,
    "Largest aggregate validation error",
)

# %% [markdown]
# ## 5. Frozen Test Evaluation
#
# The architecture and restored epoch are fixed above using validation data. The
# test set is evaluated once below. NB05 is included as a six-path reference, but
# it predicts only those six dB curves rather than the complete complex matrix.

# %%
test_complex = runner.test(X_test, y_test)
test_prediction = model.predict(X_test)
test_paths = path_metrics(y_test, test_prediction)
test_deep_null = deep_null_mae(y_test, test_prediction)

true_test_il = configured_insertion_loss_db(
    y_test,
    n_ports,
    configured_entry_indices,
)
predicted_test_il = configured_insertion_loss_db(
    test_prediction,
    n_ports,
    configured_entry_indices,
)
high_frequency_mae = high_frequency_path_mae(y_test, test_prediction)

display(
    pd.DataFrame(
        [
            {
                "model": "NB05 six-path reference",
                "six_path_mae_db": nb05_benchmark["test_mae_db"],
                "six_path_rmse_db": nb05_benchmark["test_rmse_db"],
                "deep_null_mae_db": nb05_benchmark[
                    "test_deep_null_mae_db"
                ],
                "high_frequency_mae_db": nb05_benchmark[
                    "test_high_frequency_mae_db"
                ],
            },
            {
                "model": "NB06 full complex RBF",
                "six_path_mae_db": test_paths["MAE"],
                "six_path_rmse_db": test_paths["RMSE"],
                "deep_null_mae_db": test_deep_null,
                "high_frequency_mae_db": high_frequency_mae,
            },
        ]
    )
    .style.hide(axis="index")
    .format(precision=6)
)
display(pd.Series(test_complex, name="NB06 full-matrix complex test metrics"))
display(path_peak_table(y_test, test_prediction).style.hide(axis="index"))

frequency_mae = np.mean(
    np.abs(predicted_test_il - true_test_il),
    axis=(0, 2),
)
frequency_error_figure, frequency_error_axis = plt.subplots(figsize=(8, 4.5))
frequency_error_axis.plot(frequencies_ghz, frequency_mae, color="tab:green")
frequency_error_axis.set(
    title="NB06 Six-Path Test MAE by Frequency",
    xlabel="Frequency (GHz)",
    ylabel="MAE (dB)",
)
frequency_error_axis.grid(True, alpha=0.3)
frequency_error_figure.tight_layout()

# %% [markdown]
# ## 6. Physics Diagnostics
#
# Reciprocity is exact by construction. Passivity and a finite-band
# Hilbert-transform residual are diagnostic only; no passivity or causality loss
# is used. The causality value must not be interpreted as a full-band proof because
# the available grid excludes DC and frequencies above 100 GHz.

# %%
true_test_matrices = real_imag_channels_to_smatrix(y_test, n_ports)
predicted_test_matrices = real_imag_channels_to_smatrix(test_prediction, n_ports)
np.testing.assert_allclose(
    predicted_test_matrices,
    np.swapaxes(predicted_test_matrices, -1, -2),
    atol=1e-7,
)

truth_physics = physics_diagnostics(true_test_matrices)
prediction_physics = physics_diagnostics(predicted_test_matrices)
truth_physics["BandLimitedCausalityResidual"] = band_limited_causality_residual(
    true_test_matrices
)
prediction_physics["BandLimitedCausalityResidual"] = band_limited_causality_residual(
    predicted_test_matrices
)
display(
    pd.DataFrame(
        [
            {"matrix": "test truth", **truth_physics},
            {"matrix": "NB06 prediction", **prediction_physics},
        ]
    )
    .style.hide(axis="index")
    .format(precision=6)
)

# %% [markdown]
# ## 7. Final Result

# %%
success_criteria = {
    "validation MAE beats NB05": (
        validation_paths["MAE"] < nb05_benchmark["val_mae_db"]
    ),
    "test MAE beats NB05": test_paths["MAE"] < nb05_benchmark["test_mae_db"],
    "test deep-null MAE beats NB05": (
        test_deep_null < nb05_benchmark["test_deep_null_mae_db"]
    ),
    "high-frequency test MAE beats NB05": (
        high_frequency_mae < nb05_benchmark["test_high_frequency_mae_db"]
    ),
    "predicted reciprocity is numerical zero": (
        prediction_physics["ReciprocityResidual"] < 1e-7
    ),
}
display(pd.Series(success_criteria, name="passed").to_frame())
print(
    "Final validation MAE / deep-null MAE / complex NRMSE: "
    f"{validation_paths['MAE']:.4f} / "
    f"{validation_deep_null:.4f} dB / "
    f"{validation_complex['ComplexNRMSE']:.6f}"
)
print(
    "Final test MAE / deep-null MAE / high-frequency MAE: "
    f"{test_paths['MAE']:.4f} / "
    f"{test_deep_null:.4f} / "
    f"{high_frequency_mae:.4f} dB"
)

# %% [markdown]
# ## 8. Persist, Reload, Promote, and Benchmark
#
# The persisted model retains its native full real/imaginary S-matrix contract.
# A nested benchmark contract records the six derived insertion-loss paths, so
# NB06 can join the existing vector, S7_1, and per-target benchmark tables without
# mislabelling the model output representation.

# %%
validation_per_path = per_path_metrics(y_val, validation_prediction)
test_per_path = per_path_metrics(y_test, test_prediction)
benchmark_per_target = {
    target_name: {
        "validation": validation_per_path[target_name],
        "test": test_per_path[target_name],
    }
    for target_name in configured_target_names
}
benchmark_diagnostics = {
    "deep_null_threshold_db": deep_null_threshold_db,
    "high_frequency_threshold_ghz": high_frequency_threshold_ghz,
    "val_deep_null_mae_db": validation_deep_null,
    "test_deep_null_mae_db": test_deep_null,
    "val_high_frequency_mae_db": validation_high_frequency_mae,
    "test_high_frequency_mae_db": high_frequency_mae,
}
full_smatrix_data_interface = {
    "dataset_name": cfg.dataset.name,
    "input_features": feature_names,
    "target_names": target_names,
    "target_scope": "full_smatrix_curve",
    "target_units": "dimensionless",
    "target_representation": "real_imag",
    "input_shape": [len(feature_names)],
    "output_shape": [n_frequencies, n_targets],
    "frequency_units": "GHz",
    "frequencies_ghz": frequencies_ghz,
    "n_ports": n_ports,
    "configured_insertion_loss_targets": configured_target_names,
    "split_identifiers": {
        "train": train_set.simulation_indices,
        "validation": val_set.simulation_indices,
        "test": test_set.simulation_indices,
    },
}
extra_metrics = {
    "benchmark": {
        "target_names": configured_target_names,
        "target_scope": "vector",
        "target_units": "dB",
        "target_representation": "insertion_loss_db",
        "validation": validation_paths,
        "test": test_paths,
        "per_target": benchmark_per_target,
        "benchmark_diagnostics": benchmark_diagnostics,
    },
    "physics": {
        "test_truth": truth_physics,
        "test_prediction": prediction_physics,
    },
    "model_summary": {
        "parameter_count": model.keras_model.count_params(),
        "selected_epoch": model.selected_epoch_,
    },
    "success_criteria": success_criteria,
}

diagnostic_figure_paths = {
    "representative_validation_figure": runner.manager.save_figure(
        representative_figure,
        "representative_validation_paths.png",
    ),
    "deepest_validation_figure": runner.manager.save_figure(
        deepest_figure,
        "deepest_validation_paths.png",
    ),
    "worst_validation_figure": runner.manager.save_figure(
        worst_figure,
        "worst_validation_paths.png",
    ),
    "test_frequency_error_figure": runner.manager.save_figure(
        frequency_error_figure,
        "test_mae_by_frequency.png",
    ),
}
artifact_paths = runner.persist(
    data_interface=full_smatrix_data_interface,
    extra_metrics=extra_metrics,
    metric_units={
        "ComplexMAE": "dimensionless complex magnitude",
        "ComplexNRMSE": "dimensionless",
        "MAE": "dB",
        "RMSE": "dB",
    },
    refresh_benchmarks=True,
)
artifact_paths.update(diagnostic_figure_paths)

reloaded_model = runner.manager.load_model()
np.testing.assert_allclose(
    reloaded_model.predict(X_val[:2]),
    validation_prediction[:2],
    rtol=1e-5,
    atol=1e-5,
)
np.testing.assert_allclose(
    reloaded_model.predict(X_test[:2]),
    test_prediction[:2],
    rtol=1e-5,
    atol=1e-5,
)

promoted_entry = runner.registry.promote(model.name, runner.manager.run_id)
latest_benchmark_paths = refresh_benchmarks(
    cfg.paths.benchmarks,
    runner.registry,
    model.name,
    selection="latest",
)
selected_benchmark_paths = regenerate_benchmarks(
    cfg.paths.benchmarks,
    runner.registry,
    selections=("selected",),
)
expected_benchmark_names = {
    "vector_insertion_loss_db_latest.csv",
    "s7_1_insertion_loss_db_latest.csv",
    "per_target_insertion_loss_db_latest.csv",
    "vector_insertion_loss_db_selected.csv",
    "s7_1_insertion_loss_db_selected.csv",
    "per_target_insertion_loss_db_selected.csv",
}
written_benchmark_names = {
    path.name
    for path in [*latest_benchmark_paths, *selected_benchmark_paths]
}
assert expected_benchmark_names == written_benchmark_names
assert promoted_entry.run_id == runner.manager.run_id

project_root = cfg.paths.outputs.parent
print(f"Final run: {runner.manager.run_id}")
print("Completed steps:", runner.completed_steps)
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
    "Written benchmarks:",
    sorted(
        relative_to_project_root(path, project_root=project_root)
        for path in [*latest_benchmark_paths, *selected_benchmark_paths]
    ),
)

# %% [markdown]
# NB06 provides a reciprocal, passive test prediction for the complete complex
# S-matrix. Its six-path high-frequency MAE is slightly lower than NB05, but its
# aggregate and deep-null errors are higher. The largest remaining limitation is
# the underprediction of extremely weak transmission near sharp nulls.
