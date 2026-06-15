"""
Utilities for non-neural S-parameter baseline modelling notebooks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error

RIDGE_ALPHA_GRID = (0.001, 0.005, 0.01, 0.1, 1.0, 10.0)
SCATTER_MAX_POINTS = 50_000
DEFAULT_RANDOM_SEED = 42


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE and RMSE for scalar or vector predictions."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def per_target_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, names: tuple[str, ...],
) -> pd.DataFrame:
    """Return MAE and RMSE for each vector target column."""
    rows = []
    for column_index, target_name in enumerate(names):
        metrics = regression_metrics(y_true[:, column_index], y_pred[:, column_index])
        rows.append({"target": target_name, **metrics})
    return pd.DataFrame(rows)


def fit_ridge_with_validation(
    X_train: np.ndarray,  # pylint: disable=invalid-name
    y_train: np.ndarray,
    X_val: np.ndarray,  # pylint: disable=invalid-name
    y_val: np.ndarray,
    alphas: tuple[float, ...],
    model_factory: Callable[[float], Any],
) -> tuple[Any, pd.DataFrame]:
    """Fit alpha-indexed candidates and select the lowest-validation-MAE model."""
    rows = []
    best_model: Any | None = None
    best_mae = np.inf

    for alpha in alphas:
        model = model_factory(alpha)
        model.fit(X_train, y_train)
        y_val_pred = cast(np.ndarray, model.predict(X_val))
        metrics = regression_metrics(y_val, y_val_pred)
        rows.append({"alpha": alpha, **metrics})

        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            best_model = model

    if best_model is None:
        raise RuntimeError("No Ridge model was fitted.")
    return best_model, pd.DataFrame(rows)


def sample_positions(
    n_rows: int,
    *,
    max_points: int = SCATTER_MAX_POINTS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> np.ndarray:
    """Return deterministic row positions for large scatter-style plots."""
    if n_rows <= max_points:
        return np.arange(n_rows)
    rng = np.random.default_rng(random_seed)
    return np.sort(rng.choice(n_rows, size=max_points, replace=False))


def ordered_design_positions(
    dataframe: pd.DataFrame,
    simulation_index: int | None = None,
) -> tuple[int, np.ndarray]:
    """Return row positions for one design, sorted by frequency."""
    if simulation_index is None:
        simulation_index = int(dataframe["SIMU_INDEX"].iloc[0])
    design_mask = dataframe["SIMU_INDEX"].to_numpy(dtype=int) == simulation_index
    positions = np.flatnonzero(design_mask)
    frequencies = dataframe["FREQ_GHZ"].to_numpy(dtype=float)
    ordered = positions[np.argsort(frequencies[positions])]
    return simulation_index, ordered


def add_diagonal_reference(ax: Axes, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Add an ideal y=x reference line to a scatter plot."""
    lower = float(np.min([np.min(y_true), np.min(y_pred)]))
    upper = float(np.max([np.max(y_true), np.max(y_pred)]))
    if np.isclose(lower, upper):
        lower -= 1.0
        upper += 1.0
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1.0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)


def plot_scalar_curve_for_design(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    simulation_index: int | None = None,
) -> Figure:
    """Plot one true and predicted IL curve for a held-out design."""
    selected_index, positions = ordered_design_positions(dataframe, simulation_index)
    frequency = dataframe["FREQ_GHZ"].to_numpy(dtype=float)[positions]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frequency, y_true[positions], label="True", linewidth=2.0)
    ax.plot(frequency, y_pred[positions], label="Predicted", linewidth=2.0)
    ax.set_title(f"Scalar Ridge IL Curve: {target_name}, SIMU_INDEX={selected_index}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Insertion loss (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_scalar_true_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
) -> Figure:
    """Plot scalar true-vs-predicted scatter with an ideal diagonal."""
    positions = sample_positions(len(y_true))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(
        y_true[positions],
        y_pred[positions],
        s=5,
        alpha=0.15,
        rasterized=True,
    )
    add_diagonal_reference(ax, y_true[positions], y_pred[positions])
    ax.set_title(f"Predicted vs True: {target_name}")
    ax.set_xlabel("True IL (dB)")
    ax.set_ylabel("Predicted IL (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_scalar_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
) -> Figure:
    """Plot scalar residual distribution."""
    residual = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residual, bins=80, color="tab:blue", alpha=0.8)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_title(f"Residual Histogram: {target_name}")
    ax.set_xlabel("Prediction error (dB)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_scalar_residual_vs_frequency(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
) -> Figure:
    """Plot scalar residuals against frequency."""
    residual = y_pred - y_true
    positions = sample_positions(len(y_true))
    frequency = dataframe["FREQ_GHZ"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(
        frequency[positions],
        residual[positions],
        s=5,
        alpha=0.12,
        rasterized=True,
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(f"Residual vs Frequency: {target_name}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Prediction error (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def scalar_mae_by_frequency(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Return scalar MAE grouped by frequency."""
    errors = pd.DataFrame(
        {
            "FREQ_GHZ": dataframe["FREQ_GHZ"].to_numpy(dtype=float),
            "ABS_ERROR": np.abs(y_pred - y_true),
        }
    )
    return (
        errors.groupby("FREQ_GHZ", sort=True)["ABS_ERROR"]
        .mean()
        .reset_index(name="MAE")
    )


def plot_scalar_mae_by_frequency(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
) -> Figure:
    """Plot scalar MAE as a function of frequency."""
    mae_curve = scalar_mae_by_frequency(dataframe, y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(mae_curve["FREQ_GHZ"], mae_curve["MAE"], linewidth=2.0)
    ax.set_title(f"MAE by Frequency: {target_name}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("MAE (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_vector_curves_for_design(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: tuple[str, ...],
    simulation_index: int | None = None,
) -> Figure:
    """Plot six true and predicted IL curves for one held-out design."""
    selected_index, positions = ordered_design_positions(dataframe, simulation_index)
    frequency = dataframe["FREQ_GHZ"].to_numpy(dtype=float)[positions]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for column_index, ax in enumerate(np.asarray(axes).ravel()):
        ax.plot(
            frequency,
            y_true[positions, column_index],
            label="True",
            linewidth=1.8,
        )
        ax.plot(
            frequency,
            y_pred[positions, column_index],
            label="Predicted",
            linewidth=1.8,
        )
        ax.set_title(names[column_index])
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("IL (dB)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle(f"Vector Ridge IL Curves, SIMU_INDEX={selected_index}", y=1.02)
    fig.tight_layout()
    return fig


def plot_vector_true_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: tuple[str, ...],
) -> Figure:
    """Plot true-vs-predicted scatter for each vector target."""
    positions = sample_positions(y_true.shape[0])
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for column_index, ax in enumerate(np.asarray(axes).ravel()):
        true_values = y_true[positions, column_index]
        pred_values = y_pred[positions, column_index]
        ax.scatter(
            true_values,
            pred_values,
            s=4,
            alpha=0.12,
            rasterized=True,
        )
        add_diagonal_reference(ax, true_values, pred_values)
        ax.set_title(names[column_index])
        ax.set_xlabel("True IL (dB)")
        ax.set_ylabel("Predicted IL (dB)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Vector Ridge Predicted vs True", y=1.02)
    fig.tight_layout()
    return fig


def plot_vector_residual_histograms(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: tuple[str, ...],
) -> Figure:
    """Plot one residual histogram for each vector target."""
    residuals = y_pred - y_true
    lower, upper = np.percentile(residuals, [0.5, 99.5])
    bins = np.linspace(lower, upper, 80)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for column_index, ax in enumerate(np.asarray(axes).ravel()):
        ax.hist(
            residuals[:, column_index],
            bins=bins,
            color="tab:blue",
            alpha=0.8,
        )
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title(names[column_index])
        ax.set_xlabel("Prediction error (dB)")
        ax.set_ylabel("Count")
    fig.suptitle("Vector Ridge Residual Histograms", y=1.02)
    fig.tight_layout()
    return fig


def plot_vector_residual_vs_frequency(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: tuple[str, ...],
) -> Figure:
    """Plot residuals against frequency for each vector target."""
    residuals = y_pred - y_true
    frequency = dataframe["FREQ_GHZ"].to_numpy(dtype=float)
    positions = sample_positions(y_true.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for column_index, ax in enumerate(np.asarray(axes).ravel()):
        ax.scatter(
            frequency[positions],
            residuals[positions, column_index],
            s=4,
            alpha=0.1,
            rasterized=True,
        )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(names[column_index])
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Prediction error (dB)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Vector Ridge Residual vs Frequency", y=1.02)
    fig.tight_layout()
    return fig


def vector_mae_by_frequency(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: tuple[str, ...],
) -> pd.DataFrame:
    """Return per-target MAE grouped by frequency."""
    errors = pd.DataFrame(np.abs(y_pred - y_true), columns=names)
    errors["FREQ_GHZ"] = dataframe["FREQ_GHZ"].to_numpy(dtype=float)
    return errors.groupby("FREQ_GHZ", sort=True).mean()


def plot_vector_mae_by_frequency(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: tuple[str, ...],
) -> Figure:
    """Plot all vector-target MAE-by-frequency curves on one axis."""
    mae_curves = vector_mae_by_frequency(dataframe, y_true, y_pred, names)

    fig, ax = plt.subplots(figsize=(9, 5))
    for target_name in names:
        ax.plot(
            mae_curves.index.to_numpy(dtype=float),
            mae_curves[target_name],
            label=target_name,
            linewidth=1.8,
        )
    ax.set_title("Vector Ridge MAE by Frequency")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("MAE (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


__all__ = [
    "RIDGE_ALPHA_GRID",
    "SCATTER_MAX_POINTS",
    "add_diagonal_reference",
    "fit_ridge_with_validation",
    "ordered_design_positions",
    "per_target_metrics",
    "plot_scalar_curve_for_design",
    "plot_scalar_mae_by_frequency",
    "plot_scalar_residual_histogram",
    "plot_scalar_residual_vs_frequency",
    "plot_scalar_true_vs_predicted",
    "plot_vector_curves_for_design",
    "plot_vector_mae_by_frequency",
    "plot_vector_residual_histograms",
    "plot_vector_residual_vs_frequency",
    "plot_vector_true_vs_predicted",
    "regression_metrics",
    "sample_positions",
    "scalar_mae_by_frequency",
    "vector_mae_by_frequency",
]
