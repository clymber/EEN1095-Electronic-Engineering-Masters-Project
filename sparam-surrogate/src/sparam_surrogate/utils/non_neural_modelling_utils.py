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


def scalar_prediction_summary_by_frequency(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> pd.DataFrame:
    """Return scalar true/predicted median and quantile bands by frequency."""
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Expected 0 <= lower_quantile < upper_quantile <= 1.")

    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    pred_values = np.asarray(y_pred, dtype=float).reshape(-1)
    frequencies = dataframe["FREQ_GHZ"].to_numpy(dtype=float)
    if len(frequencies) != len(true_values) or len(frequencies) != len(pred_values):
        raise ValueError("dataframe, y_true, and y_pred must have the same length.")

    values = pd.DataFrame(
        {
            "FREQ_GHZ": frequencies,
            "TRUE": true_values,
            "PREDICTED": pred_values,
        }
    )
    grouped = values.groupby("FREQ_GHZ", sort=True)
    return pd.DataFrame(
        {
            "FREQ_GHZ": grouped.size().index.to_numpy(dtype=float),
            "TRUE_LOWER": grouped["TRUE"].quantile(lower_quantile).to_numpy(),
            "TRUE_MEDIAN": grouped["TRUE"].median().to_numpy(),
            "TRUE_UPPER": grouped["TRUE"].quantile(upper_quantile).to_numpy(),
            "PREDICTED_LOWER": grouped["PREDICTED"].quantile(lower_quantile).to_numpy(),
            "PREDICTED_MEDIAN": grouped["PREDICTED"].median().to_numpy(),
            "PREDICTED_UPPER": grouped["PREDICTED"].quantile(upper_quantile).to_numpy(),
            "COUNT": grouped.size().to_numpy(dtype=int),
        }
    )


def plot_scalar_prediction_band_by_frequency(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> Figure:
    """Plot true and predicted scalar IL distributions across designs."""
    summary = scalar_prediction_summary_by_frequency(
        dataframe,
        y_true,
        y_pred,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    lower_label = int(round(lower_quantile * 100))
    upper_label = int(round(upper_quantile * 100))
    frequency = summary["FREQ_GHZ"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.fill_between(
        frequency,
        summary["TRUE_LOWER"].to_numpy(dtype=float),
        summary["TRUE_UPPER"].to_numpy(dtype=float),
        color="tab:blue",
        alpha=0.18,
        label=f"True {lower_label}th-{upper_label}th percentile",
    )
    ax.plot(
        frequency,
        summary["TRUE_MEDIAN"].to_numpy(dtype=float),
        color="tab:blue",
        linewidth=2.0,
        label="True median",
    )
    ax.fill_between(
        frequency,
        summary["PREDICTED_LOWER"].to_numpy(dtype=float),
        summary["PREDICTED_UPPER"].to_numpy(dtype=float),
        color="tab:orange",
        alpha=0.20,
        label=f"Predicted {lower_label}th-{upper_label}th percentile",
    )
    ax.plot(
        frequency,
        summary["PREDICTED_MEDIAN"].to_numpy(dtype=float),
        color="tab:orange",
        linewidth=2.0,
        label="Predicted median",
    )
    ax.set_title(f"Scalar Ridge IL Distribution: {target_name}")
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


def vector_prediction_summary_by_frequency(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: tuple[str, ...],
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> pd.DataFrame:
    """Return vector true/predicted median and quantile bands by frequency."""
    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    if true_values.shape != pred_values.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if true_values.ndim != 2:
        raise ValueError("Vector prediction summaries require two-dimensional arrays.")
    if true_values.shape[1] != len(names):
        raise ValueError("names must match the number of vector target columns.")

    summaries = []
    for column_index, target_name in enumerate(names):
        summary = scalar_prediction_summary_by_frequency(
            dataframe,
            true_values[:, column_index],
            pred_values[:, column_index],
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        summary.insert(1, "TARGET", target_name)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def plot_vector_prediction_bands_by_frequency(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: tuple[str, ...],
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> Figure:
    """Plot vector IL distributions across designs for each target."""
    summary = vector_prediction_summary_by_frequency(
        dataframe,
        y_true,
        y_pred,
        names,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    lower_label = int(round(lower_quantile * 100))
    upper_label = int(round(upper_quantile * 100))
    n_targets = len(names)
    n_cols = min(3, n_targets)
    n_rows = int(np.ceil(n_targets / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 3.4 * n_rows),
        sharex=True,
        squeeze=False,
    )
    for column_index, ax in enumerate(axes.ravel()):
        if column_index >= n_targets:
            ax.set_visible(False)
            continue

        target_name = names[column_index]
        target_summary = summary.loc[summary["TARGET"] == target_name]
        frequency = target_summary["FREQ_GHZ"].to_numpy(dtype=float)
        ax.fill_between(
            frequency,
            target_summary["TRUE_LOWER"].to_numpy(dtype=float),
            target_summary["TRUE_UPPER"].to_numpy(dtype=float),
            color="tab:blue",
            alpha=0.18,
            label=f"True {lower_label}th-{upper_label}th percentile",
        )
        ax.plot(
            frequency,
            target_summary["TRUE_MEDIAN"].to_numpy(dtype=float),
            color="tab:blue",
            linewidth=1.8,
            label="True median",
        )
        ax.fill_between(
            frequency,
            target_summary["PREDICTED_LOWER"].to_numpy(dtype=float),
            target_summary["PREDICTED_UPPER"].to_numpy(dtype=float),
            color="tab:orange",
            alpha=0.20,
            label=f"Predicted {lower_label}th-{upper_label}th percentile",
        )
        ax.plot(
            frequency,
            target_summary["PREDICTED_MEDIAN"].to_numpy(dtype=float),
            color="tab:orange",
            linewidth=1.8,
            label="Predicted median",
        )
        ax.set_title(target_name)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("IL (dB)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Vector Ridge IL Distributions", y=1.02)
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
    "per_target_metrics",
    "plot_scalar_mae_by_frequency",
    "plot_scalar_prediction_band_by_frequency",
    "plot_scalar_residual_histogram",
    "plot_scalar_residual_vs_frequency",
    "plot_scalar_true_vs_predicted",
    "plot_vector_mae_by_frequency",
    "plot_vector_prediction_bands_by_frequency",
    "plot_vector_residual_histograms",
    "plot_vector_residual_vs_frequency",
    "plot_vector_true_vs_predicted",
    "regression_metrics",
    "sample_positions",
    "scalar_mae_by_frequency",
    "scalar_prediction_summary_by_frequency",
    "vector_mae_by_frequency",
    "vector_prediction_summary_by_frequency",
]
