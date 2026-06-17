"""
Utilities for non-neural S-parameter baseline modelling notebooks.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    ax.set_title(f"Scalar Ridge Magnitude Distribution: {target_name}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Transmission Magnitude (dB)")
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
    ax.set_xlabel("True Transmission Magnitude (dB)")
    ax.set_ylabel("Predicted Transmission Magnitude (dB)")
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


def plot_shared_target_mae_comparison(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    target_name: str,
) -> Figure:
    """Plot MAE-by-frequency curves for several models on one scalar target."""
    if not predictions:
        raise ValueError("predictions must contain at least one model.")

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, y_pred in predictions.items():
        mae_curve = scalar_mae_by_frequency(dataframe, y_true, y_pred)
        ax.plot(
            mae_curve["FREQ_GHZ"].to_numpy(dtype=float),
            mae_curve["MAE"].to_numpy(dtype=float),
            label=model_name,
            linewidth=1.8,
        )

    ax.set_title(f"{target_name} MAE by Frequency")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("MAE (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_shared_target_prediction_bands(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    target_name: str,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> Figure:
    """Plot true and per-model predicted distribution bands for one target."""
    if not predictions:
        raise ValueError("predictions must contain at least one model.")

    summaries = {
        model_name: scalar_prediction_summary_by_frequency(
            dataframe,
            y_true,
            y_pred,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        for model_name, y_pred in predictions.items()
    }
    first_summary = next(iter(summaries.values()))
    frequency = first_summary["FREQ_GHZ"].to_numpy(dtype=float)
    lower_label = int(round(lower_quantile * 100))
    upper_label = int(round(upper_quantile * 100))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.fill_between(
        frequency,
        first_summary["TRUE_LOWER"].to_numpy(dtype=float),
        first_summary["TRUE_UPPER"].to_numpy(dtype=float),
        color="tab:blue",
        alpha=0.16,
        label=f"True {lower_label}th-{upper_label}th percentile",
    )
    ax.plot(
        frequency,
        first_summary["TRUE_MEDIAN"].to_numpy(dtype=float),
        color="tab:blue",
        linewidth=2.4,
        label="True median",
    )

    for model_index, (model_name, summary) in enumerate(summaries.items()):
        color = colors[(model_index + 1) % len(colors)]
        ax.fill_between(
            frequency,
            summary["PREDICTED_LOWER"].to_numpy(dtype=float),
            summary["PREDICTED_UPPER"].to_numpy(dtype=float),
            color=color,
            alpha=0.10,
            label=f"{model_name} predicted band",
        )
        ax.plot(
            frequency,
            summary["PREDICTED_MEDIAN"].to_numpy(dtype=float),
            color=color,
            linewidth=1.9,
            label=f"{model_name} predicted median",
        )

    ax.set_title(f"{target_name} Predicted Distribution Comparison")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S7_1_DB (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize="small")
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
    model_name: str = "Vector Ridge",
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
        ax.set_ylabel("Transmission Magnitude (dB)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle(f"{model_name} Transmission Magnitude Distributions", y=1.02)
    fig.tight_layout()
    return fig


def plot_vector_true_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: tuple[str, ...],
    model_name: str = "Vector Ridge",
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
        ax.set_xlabel("True Transmission Magnitude (dB)")
        ax.set_ylabel("Predicted Transmission Magnitude (dB)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{model_name} Predicted vs True", y=1.02)
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
    model_name: str = "Vector Ridge",
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
    ax.set_title(f"{model_name} MAE by Frequency")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("MAE (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_model_mae_comparison_by_frequency(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    names: tuple[str, ...],
) -> Figure:
    """Plot mean vector-target MAE by frequency for several models."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, y_pred in predictions.items():
        mae_curves = vector_mae_by_frequency(dataframe, y_true, y_pred, names)
        ax.plot(
            mae_curves.index.to_numpy(dtype=float),
            mae_curves.mean(axis=1).to_numpy(dtype=float),
            label=model_name,
            linewidth=1.8,
        )

    ax.set_title("Model MAE Comparison by Frequency")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Mean MAE across targets (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


__all__ = [
    "SCATTER_MAX_POINTS",
    "add_diagonal_reference",
    "per_target_metrics",
    "plot_model_mae_comparison_by_frequency",
    "plot_scalar_mae_by_frequency",
    "plot_scalar_prediction_band_by_frequency",
    "plot_scalar_true_vs_predicted",
    "plot_shared_target_mae_comparison",
    "plot_shared_target_prediction_bands",
    "plot_vector_mae_by_frequency",
    "plot_vector_prediction_bands_by_frequency",
    "plot_vector_true_vs_predicted",
    "regression_metrics",
    "sample_positions",
    "scalar_mae_by_frequency",
    "scalar_prediction_summary_by_frequency",
    "vector_mae_by_frequency",
    "vector_prediction_summary_by_frequency",
]
