"""
Shared prediction-curve plotting utilities for surrogate models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    import pandas as pd

    from sparam_surrogate.data import PointwiseDataset, TouchstoneLoader
    from sparam_surrogate.models.base import SparamModel

SIMULATION_COLUMN = "SIMU_INDEX"
FREQUENCY_COLUMN = "FREQ_GHZ"


def plot_design_prediction_curves(
    model: SparamModel,
    dataset: PointwiseDataset,
    target_loader: TouchstoneLoader,
    design_ids: Sequence[int | float] | np.ndarray,
) -> Figure:
    """
    Plot true and predicted curves for selected held-out designs.

    The subplot grid is arranged as targets by designs. This keeps scalar models
    compact while still showing every output from vector or neural models.
    """
    feature_values = np.asarray(dataset.features, dtype=float)
    metadata = dataset.row_metadata
    plot_frame = metadata.loc[:, [SIMULATION_COLUMN, FREQUENCY_COLUMN]].copy()
    plot_frame["_ROW_POSITION"] = np.arange(len(plot_frame))
    selected_design_ids = np.asarray(design_ids)
    n_selected_designs = len(selected_design_ids)
    target_names = tuple(target_loader.target_names)

    design_curves = []
    simulation_values = np.asarray(plot_frame[SIMULATION_COLUMN])
    for simulation_index in selected_design_ids:
        design_rows = plot_frame.loc[
            simulation_values == simulation_index
        ].sort_values(FREQUENCY_COLUMN)
        row_positions = np.asarray(design_rows["_ROW_POSITION"], dtype=int)
        frequency = np.asarray(design_rows[FREQUENCY_COLUMN], dtype=float)

        y_true_design = _load_targets_for_rows(
            target_loader,
            feature_values,
            metadata,
            row_positions,
        )
        y_pred_design = _as_2d_array(
            np.asarray(model.predict(feature_values[row_positions]), dtype=float),
            "model predictions",
        )
        if y_pred_design.shape[1] != y_true_design.shape[1]:
            raise ValueError(
                "model prediction columns must match Touchstone target columns."
            )
        design_curves.append(
            (simulation_index, frequency, y_true_design, y_pred_design)
        )

    n_targets = design_curves[0][2].shape[1]

    fig, axes = plt.subplots(
        n_targets,
        n_selected_designs,
        figsize=(3.2 * n_selected_designs, 2.5 * n_targets),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for design_column, curve in enumerate(design_curves):
        simulation_index, frequency, y_true_design, y_pred_design = curve
        for target_row in range(n_targets):
            ax = axes[target_row, design_column]
            ax.plot(
                frequency,
                y_true_design[:, target_row],
                color="tab:blue",
                linewidth=1.5,
                label="True",
            )
            ax.plot(
                frequency,
                y_pred_design[:, target_row],
                color="tab:orange",
                linewidth=1.5,
                linestyle="--",
                label="Predicted",
            )
            if target_row == 0:
                ax.set_title(f"{SIMULATION_COLUMN} {simulation_index}")
            if design_column == 0:
                ax.set_ylabel(f"{target_names[target_row]}")
            if target_row == n_targets - 1:
                ax.set_xlabel("Frequency (GHz)")
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize="small")
    fig.suptitle(f"{model.model_name()}: Selected Test Design Curves", y=1.01)
    fig.tight_layout()
    return fig


def plot_design_model_comparison_curves(
    models: Mapping[str, SparamModel],
    dataset: PointwiseDataset,
    target_loader: TouchstoneLoader,
    design_ids: Sequence[int | float] | np.ndarray,
) -> Figure:
    """
    Plot true and multiple model-predicted curves for held-out designs.
    """
    if not models:
        raise ValueError("models must contain at least one fitted model.")

    feature_values = np.asarray(dataset.features, dtype=float)
    metadata = dataset.row_metadata
    plot_frame = metadata.loc[:, [SIMULATION_COLUMN, FREQUENCY_COLUMN]].copy()
    plot_frame["_ROW_POSITION"] = np.arange(len(plot_frame))
    selected_design_ids = np.asarray(design_ids)
    n_selected_designs = len(selected_design_ids)
    target_names = tuple(target_loader.target_names)

    design_curves = []
    simulation_values = np.asarray(plot_frame[SIMULATION_COLUMN])
    for simulation_index in selected_design_ids:
        design_rows = plot_frame.loc[
            simulation_values == simulation_index
        ].sort_values(FREQUENCY_COLUMN)
        row_positions = np.asarray(design_rows["_ROW_POSITION"], dtype=int)
        frequency = np.asarray(design_rows[FREQUENCY_COLUMN], dtype=float)
        y_true_design = _load_targets_for_rows(
            target_loader,
            feature_values,
            metadata,
            row_positions,
        )
        model_predictions = {
            model_name: _as_2d_array(
                np.asarray(model.predict(feature_values[row_positions]), dtype=float),
                f"{model_name} predictions",
            )
            for model_name, model in models.items()
        }
        for model_name, y_pred_design in model_predictions.items():
            if y_pred_design.shape[1] != y_true_design.shape[1]:
                raise ValueError(
                    f"{model_name} prediction columns must match target columns."
                )
        design_curves.append(
            (simulation_index, frequency, y_true_design, model_predictions)
        )

    n_targets = design_curves[0][2].shape[1]
    fig, axes = plt.subplots(
        n_targets,
        n_selected_designs,
        figsize=(3.4 * n_selected_designs, 2.5 * n_targets),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for design_column, curve in enumerate(design_curves):
        simulation_index, frequency, y_true_design, model_predictions = curve
        for target_row in range(n_targets):
            ax = axes[target_row, design_column]
            ax.plot(
                frequency,
                y_true_design[:, target_row],
                color="black",
                linewidth=1.6,
                label="True",
            )
            for model_name, y_pred_design in model_predictions.items():
                ax.plot(
                    frequency,
                    y_pred_design[:, target_row],
                    linewidth=1.4,
                    linestyle="--",
                    label=model_name,
                )
            if target_row == 0:
                ax.set_title(f"{SIMULATION_COLUMN} {simulation_index}")
            if design_column == 0:
                ax.set_ylabel(f"{target_names[target_row]}")
            if target_row == n_targets - 1:
                ax.set_xlabel("Frequency (GHz)")
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize="small")
    fig.suptitle("Selected Test Design Curve Comparison", y=1.01)
    fig.tight_layout()
    return fig


def _load_targets_for_rows(
    target_loader: TouchstoneLoader,
    feature_values: np.ndarray,
    metadata: pd.DataFrame,
    row_positions: np.ndarray,
) -> np.ndarray:
    """
    Load Touchstone targets for the row positions being plotted.
    """
    metadata_rows = metadata.iloc[row_positions].to_dict("records")
    targets = np.stack(
        [
            np.asarray(
                target_loader(feature_values[row_position], row_metadata),
                dtype=float,
            )
            for row_position, row_metadata in zip(
                row_positions,
                metadata_rows,
                strict=True,
            )
        ]
    )
    return _as_2d_array(targets, "targets")


def _as_2d_array(values: np.ndarray, label: str) -> np.ndarray:
    """
    Return one-dimensional arrays as single-column two-dimensional arrays.
    """
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.ndim == 2:
        return values
    raise ValueError(f"{label} must be one- or two-dimensional.")


__all__ = [
    "plot_design_model_comparison_curves",
    "plot_design_prediction_curves",
]
