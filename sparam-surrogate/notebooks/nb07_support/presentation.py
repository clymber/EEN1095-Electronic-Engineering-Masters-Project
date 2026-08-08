"""
Table and figure presentation helpers for the NB07 evaluation notebook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def format_binary_size(size_bytes: int) -> str:
    """
    Format a byte count with an appropriate binary unit.
    """
    if size_bytes >= 2**30:
        return f"{size_bytes / 2**30:.2f} GiB"
    if size_bytes >= 2**20:
        return f"{size_bytes / 2**20:.2f} MiB"
    if size_bytes >= 2**10:
        return f"{size_bytes / 2**10:.2f} KiB"
    return f"{size_bytes} B"


def display_selected_model_overview(provenance: pd.DataFrame) -> None:
    """
    Display selected run identity, output scope, and saved-artifact size.
    """
    table = provenance.loc[
        :,
        [
            "model_name",
            "run_id",
            "target_scope",
            "artifact_size",
        ],
    ].rename(
        columns={
            "model_name": "Model",
            "run_id": "Selected run",
            "target_scope": "Output scope",
            "artifact_size": "Artifact size",
        }
    )
    _display_table(table)


def display_split_summary(
    canonical_ids: Mapping[str, np.ndarray],
    point_features: Mapping[str, np.ndarray],
    *,
    split_names: Sequence[str],
) -> None:
    """
    Display design and point-wise row counts for the requested data splits.
    """
    table = pd.DataFrame(
        {
            "split": split_names,
            "designs": [len(canonical_ids[name]) for name in split_names],
            "point-wise rows": [len(point_features[name]) for name in split_names],
        }
    )
    _display_table(table)


def display_reference_metrics(reference_table: pd.DataFrame) -> None:
    """
    Display validation and test metrics for the train-only references.
    """
    table = reference_table.rename(
        columns={
            "reference": "Reference",
            "split": "Split",
            "MAE_dB": "MAE (dB)",
            "RMSE_dB": "RMSE (dB)",
        }
    )
    _display_table(
        table,
        formatter={"MAE (dB)": "{:.4f}", "RMSE (dB)": "{:.4f}"},
    )


def display_key_value_summary(
    values: Mapping[str, object] | pd.Series,
    *,
    key_label: str,
    value_label: str,
    precision: int,
) -> None:
    """
    Display one set of named values as a narrow key-value table.
    """
    table = pd.DataFrame([dict(values)]).T.rename_axis(key_label).reset_index()
    table = table.rename(columns={0: value_label})
    _display_table(table, precision=precision)


def display_model_metrics(
    result_rows: Sequence[Mapping[str, object]],
    *,
    model_name: str,
    metric_names: Sequence[str],
    metric_labels: Mapping[str, str],
) -> None:
    """
    Display one model's metrics in a narrow split-by-column table.
    """
    model_rows = pd.DataFrame(result_rows).query("model_name == @model_name")
    split_order = [
        split
        for split in ("train", "validation", "test")
        if split in set(model_rows["split"])
    ]
    metric_table = (
        model_rows.set_index("split")
        .loc[split_order, list(metric_names)]
        .T.rename(index=metric_labels)
        .rename(
            columns={
                "train": "Train",
                "validation": "Validation",
                "test": "Test",
            }
        )
    )
    metric_table.index.name = "Metric"
    _display_table(metric_table.reset_index(), precision=4)


def display_transition_result(
    row: Mapping[str, object],
    *,
    predecessor_label: str,
    current_label: str,
    practical_margin_db: float,
) -> None:
    """
    Display one precomputed transition estimate and its interpretation.
    """
    transition_display = pd.DataFrame(
        [
            {
                "Comparison": f"{predecessor_label} → {current_label}",
                "Δ MAE (dB)": row["delta_MAE_dB"],
                "Δ (%)": row["delta_percent"],
                "95% CI (dB)": (
                    f"[{row['CI95_low_dB']:+.4f}, " f"{row['CI95_high_dB']:+.4f}]"
                ),
            }
        ]
    )
    _display_table(
        transition_display,
        formatter={
            "Δ MAE (dB)": "{:+.4f}",
            "Δ (%)": "{:+.2f}%",
        },
    )
    print(
        "Signed change is current minus predecessor. The result is classified as "
        f"'{row['classification']}' under the predeclared ±"
        f"{practical_margin_db:.2f} dB margin."
    )


def display_s7_diagnostics(
    *,
    frequencies_ghz: np.ndarray,
    truth: np.ndarray,
    current_prediction: np.ndarray,
    current_label: str,
    example_position: int,
    example_id: int,
    high_frequency_start_ghz: float,
    predecessor_prediction: np.ndarray | None = None,
    predecessor_label: str | None = None,
) -> None:
    """
    Display distribution, fixed-example, and frequency-MAE S7 diagnostics.
    """
    current_color = "tab:blue"

    if predecessor_label is None:
        distribution_figure, distribution_axis = plt.subplots(figsize=(8.5, 4.6))
        _distribution_panel(
            distribution_axis,
            frequencies_ghz,
            truth,
            current_prediction,
            current_label,
            current_color,
        )
    else:
        if predecessor_prediction is None:
            raise ValueError("A predecessor prediction is required for comparison.")
        distribution_figure, distribution_axes = plt.subplots(
            1,
            2,
            figsize=(13, 4.5),
            sharex=True,
            sharey=True,
        )
        _distribution_panel(
            distribution_axes[0],
            frequencies_ghz,
            truth,
            predecessor_prediction,
            predecessor_label,
            "tab:orange",
        )
        _distribution_panel(
            distribution_axes[1],
            frequencies_ghz,
            truth,
            current_prediction,
            current_label,
            current_color,
        )
    distribution_figure.suptitle("Test IL_S7_1_DB distribution across designs")
    distribution_figure.tight_layout()

    example_figure, example_axis = plt.subplots(figsize=(8.5, 4.6))
    example_axis.plot(
        frequencies_ghz,
        truth[example_position],
        color="black",
        linewidth=2.0,
        label="truth",
    )
    if predecessor_label is not None and predecessor_prediction is not None:
        example_axis.plot(
            frequencies_ghz,
            predecessor_prediction[example_position],
            color="tab:orange",
            label=predecessor_label,
        )
    example_axis.plot(
        frequencies_ghz,
        current_prediction[example_position],
        color=current_color,
        label=current_label,
    )
    example_axis.set(
        title=f"Fixed test design: SIMU_INDEX {example_id}",
        xlabel="Frequency (GHz)",
        ylabel="IL_S7_1_DB (dB)",
        xlim=(frequencies_ghz[0], frequencies_ghz[-1]),
    )
    example_axis.grid(True, alpha=0.25)
    example_axis.legend()
    example_figure.tight_layout()

    frequency_figure, frequency_axis = plt.subplots(figsize=(8.5, 4.6))
    if predecessor_label is not None and predecessor_prediction is not None:
        previous_frequency_mae = np.mean(
            np.abs(predecessor_prediction - truth),
            axis=0,
        )
        frequency_axis.plot(
            frequencies_ghz,
            previous_frequency_mae,
            color="tab:orange",
            label=predecessor_label,
        )
    current_frequency_mae = np.mean(np.abs(current_prediction - truth), axis=0)
    frequency_axis.plot(
        frequencies_ghz,
        current_frequency_mae,
        color=current_color,
        label=current_label,
    )
    frequency_axis.axvspan(
        high_frequency_start_ghz,
        frequencies_ghz[-1],
        color="0.85",
        alpha=0.35,
        label=f"high frequency (≥ {high_frequency_start_ghz:g} GHz)",
    )
    frequency_axis.set(
        title="Test MAE by frequency",
        xlabel="Frequency (GHz)",
        ylabel="MAE of IL_S7_1_DB (dB)",
        xlim=(frequencies_ghz[0], frequencies_ghz[-1]),
    )
    frequency_axis.grid(True, alpha=0.25)
    frequency_axis.legend()
    frequency_figure.tight_layout()

    _display_and_close(
        (
            distribution_figure,
            example_figure,
            frequency_figure,
        )
    )


def display_full_smatrix_diagnostics(
    complex_table: pd.DataFrame,
    physics_table: pd.DataFrame,
    *,
    complex_title: str | None = None,
) -> None:
    """
    Display complex-error and physical-diagnostic tables in printable groups.
    """
    complex_display = complex_table.rename(
        columns={
            "split": "Split",
            "ComplexMAE": "Complex MAE",
            "ComplexNRMSE": "Complex NRMSE",
        }
    )
    physics_magnitude = physics_table.loc[
        :,
        [
            "split",
            "matrix",
            "MagnitudeMin",
            "MagnitudeMax",
            "MaximumSingularValue",
        ],
    ].rename(
        columns={
            "split": "Split",
            "matrix": "Matrix",
            "MagnitudeMin": "Minimum |S|",
            "MagnitudeMax": "Maximum |S|",
            "MaximumSingularValue": "Maximum singular value",
        }
    )
    physics_constraints = physics_table.loc[
        :,
        [
            "split",
            "matrix",
            "ReciprocityResidual",
            "PassivityViolationFraction",
            "MeanPassivityExcess",
            "PassivityPenalty",
        ],
    ].rename(
        columns={
            "split": "Split",
            "matrix": "Matrix",
            "ReciprocityResidual": "Reciprocity residual",
            "PassivityViolationFraction": "Passivity violation fraction",
            "MeanPassivityExcess": "Mean passivity excess",
            "PassivityPenalty": "Passivity penalty",
        }
    )
    physics_causality = physics_table.loc[
        :,
        ["split", "matrix", "BandLimitedCausalityResidual"],
    ].rename(
        columns={
            "split": "Split",
            "matrix": "Matrix",
            "BandLimitedCausalityResidual": ("Band-limited causality residual"),
        }
    )

    _display_table(complex_display, title=complex_title, precision=6)
    _display_table(
        physics_magnitude,
        title="Magnitude range and maximum singular value",
        precision=6,
    )
    _display_table(
        physics_constraints,
        title="Reciprocity and passivity diagnostics",
        precision=6,
    )
    _display_table(
        physics_causality,
        title="Band-limited causality diagnostic",
        precision=6,
    )


def display_headline_metrics(
    results: pd.DataFrame,
    *,
    model_order: Sequence[str],
    model_labels: Mapping[str, str],
) -> None:
    """
    Display train, validation, and test S7 MAE and RMSE tables.
    """
    headline = results.pivot(
        index="model_name",
        columns="split",
        values=["MAE_dB", "RMSE_dB"],
    ).reindex(model_order)
    headline = headline.reindex(
        columns=pd.MultiIndex.from_product(
            [
                ("MAE_dB", "RMSE_dB"),
                ("train", "validation", "test"),
            ]
        )
    )
    headline.columns = [
        f"{split}_{metric}" for metric, split in headline.columns.to_flat_index()
    ]
    headline = headline.reset_index()

    headline_mae = _headline_metric_table(
        headline,
        metric_name="MAE_dB",
        model_labels=model_labels,
    )
    headline_rmse = _headline_metric_table(
        headline,
        metric_name="RMSE_dB",
        model_labels=model_labels,
    )
    _display_table(headline_mae, title="S7 MAE (dB)", precision=4, na_rep="—")
    _display_table(
        headline_rmse,
        title="S7 RMSE (dB)",
        precision=4,
        na_rep="—",
    )


def display_transition_summary(
    transition_rows: Sequence[Mapping[str, object]],
    *,
    model_labels: Mapping[str, str],
) -> None:
    """
    Display all precomputed model transitions in experiment order.
    """
    table = pd.DataFrame(
        [
            {
                "Comparison": (
                    f"{model_labels[str(row['predecessor'])]} → "
                    f"{model_labels[str(row['current'])]}"
                ),
                "Δ MAE (dB)": row["delta_MAE_dB"],
                "Δ (%)": row["delta_percent"],
                "95% CI (dB)": (
                    f"[{row['CI95_low_dB']:+.4f}, " f"{row['CI95_high_dB']:+.4f}]"
                ),
                "Interpretation": row["classification"],
            }
            for row in transition_rows
        ]
    )
    _display_table(
        table,
        formatter={
            "Δ MAE (dB)": "{:+.4f}",
            "Δ (%)": "{:+.2f}%",
        },
    )


def display_model_choices(
    choice_table: pd.DataFrame,
    *,
    model_labels: Mapping[str, str],
) -> None:
    """
    Display model choices as separate accuracy and trade-off tables.
    """
    choice_summary = choice_table.loc[
        :,
        ["use case", "choose", "test S7 MAE (dB)", "artifact size"],
    ].rename(columns={"use case": "Use case", "choose": "Model"})
    choice_summary["Model"] = choice_summary["Model"].map(model_labels)
    choice_tradeoffs = choice_table.loc[
        :,
        ["choose", "output scope", "physics status", "give up"],
    ].rename(
        columns={
            "choose": "Model",
            "output scope": "Output scope",
            "physics status": "Physics status",
            "give up": "Trade-off",
        }
    )
    choice_tradeoffs["Model"] = choice_tradeoffs["Model"].map(model_labels)

    _display_table(
        choice_summary,
        title="Accuracy and saved-model size",
        formatter={"test S7 MAE (dB)": "{:.4f}"},
    )
    _display_table(
        choice_tradeoffs,
        title="Output scope and physical properties",
    )


def display_provenance_tables(
    provenance: pd.DataFrame,
    *,
    model_labels: Mapping[str, str],
) -> None:
    """
    Display selected-run identity and saved-model cost tables.
    """
    provenance_identity = provenance.loc[
        :,
        ["model_name", "run_id", "target_scope"],
    ].rename(
        columns={
            "model_name": "Model",
            "run_id": "Selected run",
            "target_scope": "Output scope",
        }
    )
    provenance_identity["Model"] = provenance_identity["Model"].map(model_labels)
    provenance_cost = provenance.loc[
        :,
        ["model_name", "artifact_size", "parameter_count"],
    ].rename(
        columns={
            "model_name": "Model",
            "artifact_size": "Artifact size",
            "parameter_count": "Parameters",
        }
    )
    provenance_cost["Model"] = provenance_cost["Model"].map(model_labels)

    _display_table(
        provenance_identity,
        title="Selected runs and output scope",
    )
    _display_table(
        provenance_cost,
        title="Saved-model size and parameter count",
        formatter={"Parameters": "{:.0f}"},
        na_rep="—",
    )


def display_native_six_metrics(
    six_rows: Sequence[Mapping[str, object]],
    *,
    model_labels: Mapping[str, str],
) -> None:
    """
    Display native six-path validation and test metrics.
    """
    table = pd.DataFrame(six_rows).rename(
        columns={
            "model_name": "Model",
            "split": "Split",
            "MAE_dB": "MAE (dB)",
            "RMSE_dB": "RMSE (dB)",
        }
    )
    table["Model"] = table["Model"].map(model_labels)
    _display_table(
        table,
        title="Native six-path accuracy",
        formatter={"MAE (dB)": "{:.4f}", "RMSE (dB)": "{:.4f}"},
    )


def display_runtime_metrics(
    runtime_table: pd.DataFrame,
    *,
    model_labels: Mapping[str, str],
) -> None:
    """
    Display measured loading and split-prediction times.
    """
    table = runtime_table.loc[
        :,
        [
            "model_name",
            "load_seconds",
            "validation_prediction_seconds",
            "test_prediction_seconds",
        ],
    ].rename(
        columns={
            "model_name": "Model",
            "load_seconds": "Load (s)",
            "validation_prediction_seconds": "Validation prediction (s)",
            "test_prediction_seconds": "Test prediction (s)",
        }
    )
    table["Model"] = table["Model"].map(model_labels)
    _display_table(
        table,
        title="Measured loading and prediction time",
        precision=3,
        na_rep="—",
    )


def display_validation_sweep(
    sweep: pd.DataFrame,
    *,
    model_label: str,
    plot: bool = True,
    degree_column: str | None = None,
) -> None:
    """
    Display one saved validation sweep and its Ridge plot when requested.
    """
    display(Markdown(f"#### {model_label}"))
    _display_table(sweep, precision=6, na_rep="None")
    if not plot:
        return

    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    if degree_column is not None:
        for degree, degree_rows in sweep.groupby(degree_column):
            axis.plot(
                degree_rows["alpha"],
                degree_rows["MAE"],
                marker="o",
                label=f"degree {int(degree)}",
            )
        axis.legend()
    else:
        axis.plot(sweep["alpha"], sweep["MAE"], marker="o")
    axis.set_xscale("log")
    axis.set(
        title=f"{model_label} validation sweep",
        xlabel="Ridge alpha",
        ylabel="Validation MAE (dB)",
    )
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    _display_and_close((figure,))


def display_training_history(
    history: pd.DataFrame,
    *,
    model_label: str,
    marker_epoch: int,
    marker_label: str,
    primary_ylabel: str,
    secondary_train_column: str | None = None,
    secondary_validation_column: str | None = None,
    secondary_ylabel: str | None = None,
) -> None:
    """
    Display one neural training history with its reported epoch marker.
    """
    epochs = history["epoch"].to_numpy(dtype=int)
    has_secondary_panel = secondary_train_column is not None

    if has_secondary_panel:
        figure, axes_array = plt.subplots(1, 2, figsize=(12, 4.4))
        axes = list(axes_array)
    else:
        figure, axis = plt.subplots(figsize=(8, 4.4))
        axes = [axis]

    axes[0].plot(epochs, history["loss"], label="training")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(ylabel=primary_ylabel)

    if has_secondary_panel:
        if secondary_validation_column is None or secondary_ylabel is None:
            raise ValueError("A complete secondary metric specification is required.")
        axes[1].plot(
            epochs,
            history[secondary_train_column],
            label="training",
        )
        axes[1].plot(
            epochs,
            history[secondary_validation_column],
            label="validation",
        )
        axes[1].set(ylabel=secondary_ylabel)

    for axis in axes:
        axis.axvline(
            marker_epoch,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=marker_label,
        )
        axis.set(xlabel="Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(f"{model_label} training history")
    figure.tight_layout()
    _display_and_close((figure,))


def display_reproduction_summary(
    reproduction_summary: pd.DataFrame,
    *,
    model_labels: Mapping[str, str],
) -> None:
    """
    Display the per-model metric-reproduction audit summary.
    """
    table = reproduction_summary.rename(
        columns={
            "model_name": "Model",
            "checks": "Checks",
            "largest_tolerance_fraction": "Largest difference / tolerance",
            "status": "Status",
        }
    )
    table["Model"] = table["Model"].map(model_labels)
    _display_table(
        table,
        formatter={"Largest difference / tolerance": "{:.3f}"},
    )


def _display_table(
    table: pd.DataFrame,
    *,
    title: str | None = None,
    formatter: Any = None,
    precision: int | None = None,
    na_rep: str | None = None,
) -> None:
    """
    Display one index-free pandas table with optional report formatting.
    """
    if title is not None:
        display(Markdown(f"**{title}**"))

    styled = table.style.hide(axis="index")
    format_options: dict[str, Any] = {}
    if formatter is not None:
        format_options["formatter"] = formatter
    if precision is not None:
        format_options["precision"] = precision
    if na_rep is not None:
        format_options["na_rep"] = na_rep
    if format_options:
        styled = styled.format(**format_options)
    display(styled)


def _distribution_panel(
    axis: Axes,
    frequencies_ghz: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    title: str,
    color: str,
) -> None:
    """
    Draw true and predicted frequency-wise medians and central 80% bands.
    """
    true_q10, true_median, true_q90 = np.quantile(
        truth,
        [0.1, 0.5, 0.9],
        axis=0,
    )
    pred_q10, pred_median, pred_q90 = np.quantile(
        prediction,
        [0.1, 0.5, 0.9],
        axis=0,
    )
    axis.fill_between(
        frequencies_ghz,
        true_q10,
        true_q90,
        color="0.75",
        alpha=0.45,
        label="truth 10–90%",
    )
    axis.plot(
        frequencies_ghz,
        true_median,
        color="black",
        label="truth median",
    )
    axis.fill_between(
        frequencies_ghz,
        pred_q10,
        pred_q90,
        color=color,
        alpha=0.18,
        label="prediction 10–90%",
    )
    axis.plot(
        frequencies_ghz,
        pred_median,
        color=color,
        label="prediction median",
    )
    axis.set(
        title=title,
        xlabel="Frequency (GHz)",
        ylabel="IL_S7_1_DB (dB)",
        xlim=(frequencies_ghz[0], frequencies_ghz[-1]),
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)


def _headline_metric_table(
    headline: pd.DataFrame,
    *,
    metric_name: str,
    model_labels: Mapping[str, str],
) -> pd.DataFrame:
    """
    Return one display-ready headline metric table.
    """
    table = headline.loc[
        :,
        [
            "model_name",
            f"train_{metric_name}",
            f"validation_{metric_name}",
            f"test_{metric_name}",
        ],
    ].rename(
        columns={
            "model_name": "Model",
            f"train_{metric_name}": "Train",
            f"validation_{metric_name}": "Validation",
            f"test_{metric_name}": "Test",
        }
    )
    table["Model"] = table["Model"].map(model_labels)
    return table


def _display_and_close(figures: Sequence[Figure]) -> None:
    """
    Display figures in order and close their Matplotlib resources.
    """
    for figure in figures:
        display(figure)
        plt.close(figure)
