"""
Tests for non-neural modelling plotting summaries.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sparam_surrogate.utils.non_neural_modelling_utils import (  # noqa: E402
    plot_model_mae_comparison_by_frequency,
    plot_scalar_prediction_band_by_frequency,
    plot_shared_target_mae_comparison,
    plot_shared_target_prediction_bands,
    plot_vector_prediction_bands_by_frequency,
    scalar_prediction_summary_by_frequency,
    vector_prediction_summary_by_frequency,
)


class TestScalarPredictionBand:
    """
    Unit tests for scalar prediction distribution summaries.
    """

    def test_summary_groups_true_and_predicted_values_by_frequency(self) -> None:
        """
        Frequency groups return sorted medians, min/max bands, and counts.
        """
        frame = pd.DataFrame(
            {
                "FREQ_GHZ": [2.0, 1.0, 1.0, 2.0, 1.0, 2.0],
            }
        )
        y_true = np.asarray([-10.0, -1.0, -3.0, -20.0, -5.0, -30.0])
        y_pred = np.asarray([-12.0, -2.0, -4.0, -18.0, -6.0, -24.0])

        summary = scalar_prediction_summary_by_frequency(
            frame,
            y_true,
            y_pred,
            lower_quantile=0.0,
            upper_quantile=1.0,
        )

        assert summary["FREQ_GHZ"].tolist() == [1.0, 2.0]
        assert summary["COUNT"].tolist() == [3, 3]
        np.testing.assert_allclose(summary["TRUE_MEDIAN"], [-3.0, -20.0])
        np.testing.assert_allclose(summary["TRUE_LOWER"], [-5.0, -30.0])
        np.testing.assert_allclose(summary["TRUE_UPPER"], [-1.0, -10.0])
        np.testing.assert_allclose(summary["PREDICTED_MEDIAN"], [-4.0, -18.0])
        np.testing.assert_allclose(summary["PREDICTED_LOWER"], [-6.0, -24.0])
        np.testing.assert_allclose(summary["PREDICTED_UPPER"], [-2.0, -12.0])

    def test_summary_rejects_invalid_lengths(self) -> None:
        """
        Mismatched dataframe and target lengths are reported clearly.
        """
        frame = pd.DataFrame({"FREQ_GHZ": [1.0, 2.0]})

        with pytest.raises(ValueError, match="same length"):
            scalar_prediction_summary_by_frequency(
                frame,
                np.asarray([-1.0]),
                np.asarray([-1.0, -2.0]),
            )

    def test_summary_rejects_invalid_quantiles(self) -> None:
        """
        Quantile bounds must be ordered probabilities.
        """
        frame = pd.DataFrame({"FREQ_GHZ": [1.0]})

        with pytest.raises(ValueError, match="lower_quantile"):
            scalar_prediction_summary_by_frequency(
                frame,
                np.asarray([-1.0]),
                np.asarray([-1.0]),
                lower_quantile=0.9,
                upper_quantile=0.1,
            )

    def test_plot_returns_figure_with_true_and_predicted_bands(self) -> None:
        """
        Plotting creates one axis with two median lines and two percentile bands.
        """
        frame = pd.DataFrame({"FREQ_GHZ": [1.0, 1.0, 2.0, 2.0]})
        y_true = np.asarray([-1.0, -3.0, -2.0, -6.0])
        y_pred = np.asarray([-2.0, -2.5, -4.0, -4.5])

        fig = plot_scalar_prediction_band_by_frequency(
            frame,
            y_true,
            y_pred,
            "IL_S7_1_DB",
        )

        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == (
            "Scalar Ridge Insertion-Loss Distribution: IL_S7_1_DB"
        )
        assert ax.get_ylabel() == "Insertion Loss (dB)"
        assert len(ax.lines) == 2
        assert len(ax.collections) == 2
        plt.close(fig)


class TestVectorPredictionBand:
    """
    Unit tests for vector prediction distribution summaries.
    """

    def test_summary_groups_each_target_by_frequency(self) -> None:
        """
        Vector summaries stack one scalar frequency summary per target.
        """
        frame = pd.DataFrame({"FREQ_GHZ": [1.0, 1.0, 2.0, 2.0]})
        y_true = np.asarray(
            [
                [-1.0, -10.0],
                [-3.0, -30.0],
                [-2.0, -20.0],
                [-6.0, -60.0],
            ]
        )
        y_pred = np.asarray(
            [
                [-2.0, -11.0],
                [-2.5, -25.0],
                [-4.0, -22.0],
                [-4.5, -55.0],
            ]
        )

        summary = vector_prediction_summary_by_frequency(
            frame,
            y_true,
            y_pred,
            ("IL_S7_1_DB", "IL_S8_2_DB"),
            lower_quantile=0.0,
            upper_quantile=1.0,
        )

        assert summary["TARGET"].tolist() == [
            "IL_S7_1_DB",
            "IL_S7_1_DB",
            "IL_S8_2_DB",
            "IL_S8_2_DB",
        ]
        assert summary["FREQ_GHZ"].tolist() == [1.0, 2.0, 1.0, 2.0]
        np.testing.assert_allclose(
            summary["TRUE_MEDIAN"],
            [-2.0, -4.0, -20.0, -40.0],
        )
        np.testing.assert_allclose(
            summary["PREDICTED_MEDIAN"],
            [-2.25, -4.25, -18.0, -38.5],
        )

    def test_summary_rejects_name_count_mismatch(self) -> None:
        """
        Target names must align with vector target columns.
        """
        frame = pd.DataFrame({"FREQ_GHZ": [1.0]})
        y_true = np.asarray([[-1.0, -2.0]])
        y_pred = np.asarray([[-1.0, -2.0]])

        with pytest.raises(ValueError, match="names"):
            vector_prediction_summary_by_frequency(
                frame,
                y_true,
                y_pred,
                ("S7_1_DB",),
            )

    def test_plot_returns_one_axis_per_vector_target(self) -> None:
        """
        Vector plotting creates one populated subplot for each target name.
        """
        frame = pd.DataFrame({"FREQ_GHZ": [1.0, 1.0, 2.0, 2.0]})
        y_true = np.asarray(
            [
                [-1.0, -10.0],
                [-3.0, -30.0],
                [-2.0, -20.0],
                [-6.0, -60.0],
            ]
        )
        y_pred = np.asarray(
            [
                [-2.0, -11.0],
                [-2.5, -25.0],
                [-4.0, -22.0],
                [-4.5, -55.0],
            ]
        )

        fig = plot_vector_prediction_bands_by_frequency(
            frame,
            y_true,
            y_pred,
            ("IL_S7_1_DB", "IL_S8_2_DB"),
            model_name="Polynomial",
        )

        assert len(fig.axes) == 2
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == (
            "Polynomial Insertion-Loss Distributions"
        )
        assert [ax.get_title() for ax in fig.axes] == [
            "IL_S7_1_DB",
            "IL_S8_2_DB",
        ]
        for ax in fig.axes:
            assert ax.get_ylabel() == "Insertion Loss (dB)"
            assert len(ax.lines) == 2
            assert len(ax.collections) == 2
        plt.close(fig)


class TestModelMaeComparison:
    """
    Unit tests for model-comparison plotting.
    """

    def test_plot_returns_one_line_per_model(self) -> None:
        """
        Mean vector MAE is plotted once for each named prediction matrix.
        """
        frame = pd.DataFrame({"FREQ_GHZ": [1.0, 1.0, 2.0, 2.0]})
        y_true = np.asarray(
            [
                [-1.0, -10.0],
                [-3.0, -30.0],
                [-2.0, -20.0],
                [-6.0, -60.0],
            ]
        )
        predictions = {
            "Vector Ridge": np.asarray(
                [
                    [-2.0, -11.0],
                    [-2.5, -25.0],
                    [-4.0, -22.0],
                    [-4.5, -55.0],
                ]
            ),
            "Polynomial": np.asarray(
                [
                    [-1.5, -10.5],
                    [-2.8, -28.0],
                    [-3.0, -21.0],
                    [-5.0, -58.0],
                ]
            ),
        }

        fig = plot_model_mae_comparison_by_frequency(
            frame,
            y_true,
            predictions,
            ("IL_S7_1_DB", "IL_S8_2_DB"),
        )

        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == "Model MAE Comparison by Frequency"
        assert [line.get_label() for line in ax.lines] == [
            "Vector Ridge",
            "Polynomial",
        ]
        plt.close(fig)


class TestSharedTargetComparison:
    """
    Unit tests for scalar-target comparisons across several model families.
    """

    def test_mae_plot_returns_one_line_per_model(self) -> None:
        """
        Shared-target MAE curves are plotted once for each model.
        """
        frame = pd.DataFrame({"FREQ_GHZ": [1.0, 1.0, 2.0, 2.0]})
        y_true = np.asarray([-1.0, -3.0, -2.0, -6.0])
        predictions = {
            "Scalar Ridge": np.asarray([-2.0, -2.5, -4.0, -4.5]),
            "Vector Ridge": np.asarray([-1.5, -2.8, -3.0, -5.0]),
            "Polynomial Ridge": np.asarray([-1.2, -3.2, -2.5, -5.5]),
        }

        fig = plot_shared_target_mae_comparison(
            frame,
            y_true,
            predictions,
            "IL_S7_1_DB",
        )

        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == "IL_S7_1_DB MAE by Frequency"
        assert [line.get_label() for line in ax.lines] == [
            "Scalar Ridge",
            "Vector Ridge",
            "Polynomial Ridge",
        ]
        plt.close(fig)

    def test_prediction_bands_plot_true_and_each_model_distribution(self) -> None:
        """
        Distribution comparison includes true and per-model predicted bands.
        """
        frame = pd.DataFrame({"FREQ_GHZ": [1.0, 1.0, 2.0, 2.0]})
        y_true = np.asarray([-1.0, -3.0, -2.0, -6.0])
        predictions = {
            "Scalar Ridge": np.asarray([-2.0, -2.5, -4.0, -4.5]),
            "Vector Ridge": np.asarray([-1.5, -2.8, -3.0, -5.0]),
            "Polynomial Ridge": np.asarray([-1.2, -3.2, -2.5, -5.5]),
        }

        fig = plot_shared_target_prediction_bands(
            frame,
            y_true,
            predictions,
            "IL_S7_1_DB",
        )

        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == (
            "IL_S7_1_DB Predicted Distribution Comparison"
        )
        assert ax.get_ylabel() == "Insertion Loss (dB)"
        assert len(ax.lines) == 4
        assert len(ax.collections) == 4
        assert [line.get_label() for line in ax.lines] == [
            "True median",
            "Scalar Ridge predicted median",
            "Vector Ridge predicted median",
            "Polynomial Ridge predicted median",
        ]
        plt.close(fig)
