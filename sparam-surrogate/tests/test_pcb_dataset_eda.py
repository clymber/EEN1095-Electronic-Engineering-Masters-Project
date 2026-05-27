"""
Unit tests for :class:`PcbDatasetEDA`.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.testing import assert_frame_equal

from sparam_surrogate.data.pcb_dataset_eda import PcbDatasetEDA
from sparam_surrogate.data.pcb_parameters import PcbParameters
from sparam_surrogate.data.s_parameter_dataset import SParameterDataset


def _make_parameters() -> PcbParameters:
    return PcbParameters(
        pd.DataFrame(
            {
                "START": [10.0, 20.0, 30.0],
                "PITCH": [5.0, 6.0, 7.0],
                "TRACE_LEN": [100.0, 120.0, 140.0],
                "VIAR": [1.0, 2.0, 2.0],
                "ANTIPADR": [2.0, 3.0, 4.0],
                "TDIEL": [2.0, 2.5, 4.0],
                "TLWIDTH": [1.0, 1.0, 2.0],
                "DISTTL": [3.0, 3.5, 4.0],
                "LABEL": ["a", "b", "c"],
            }
        )
    )


def _make_response_eda() -> PcbDatasetEDA:
    parameters = PcbParameters(
        pd.DataFrame(
            {
                "SIMU_INDEX": [0, 1, 2],
                "START": [10.0, 20.0, 30.0],
                "PITCH": [5.0, 6.0, 7.0],
                "TRACE_LEN": [100.0, 120.0, 140.0],
                "VIAR": [1.0, 2.0, 2.0],
                "ANTIPADR": [2.0, 3.0, 4.0],
                "TDIEL": [2.0, 2.5, 4.0],
                "TLWIDTH": [1.0, 1.0, 2.0],
                "DISTTL": [3.0, 3.5, 4.0],
                "EPS": [3.8, 4.0, 4.2],
                "TAND": [0.001, 0.01, 0.02],
            }
        )
    )
    responses = SParameterDataset(
        simulation_indices=[0, 2],
        frequencies_ghz=[1.0, 10.0],
        port_pairs=[(7, 1), (8, 2)],
        through_s_db=np.array(
            [
                [[-1.0, -1.2], [-3.0, -3.2]],
                [[-2.0, -2.2], [-5.0, -5.2]],
            ]
        ),
    )
    return PcbDatasetEDA(parameters, responses)


class TestPcbDatasetEDA:
    """
    Unit tests for derived feature exploration.
    """

    def test_derives_engineering_features_without_mutating_parameters(
        self,
    ) -> None:
        """
        Test that EDA derives engineering features without mutating inputs.
        """
        parameters = _make_parameters()

        eda = PcbDatasetEDA(parameters)

        expected = pd.DataFrame(
            {
                "BOARD_HEIGHT": [65.0, 94.0, 123.0],
                "BOARD_WIDTH": [210.0, 268.0, 326.0],
                "ANTIPAD_TO_VIA_RATIO": [2.0, 1.5, 2.0],
                "TLWIDTH_TO_DIEL_RATIO": [0.5, 0.4, 0.5],
                "TRACE_ASPECT_RATIO": [20.0, 20.0, 20.0],
                "BOARD_AREA": [13650.0, 25192.0, 40098.0],
            }
        )
        assert_frame_equal(eda.dataframe[expected.columns], expected)
        assert "BOARD_HEIGHT" not in parameters.dataframe.columns

    def test_statistical_summary_describes_derived_columns(self, capsys) -> None:
        """
        Test that the statistical summary reports requested derived columns.
        """
        eda = PcbDatasetEDA(_make_parameters())

        eda.statistical_summary(["BOARD_HEIGHT", "BOARD_WIDTH"])
        summary = capsys.readouterr().out

        assert "BOARD_HEIGHT" in summary
        assert "BOARD_WIDTH" in summary
        assert "START" not in summary

    def test_plot_distribution(self) -> None:
        """
        Test that distribution histograms render the requested parameters.
        """
        eda = PcbDatasetEDA(_make_parameters())

        with patch("sparam_surrogate.data.pcb_dataset_eda.plt.show") as show:
            fig = eda.plot_distribution_histograms(
                ["START", "PITCH", "TRACE_LEN"], show=False
            )

        assert fig is not None
        assert len(fig.axes) == 4
        assert [ax.get_title() for ax in fig.axes[:3]] == [
            "START",
            "PITCH",
            "TRACE_LEN",
        ]
        assert fig.axes[0].get_xlabel() == "START"
        assert fig.axes[0].get_ylabel() == "Count"
        assert sum(bar.get_height() for bar in fig.axes[0].patches) == 3
        assert not fig.axes[3].get_visible()
        show.assert_not_called()

        plt.close(fig)

    def test_plot_correlation_heatmap(self) -> None:
        """
        Test that the correlation heatmap includes only numeric parameters.
        """
        eda = PcbDatasetEDA(_make_parameters())

        with patch("sparam_surrogate.data.pcb_dataset_eda.plt.show") as show:
            fig = eda.plot_correlation_heatmap(["START", "PITCH", "LABEL"], show=False)

        assert fig is not None
        assert len(fig.axes) == 2
        heatmap, colorbar = fig.axes
        assert heatmap.get_title() == (
            "Correlation heatmap of physical and derived parameters"
        )
        assert [tick.get_text() for tick in heatmap.get_xticklabels()] == [
            "START",
            "PITCH",
        ]
        assert len(heatmap.images) == 1
        assert colorbar.get_ylabel() == "Pearson correlation coefficient"
        show.assert_not_called()

        plt.close(fig)

    def test_plot_physical_relationships(self) -> None:
        """
        Test that physical relationship plots include expected comparisons.
        """
        eda = PcbDatasetEDA(_make_parameters())

        with patch("sparam_surrogate.data.pcb_dataset_eda.plt.show") as show:
            fig = eda.plot_physical_relationships(show=False)

        assert fig is not None
        assert len(fig.axes) == 4
        assert fig._suptitle.get_text() == "Physical parameter relationships"
        assert [ax.get_title() for ax in fig.axes] == [
            "ANTIPADR vs VIAR",
            "TDIEL vs TLWIDTH",
            "TRACE_LEN vs PITCH",
            "TLWIDTH vs DISTTL",
        ]
        assert len(fig.axes[0].collections[0].get_offsets()) == 3
        assert [text.get_text() for text in fig.axes[0].texts] == [
            "ANTIPADR = VIAR",
        ]
        assert [text.get_text() for text in fig.axes[3].texts] == [
            "TLWIDTH = DISTTL",
        ]
        assert len(fig.axes[0].lines) == 1
        assert len(fig.axes[3].lines) == 1
        show.assert_not_called()

        plt.close(fig)

    def test_plot_board_geometry_verification(self) -> None:
        """
        Test that board geometry plots visualize derived dimensions.
        """
        eda = PcbDatasetEDA(_make_parameters())

        with patch("sparam_surrogate.data.pcb_dataset_eda.plt.show") as show:
            fig = eda.plot_board_geometry_verification(show=False)

        assert fig is not None
        assert len(fig.axes) == 2
        assert fig._suptitle.get_text() == "Derived board-geometry verification"
        assert [ax.get_title() for ax in fig.axes] == [
            "BOARD_HEIGHT vs PITCH",
            "BOARD_WIDTH vs TRACE_LEN",
        ]
        assert fig.axes[0].get_xlabel() == "PITCH"
        assert fig.axes[0].get_ylabel() == "BOARD_HEIGHT"
        assert len(fig.axes[0].collections[0].get_offsets()) == 3
        show.assert_not_called()

        plt.close(fig)

    def test_plot_ratio_relationships(self) -> None:
        """
        Test that ratio relationship plots color points by derived ratios.
        """
        eda = PcbDatasetEDA(_make_parameters())

        with patch("sparam_surrogate.data.pcb_dataset_eda.plt.show") as show:
            fig = eda.plot_ratio_relationships(show=False)

        assert fig is not None
        assert len(fig.axes) == 6
        plots = fig.axes[:3]
        colorbars = fig.axes[3:]
        assert [ax.get_title() for ax in plots] == [
            "ANTIPADR vs VIAR",
            "TLWIDTH vs TDIEL",
            "TRACE_LEN vs PITCH",
        ]
        assert [ax.get_ylabel() for ax in colorbars] == [
            "ANTIPAD_TO_VIA_RATIO",
            "TLWIDTH_TO_DIEL_RATIO",
            "TRACE_ASPECT_RATIO",
        ]
        assert len(plots[0].collections[0].get_offsets()) == 3
        show.assert_not_called()

        plt.close(fig)

    def test_correlation_pairs(self) -> None:
        """
        Test that correlation pairs are numeric, unique, and sorted by strength.
        """
        eda = PcbDatasetEDA(
            PcbParameters(
                pd.DataFrame(
                    {
                        "START": [1.0, 2.0, 3.0, 4.0],
                        "PITCH": [2.0, 4.0, 6.0, 8.0],
                        "TRACE_LEN": [1.0, 1.0, 2.0, 4.0],
                        "VIAR": [1.0, 1.0, 2.0, 2.0],
                        "ANTIPADR": [2.0, 2.0, 3.0, 3.0],
                        "TDIEL": [1.0, 1.0, 2.0, 2.0],
                        "TLWIDTH": [0.5, 0.5, 1.0, 1.0],
                        "LABEL": ["a", "b", "c", "d"],
                    }
                )
            )
        )

        pairs = eda.correlation_pairs(["START", "PITCH", "TRACE_LEN", "LABEL"])

        assert len(pairs) == 3
        assert pairs.iloc[0]["feature_1"] == "START"
        assert pairs.iloc[0]["feature_2"] == "PITCH"
        assert pairs.iloc[0]["corr"] == 1.0
        assert pairs["abs_corr"].is_monotonic_decreasing
        assert set(zip(pairs["feature_1"], pairs["feature_2"], strict=True)) == {
            ("START", "PITCH"),
            ("START", "TRACE_LEN"),
            ("PITCH", "TRACE_LEN"),
        }

    def test_response_frame_inner_joins_aligned_simulations(self) -> None:
        eda = _make_response_eda()

        response_frame = eda.response_frame_at_frequency()

        assert response_frame["SIMU_INDEX"].to_list() == [0, 2]
        assert response_frame["S7_1_DB"].to_list() == [-3.0, -5.0]
        assert "BOARD_AREA" in response_frame.columns

    def test_response_correlations_are_feature_to_target_pairs(self) -> None:
        eda = _make_response_eda()

        correlations = eda.response_correlation_pairs(features=["TRACE_LEN", "TAND"])

        assert set(correlations["feature"]) == {"TRACE_LEN", "TAND"}
        assert set(correlations["response"]) == {"S7_1_DB", "S8_2_DB"}
        assert correlations["abs_corr"].is_monotonic_decreasing

    def test_plot_response_distributions(self) -> None:
        eda = _make_response_eda()

        with patch("sparam_surrogate.data.pcb_dataset_eda.plt.show") as show:
            fig = eda.plot_response_distributions(show=False)

        assert fig is not None
        assert [axis.get_title() for axis in fig.axes] == [
            "S7_1_DB at 10 GHz",
            "S8_2_DB at 10 GHz",
        ]
        assert fig._suptitle.get_text() == "Through-path response distributions"
        show.assert_not_called()
        plt.close(fig)

    def test_plot_parameter_response_relationships(self) -> None:
        eda = _make_response_eda()

        with patch("sparam_surrogate.data.pcb_dataset_eda.plt.show") as show:
            fig = eda.plot_parameter_response_relationships(
                pair=(7, 1),
                features=["TRACE_LEN", "TAND"],
                show=False,
            )

        assert fig is not None
        assert [axis.get_title() for axis in fig.axes] == [
            "S7_1_DB vs TRACE_LEN",
            "S7_1_DB vs TAND",
        ]
        assert len(fig.axes[0].collections[0].get_offsets()) == 2
        show.assert_not_called()
        plt.close(fig)

    def test_plot_through_response_curves_with_simulation_overlay(self) -> None:
        eda = _make_response_eda()

        with patch("sparam_surrogate.data.pcb_dataset_eda.plt.show") as show:
            fig = eda.plot_through_response_curves(simulation_indices=[0], show=False)

        assert fig is not None
        assert [axis.get_title() for axis in fig.axes] == ["S7_1_DB", "S8_2_DB"]
        assert len(fig.axes[0].lines) == 2
        assert [line.get_label() for line in fig.axes[0].lines] == [
            "Median",
            "SIMU_INDEX 0",
        ]
        show.assert_not_called()
        plt.close(fig)
