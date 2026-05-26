"""
Unit tests for :class:`PcbDatasetEDA`.
"""

from unittest.mock import patch

import pandas as pd
from matplotlib import pyplot as plt
from pandas.testing import assert_frame_equal

from sparam_surrogate.data.pcb_dataset_eda import PcbDatasetEDA
from sparam_surrogate.data.pcb_parameters import PcbParameters


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
            fig = eda.plot_distribution_histograms(["START", "PITCH", "TRACE_LEN"], show=False)

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
