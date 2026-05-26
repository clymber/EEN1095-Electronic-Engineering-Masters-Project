"""
Unit tests for the :class:`PcbParameters` data wrapper.
"""

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from sparam_surrogate.data.pcb_parameters import PcbParameters


class TestPcbParameters:
    """
    Unit tests for :class:`PcbParameters`.
    """

    def test_getitem_selects_one_column(self) -> None:
        """
        Return one parameter column as a series.
        """
        params = PcbParameters(
            pd.DataFrame(
                {
                    "START": [10.0, 20.0],
                    "PITCH": [5.0, 6.0],
                    "TRACE_LEN": [100.0, 120.0],
                }
            )
        )

        assert_series_equal(
            params["START"],
            pd.Series([10.0, 20.0], name="START"),
        )

    def test_getitem_selects_multiple_columns(self) -> None:
        """
        Return multiple parameter columns selected by list or index.
        """
        parameters = PcbParameters(
            pd.DataFrame(
                {
                    "START": [10.0, 20.0],
                    "PITCH": [5.0, 6.0],
                    "TRACE_LEN": [100.0, 120.0],
                }
            )
        )
        expected = pd.DataFrame(
            {
                "PITCH": [5.0, 6.0],
                "TRACE_LEN": [100.0, 120.0],
            }
        )

        assert_frame_equal(
            parameters[["PITCH", "TRACE_LEN"]],
            expected,
        )
        assert_frame_equal(
            parameters[pd.Index(["PITCH", "TRACE_LEN"])],
            expected,
        )

    def test_assign_columns(self) -> None:
        """
        Store derived feature columns calculated from parameter data.
        """
        parameters = PcbParameters(
            pd.DataFrame(
                {
                    "START": [10.0, 20.0],
                    "PITCH": [5.0, 6.0],
                    "TRACE_LEN": [100.0, 120.0],
                }
            )
        )

        parameters.assign_columns(
            BOARD_HEIGHT=(
                2 * parameters.dataframe["START"] + 9 * parameters.dataframe["PITCH"]
            ),
            BOARD_WIDTH=(
                2 * parameters.dataframe["START"]
                + parameters.dataframe["TRACE_LEN"]
                + 18 * parameters.dataframe["PITCH"]
            ),
        )

        assert_frame_equal(
            parameters.dataframe[["BOARD_HEIGHT", "BOARD_WIDTH"]],
            pd.DataFrame(
                {
                    "BOARD_HEIGHT": [65.0, 94.0],
                    "BOARD_WIDTH": [210.0, 268.0],
                }
            ),
        )

    def test_preview_selects_one_column(self) -> None:
        """
        Restrict the compact preview to a requested parameter column.
        """
        parameters = PcbParameters(
            pd.DataFrame(
                {
                    "START": [10.0, 20.0],
                    "PITCH": [5.0, 6.0],
                }
            )
        )

        preview = parameters.preview("PITCH")

        assert "PITCH" in preview
        assert "START" not in preview

    def test_structural_summary_selects_columns(self, capsys) -> None:
        """
        Restrict DataFrame structural output to requested parameter columns.
        """
        parameters = PcbParameters(
            pd.DataFrame(
                {
                    "START": [10.0, 20.0],
                    "PITCH": [5.0, 6.0],
                    "TRACE_LEN": [100.0, 120.0],
                }
            )
        )

        parameters.structural_summary(["PITCH", "TRACE_LEN"])
        summary = capsys.readouterr().out

        assert "PITCH" in summary
        assert "TRACE_LEN" in summary
        assert "START" not in summary

    def test_statistical_summary_selects_columns(self, capsys) -> None:
        """
        Describe requested physical columns while excluding the identifier.
        """
        parameters = PcbParameters(
            pd.DataFrame(
                {
                    "START": [10.0, 20.0, 30.0],
                    "PITCH": [5.0, 6.0, 7.0],
                    "SIMU_INDEX": [1, 2, 3],
                }
            )
        )

        parameters.statistical_summary(pd.Index(["PITCH", "SIMU_INDEX"]))
        summary = capsys.readouterr().out

        assert "PITCH" in summary
        assert "SIMU_INDEX" not in summary
        assert "START" not in summary

    def test_statistical_summary_handles_tables_without_identifier(
        self, capsys
    ) -> None:
        """
        Describe derived or partial parameter tables without SIMU_INDEX.
        """
        parameters = PcbParameters(pd.DataFrame({"PITCH": [5.0, 6.0, 7.0]}))

        parameters.statistical_summary("PITCH")
        summary = capsys.readouterr().out

        assert "PITCH" in summary
