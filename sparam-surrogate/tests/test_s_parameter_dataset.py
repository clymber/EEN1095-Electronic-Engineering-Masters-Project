"""
Unit tests for :class:`SParameterDataset`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparam_surrogate.data import (
    PcbParameters,
    RawData,
    SParameterDataset,
)

def _create_mock_touchstone_s2p(
    path: Path,
    frequencies_ghz: list[float],
    s21_magnitudes: list[float]
) -> None:
    """
    Write a minimal two-port Touchstone response in RI format.
    """
    lines = ["# GHz S RI R 50"]
    for frequency, through in zip(frequencies_ghz, s21_magnitudes, strict=True):
        # Touchstone two-port order is S11, S21, S12, S22.
        lines.append(f"{frequency} 0.1 0.0 {through} 0.0 0.01 0.0 0.1 0.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_mock_rawdata(
    tmp_path: Path,
    parameter_indices: list[int],
    curves: dict[int, tuple[list[float], list[float]]]
) -> tuple[PcbParameters, RawData]:
    """
    Create parameter records and corresponding synthetic Touchstones.
    """
    raw_path = tmp_path / "raw"
    variation_path = raw_path / "variation"
    variation_path.mkdir(parents=True)
    parameter_frame = pd.DataFrame({"SIMU_INDEX": parameter_indices})
    parameter_frame.to_csv(raw_path / "parameter.csv", index=False)
    for simulation_index, (frequencies, values) in curves.items():
        _create_mock_touchstone_s2p(
            variation_path / f"simu_{simulation_index}.s2p",
            frequencies,
            values,
        )
    return PcbParameters(parameter_frame), RawData(raw_path, nports=2)


class TestSParameterDataset:
    """
    Unit tests for aligned Touchstone response extraction.
    """

    def test_extracts_requested_path_in_db_with_touchstone_port_order(
        self, tmp_path: Path
    ) -> None:
        """Extract the requested Touchstone path and convert magnitudes to dB."""
        parameters, raw_data = _create_mock_rawdata(
            tmp_path,
            parameter_indices=[0, 1],
            curves={
                0: ([1.0, 10.0], [0.5, 0.25]),
                1: ([1.0, 10.0], [1.0, 0.5]),
            },
        )

        responses = SParameterDataset.from_touchstones(parameters, raw_data, [(2, 1)])
        response_at_10ghz = responses.at_frequency(10.0)

        assert responses.port_pairs == ((2, 1),)
        assert responses.through_s_db.shape == (2, 2, 1)
        np.testing.assert_allclose(
            response_at_10ghz["S2_1_DB"],
            [20 * np.log10(0.25), 20 * np.log10(0.5)],
        )

    def test_aligns_on_simulation_index_and_preserves_consistency_report(
        self, tmp_path: Path
    ) -> None:
        """Align matching simulations and report missing or extra source records."""
        parameters, raw_data = _create_mock_rawdata(
            tmp_path,
            parameter_indices=[0, 1, 2],
            curves={
                0: ([1.0, 10.0], [0.8, 0.7]),
                2: ([1.0, 10.0], [0.7, 0.6]),
                4: ([1.0, 10.0], [0.6, 0.5]),
            },
        )

        responses = SParameterDataset.from_touchstones(parameters, raw_data, [(2, 1)])

        np.testing.assert_array_equal(responses.simulation_indices, [0, 2])
        assert responses.alignment_report == {
            "parameter_count": 3,
            "touchstone_count": 3,
            "missing_parameter_records": [4],
            "missing_touchstones": [1],
            "extra_touchstone_files": ["simu_4.s2p"],
        }

    def test_rejects_invalid_pair_and_unavailable_frequency(
        self, tmp_path: Path
    ) -> None:
        """Reject out-of-range port pairs and absent requested frequencies."""
        parameters, raw_data = _create_mock_rawdata(
            tmp_path,
            parameter_indices=[0],
            curves={0: ([1.0, 10.0], [0.8, 0.7])},
        )

        with pytest.raises(ValueError, match="ports between 1 and 2"):
            SParameterDataset.from_touchstones(parameters, raw_data, [(3, 1)])

        responses = SParameterDataset.from_touchstones(parameters, raw_data, [(2, 1)])
        with pytest.raises(ValueError, match="is not present"):
            responses.at_frequency(5.0)

    def test_rejects_inconsistent_grids_and_nonfinite_response(
        self, tmp_path: Path
    ) -> None:
        """Reject mismatched frequency grids and non-finite dB responses."""
        parameters, raw_data = _create_mock_rawdata(
            tmp_path / "grid",
            parameter_indices=[0, 1],
            curves={
                0: ([1.0, 10.0], [0.8, 0.7]),
                1: ([1.0, 11.0], [0.8, 0.7]),
            },
        )

        with pytest.raises(ValueError, match="common frequency grid"):
            SParameterDataset.from_touchstones(parameters, raw_data, [(2, 1)])

        parameters, raw_data = _create_mock_rawdata(
            tmp_path / "finite",
            parameter_indices=[0],
            curves={0: ([1.0, 10.0], [0.8, 0.0])},
        )
        with pytest.raises(ValueError, match="Non-finite dB response"):
            SParameterDataset.from_touchstones(parameters, raw_data, [(2, 1)])

    def test_cache_round_trip_and_explicit_rebuild(self, tmp_path: Path) -> None:
        """Reuse cached responses unless an explicit rebuild is requested."""
        parameters, raw_data = _create_mock_rawdata(
            tmp_path,
            parameter_indices=[0],
            curves={0: ([1.0, 10.0], [0.8, 0.5])},
        )
        cache_path = tmp_path / "interim" / "responses.npz"
        first = SParameterDataset.from_touchstones(
            parameters, raw_data, [(2, 1)], cache_path=cache_path
        )

        _create_mock_touchstone_s2p(raw_data.touchstone(0), [1.0, 10.0], [0.8, 0.25])
        cached = SParameterDataset.from_touchstones(
            parameters, raw_data, [(2, 1)], cache_path=cache_path
        )
        rebuilt = SParameterDataset.from_touchstones(
            parameters,
            raw_data,
            [(2, 1)],
            cache_path=cache_path,
            rebuild_cache=True,
        )

        assert cache_path.is_file()
        assert first.at_frequency(10.0)["S2_1_DB"].iat[0] == pytest.approx(
            20 * np.log10(0.5)
        )
        assert cached.at_frequency(10.0)["S2_1_DB"].iat[0] == pytest.approx(
            20 * np.log10(0.5)
        )
        assert rebuilt.at_frequency(10.0)["S2_1_DB"].iat[0] == pytest.approx(
            20 * np.log10(0.25)
        )
