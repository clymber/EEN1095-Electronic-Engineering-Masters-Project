"""
Unit tests for :class:`SParameterDataset`.

The helpers in this module build small, synthetic Touchstone datasets so the
tests can exercise response extraction, alignment, validation, and cache
behaviour without relying on external files.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sparam_surrogate.data import (
    PcbParameters,
    RawData,
    SParameterDataset,
)


def _create_mock_touchstone_s2p(
    path: Path, frequencies_ghz: list[float], s21_magnitudes: list[float]
) -> None:
    """
    Write a minimal two-port Touchstone response in RI format.

    path:
        Destination ``.s2p`` file to create.
    frequencies_ghz:
        Frequency samples in GHz. Each sample is paired with the S21 magnitude at the
        same list position.
    s21_magnitudes:
        Real S21 magnitudes to write for the through path. The imaginary part is written
        as zero, and the other S-parameters use fixed finite values.
    """
    lines = ["# GHz S RI R 50"]
    for frequency, through in zip(frequencies_ghz, s21_magnitudes, strict=True):
        # Touchstone two-port order is S11, S21, S12, S22.
        lines.append(f"{frequency} 0.1 0.0 {through} 0.0 0.01 0.0 0.1 0.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_mock_touchstone_s2p_full(
    path: Path,
    frequencies_ghz: list[float],
    matrices: list[np.ndarray],
) -> None:
    """
    Write a minimal two-port Touchstone response with full complex matrices.

    path:
        Destination ``.s2p`` file to create.
    frequencies_ghz:
        Frequency samples in GHz. Each sample is paired with the complex matrix at the
        same list position.
    matrices:
        Complex two-by-two S-parameter matrices, indexed as ``[receiver, source]`` with
        zero-based NumPy axes. Values are written in Touchstone two-port order.
    """
    lines = ["# GHz S RI R 50"]
    for frequency, matrix in zip(frequencies_ghz, matrices, strict=True):
        # Touchstone two-port order is S11, S21, S12, S22.
        values = [
            matrix[0, 0], # S11
            matrix[1, 0], # S21
            matrix[0, 1], # S12
            matrix[1, 1], # S22
        ]
        row = [f"{frequency:g}"]
        for value in values:
            row.extend([f"{value.real:g}", f"{value.imag:g}"])
        lines.append(" ".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_mock_rawdata(
    tmp_path: Path,
    parameter_indices: list[int],
    curves: dict[int, tuple[list[float], list[float]]],
) -> tuple[PcbParameters, RawData]:
    """
    Create parameter records and corresponding synthetic Touchstones.

    tmp_path:
        Temporary directory where the ``raw`` and ``variation`` folders are created.
    parameter_indices:
        Simulation identifiers written to ``parameter.csv`` as ``SIMU_INDEX`` rows.
    curves:
        Mapping from simulation index to ``(frequencies_ghz, s21_magnitudes)`` values
        used to create ``simu_<index>.s2p`` files.

    Returns: tuple[PcbParameters, RawData]
        Parameter table wrapper and raw-data locator pointing at the generated two-port
        Touchstone files.
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


def _create_mock_rawdata_full(
    tmp_path: Path,
    parameter_indices: list[int],
    responses: dict[int, tuple[list[float], list[np.ndarray]]],
) -> tuple[PcbParameters, RawData]:
    """
    Create parameter records and full complex two-port Touchstones.

    tmp_path:
        Temporary directory where the ``raw`` and ``variation`` folders are created.
    parameter_indices:
        Simulation identifiers written to ``parameter.csv`` as ``SIMU_INDEX`` rows.
    responses:
        Mapping from simulation index to ``(frequencies_ghz, matrices)`` values used to
        create ``simu_<index>.s2p`` files with complete complex S-matrices.
    Returns: tuple[PcbParameters, RawData]
        Parameter table wrapper and raw-data locator pointing at the generated two-port
        Touchstone files.
    """
    raw_path = tmp_path / "raw"
    variation_path = raw_path / "variation"
    variation_path.mkdir(parents=True)
    parameter_frame = pd.DataFrame({"SIMU_INDEX": parameter_indices})
    parameter_frame.to_csv(raw_path / "parameter.csv", index=False)
    for simulation_index, (frequencies, matrices) in responses.items():
        _create_mock_touchstone_s2p_full(
            variation_path / f"simu_{simulation_index}.s2p",
            frequencies,
            matrices,
        )
    return PcbParameters(parameter_frame), RawData(raw_path, nports=2)


class TestSParameterDataset:
    """
    Unit tests for aligned Touchstone response extraction.
    """

    def test_extracts_requested_path_in_db_with_touchstone_port_order(
        self, tmp_path: Path
    ) -> None:
        """
        Extract the requested Touchstone path and convert magnitudes to dB.

        tmp_path:
            Pytest temporary directory used to build the synthetic raw-data tree.
        """
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
        assert responses.through_s_db.shape == (2, 2, 1) # (simulation, frequency, path)
        assert responses.through_s_db.dtype == float
        np.testing.assert_allclose(
            response_at_10ghz["S2_1_DB"],
            [20 * np.log10(0.25), 20 * np.log10(0.5)],
        )

    def test_extracts_full_complex_s_matrix(self, tmp_path: Path) -> None:
        """
        Preserve full complex S-matrices alongside selected dB paths.

        tmp_path:
            Pytest temporary directory used to build the synthetic raw-data tree.
        """
        first_matrix = np.array(
            [
                [0.10 + 0.01j, 0.02 - 0.03j],
                [0.50 + 0.20j, 0.30 - 0.04j],
            ],
            dtype=complex,
        )
        second_matrix = np.array(
            [
                [0.11 + 0.02j, 0.03 - 0.04j],
                [0.25 - 0.10j, 0.31 - 0.05j],
            ],
            dtype=complex,
        )
        parameters, raw_data = _create_mock_rawdata_full(
            tmp_path,
            parameter_indices=[0],
            responses={0: ([1.0, 10.0], [first_matrix, second_matrix])},
        )

        responses = SParameterDataset.from_touchstones(parameters, raw_data, [(2, 1)])

        # shape: (simulation, frequency, row, column)
        assert responses.full_s_matrix.shape == (1, 2, 2, 2)
        assert responses.full_s_matrix.dtype == complex
        np.testing.assert_allclose(responses.full_s_matrix[0, 0], first_matrix)
        np.testing.assert_allclose(responses.full_s_matrix[0, 1], second_matrix)
        np.testing.assert_allclose(
            responses.through_s_db[:, :, 0],
            [
                [
                    20 * np.log10(abs(first_matrix[1, 0])),
                    20 * np.log10(abs(second_matrix[1, 0])),
                ]
            ],
        )

    def test_full_complex_s_matrix_cache_round_trip(self, tmp_path: Path) -> None:
        """
        Cache and reload full complex matrices without reparsing Touchstones.

        tmp_path:
            Pytest temporary directory used to build the source Touchstone and interim
            cache files.
        """
        original_matrix = np.array(
            [
                [0.10 + 0.01j, 0.02 - 0.03j],
                [0.50 + 0.20j, 0.30 - 0.04j],
            ],
            dtype=complex,
        )
        changed_matrix = np.array(
            [
                [0.20 + 0.01j, 0.02 - 0.03j],
                [0.25 + 0.10j, 0.30 - 0.04j],
            ],
            dtype=complex,
        )
        parameters, raw_data = _create_mock_rawdata_full(
            tmp_path,
            parameter_indices=[0],
            responses={0: ([1.0], [original_matrix])},
        )
        cache_path = tmp_path / "interim" / "responses_full.npz"

        first = SParameterDataset.from_touchstones(
            parameters, raw_data, [(2, 1)], cache_path=cache_path
        )
        _create_mock_touchstone_s2p_full(
            raw_data.touchstone(0),
            [1.0],
            [changed_matrix],
        )
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

        np.testing.assert_allclose(first.full_s_matrix[0, 0], original_matrix)
        np.testing.assert_allclose(cached.full_s_matrix[0, 0], original_matrix)
        np.testing.assert_allclose(rebuilt.full_s_matrix[0, 0], changed_matrix)

    def test_aligns_on_simulation_index_and_preserves_consistency_report(
        self, tmp_path: Path
    ) -> None:
        """
        Align matching simulations and report missing or extra source records.

        tmp_path:
            Pytest temporary directory used to create partially mismatched parameter
            records and Touchstone files.
        """
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
        """
        Reject out-of-range port pairs and absent requested frequencies.

        tmp_path:
            Pytest temporary directory used to build a valid baseline dataset before
            invalid requests are exercised.
        """
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
        """
        Reject mismatched frequency grids and non-finite dB responses.

        tmp_path:
            Pytest temporary directory used to create independent raw-data trees for the
            grid-mismatch and non-finite-response cases.
        """
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

    def test_rejects_unexpected_touchstone_port_count(self, tmp_path: Path) -> None:
        """
        Reject Touchstone files whose network size differs from config.

        tmp_path:
            Pytest temporary directory used to build the synthetic raw-data tree before
            the network reader is patched.
        """
        parameters, raw_data = _create_mock_rawdata(
            tmp_path,
            parameter_indices=[0],
            curves={0: ([1.0], [0.8])},
        )

        class MismatchedNetwork:
            """Minimal stand-in for a Touchstone network with the wrong port count."""

            nports = 1

        with patch(
            "sparam_surrogate.data.s_parameter_dataset.rf.Network",
            return_value=MismatchedNetwork(),
        ):
            with pytest.raises(ValueError, match="expected 2"):
                SParameterDataset.from_touchstones(parameters, raw_data, [(1, 1)])

    def test_cache_round_trip_and_explicit_rebuild(self, tmp_path: Path) -> None:
        """
        Reuse cached responses unless an explicit rebuild is requested.

        tmp_path:
            Pytest temporary directory used to build the source Touchstone and interim
            cache files.
        """
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
