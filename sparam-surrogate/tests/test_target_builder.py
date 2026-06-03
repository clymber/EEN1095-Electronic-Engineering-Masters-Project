"""
Unit tests for :class:`TargetBuilder`.
"""

import numpy as np
import pandas as pd
import pytest

from sparam_surrogate.data import (
    PcbFeatureTransformer,
    PcbParameters,
    SParameterDataset,
    TargetBuilder,
)


def _responses() -> SParameterDataset:
    """
    Return a tiny aligned response dataset with two designs and two frequencies.
    """
    full_s_matrix = np.array(
        [
            [
                [[1.0 + 0.1j, 2.0 + 0.2j], [3.0 + 0.3j, 4.0 + 0.4j]],
                [[5.0 + 0.5j, 6.0 + 0.6j], [7.0 + 0.7j, 8.0 + 0.8j]],
            ],
            [
                [[9.0 + 0.9j, 10.0 + 1.0j], [11.0 + 1.1j, 12.0 + 1.2j]],
                [[13.0 + 1.3j, 14.0 + 1.4j], [15.0 + 1.5j, 16.0 + 1.6j]],
            ],
        ],
        dtype=complex,
    )
    return SParameterDataset(
        simulation_indices=[10, 20],
        frequencies_ghz=[1.0, 2.0],
        port_pairs=[(2, 1)],
        through_s_db=np.array([[[30.0], [70.0]], [[110.0], [150.0]]]),
        full_s_matrix=full_s_matrix,
    )


class TestTargetBuilder:
    """
    Unit tests for scalar and full-S-matrix target construction.
    """

    def test_builds_scalar_target_in_design_frequency_order(self) -> None:
        """
        Scalar targets are flattened in design-major, frequency-minor order.
        """
        target = TargetBuilder.build_scalar(_responses(), pair=(2, 1))

        assert target.target.shape == (4, 1)
        assert target.target_names == ("S2_1_DB",)
        np.testing.assert_allclose(target.target[:, 0], [30.0, 70.0, 110.0, 150.0])
        np.testing.assert_array_equal(target.simulation_indices, [10, 10, 20, 20])
        np.testing.assert_allclose(target.frequencies_ghz, [1.0, 2.0, 1.0, 2.0])

    def test_rejects_unavailable_scalar_port_pair(self) -> None:
        """
        Scalar targets must refer to a response path present in the dataset.
        """
        with pytest.raises(ValueError, match="not available"):
            TargetBuilder.build_scalar(_responses(), pair=(1, 2))

    def test_builds_full_smatrix_target_with_interleaved_real_imag_columns(
        self,
    ) -> None:
        """
        Full S-matrices flatten row-major with real/imag adjacent per entry.
        """
        target = TargetBuilder.build_full_smatrix(_responses())

        assert target.target.shape == (4, 8)
        assert target.target_names == (
            "REAL_S1_1",
            "IMAG_S1_1",
            "REAL_S1_2",
            "IMAG_S1_2",
            "REAL_S2_1",
            "IMAG_S2_1",
            "REAL_S2_2",
            "IMAG_S2_2",
        )
        np.testing.assert_allclose(
            target.target[0],
            [1.0, 0.1, 2.0, 0.2, 3.0, 0.3, 4.0, 0.4],
        )
        np.testing.assert_allclose(
            target.target[-1],
            [13.0, 1.3, 14.0, 1.4, 15.0, 1.5, 16.0, 1.6],
        )

    def test_target_row_metadata_matches_pcb_feature_transformer_order(self) -> None:
        """
        Target rows align with design-major feature rows.
        """
        parameters = PcbParameters(
            pd.DataFrame(
                {
                    "SIMU_INDEX": [10, 20],
                    "EPS": [3.1, 3.4],
                }
            )
        )
        features = PcbFeatureTransformer(
            feature_columns=["EPS"],
            scale=False,
        ).fit_transform(
            parameters,
            frequencies_ghz=[1.0, 2.0],
            split_labels=["train", "test"],
        )
        target = TargetBuilder.build_scalar(_responses(), pair=(2, 1))

        np.testing.assert_array_equal(
            target.simulation_indices,
            features.simulation_indices,
        )
        np.testing.assert_allclose(target.frequencies_ghz, features.frequencies_ghz)
