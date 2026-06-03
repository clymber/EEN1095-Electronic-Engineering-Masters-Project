"""
Unit tests for :class:`PcbFeatureTransformer`.
"""

import numpy as np
import pandas as pd
import pytest

from sparam_surrogate.data import PcbFeatureTransformer, PcbParameters


def _parameters() -> PcbParameters:
    """
    Return a small parameter table with stable simulation indices.
    """
    return PcbParameters(
        pd.DataFrame(
            {
                "SIMU_INDEX": [10, 20, 30, 40],
                "EPS": [1.0, 3.0, 5.0, 7.0],
                "PITCH": [10.0, 30.0, 50.0, 70.0],
            }
        )
    )


class TestPcbFeatureTransformer:
    """
    Unit tests for design-frequency feature construction and scaling.
    """

    def test_expands_design_parameters_and_frequency_grid_in_design_major_order(
        self,
    ) -> None:
        """
        Repeat each design row across all frequencies before the next design.
        """
        transformer = PcbFeatureTransformer(
            feature_columns=["EPS", "PITCH"],
            scale=False,
        )

        result = transformer.fit_transform(
            _parameters(),
            frequencies_ghz=[1.0, 2.0, 3.0],
            split_labels=["train", "train", "val", "test"],
        )

        expected_X = np.array(
            [
                [1.0, 10.0, 1.0],
                [1.0, 10.0, 2.0],
                [1.0, 10.0, 3.0],
                [3.0, 30.0, 1.0],
                [3.0, 30.0, 2.0],
                [3.0, 30.0, 3.0],
                [5.0, 50.0, 1.0],
                [5.0, 50.0, 2.0],
                [5.0, 50.0, 3.0],
                [7.0, 70.0, 1.0],
                [7.0, 70.0, 2.0],
                [7.0, 70.0, 3.0],
            ],
            dtype=float,
        )

        assert result.X.shape == (12, 3)
        np.testing.assert_allclose(result.X, expected_X)
        np.testing.assert_array_equal(
            result.simulation_indices,
            np.array([10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40]),
        )
        np.testing.assert_allclose(
            result.frequencies_ghz,
            np.array([1.0, 2.0, 3.0] * 4),
        )

    def test_uses_train_rows_only_for_scaling_statistics(self) -> None:
        """
        Fit scaling statistics on train rows and apply them to all splits.
        """
        transformer = PcbFeatureTransformer(feature_columns=["EPS", "PITCH"])

        result = transformer.fit_transform(
            _parameters(),
            frequencies_ghz=[100.0, 200.0],
            split_labels=["train", "train", "val", "test"],
        )

        np.testing.assert_allclose(transformer.mean_, [2.0, 20.0, 150.0])
        np.testing.assert_allclose(transformer.scale_, [1.0, 10.0, 50.0])
        np.testing.assert_allclose(
            result.X,
            np.array(
                [
                    [-1.0, -1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 1.0],
                    [3.0, 3.0, -1.0],
                    [3.0, 3.0, 1.0],
                    [5.0, 5.0, -1.0],
                    [5.0, 5.0, 1.0],
                ],
                dtype=float,
            ),
        )

    def test_transform_reuses_fitted_scaling_statistics(self) -> None:
        """
        Transform data with the statistics learned during ``fit_transform``.
        """
        transformer = PcbFeatureTransformer(feature_columns=["EPS", "PITCH"])
        transformer.fit_transform(
            _parameters(),
            frequencies_ghz=[100.0, 200.0],
            split_labels=["train", "train", "val", "test"],
        )

        transformed = transformer.transform(
            PcbParameters(
                pd.DataFrame(
                    {
                        "SIMU_INDEX": [50],
                        "EPS": [4.0],
                        "PITCH": [40.0],
                    }
                )
            ),
            frequencies_ghz=[100.0, 200.0],
        )

        np.testing.assert_allclose(
            transformed.X,
            np.array([[2.0, 2.0, -1.0], [2.0, 2.0, 1.0]], dtype=float),
        )
        np.testing.assert_array_equal(transformed.simulation_indices, [50, 50])
        np.testing.assert_allclose(transformed.frequencies_ghz, [100.0, 200.0])

    def test_feature_names_include_frequency_column(self) -> None:
        """
        Preserve selected feature order and append the frequency feature.
        """
        transformer = PcbFeatureTransformer(
            feature_columns=["PITCH", "EPS"],
            frequency_column_name="FREQ_GHZ",
            scale=False,
        )

        result = transformer.fit_transform(
            _parameters(),
            frequencies_ghz=[1.0],
            split_labels=["train", "train", "val", "test"],
        )

        assert result.feature_names == ("PITCH", "EPS", "FREQ_GHZ")
        assert transformer.feature_names_ == ("PITCH", "EPS", "FREQ_GHZ")

    def test_rejects_missing_feature_columns(self) -> None:
        """
        Fail clearly when a requested parameter feature is unavailable.
        """
        transformer = PcbFeatureTransformer(feature_columns=["EPS", "TRACE_LEN"])

        with pytest.raises(ValueError, match="Missing feature columns"):
            transformer.fit_transform(
                _parameters(),
                frequencies_ghz=[1.0],
                split_labels=["train", "train", "val", "test"],
            )

    def test_rejects_non_finite_feature_or_frequency_values(self) -> None:
        """
        Reject invalid numeric values before fitting scaler statistics.
        """
        with pytest.raises(ValueError, match="non-finite feature"):
            PcbFeatureTransformer(feature_columns=["EPS", "PITCH"]).fit_transform(
                PcbParameters(
                    pd.DataFrame(
                        {
                            "SIMU_INDEX": [10, 20],
                            "EPS": [1.0, np.nan],
                            "PITCH": [10.0, 20.0],
                        }
                    )
                ),
                frequencies_ghz=[1.0],
                split_labels=["train", "val"],
            )

        with pytest.raises(ValueError, match="non-finite frequency"):
            PcbFeatureTransformer(feature_columns=["EPS", "PITCH"]).fit_transform(
                _parameters(),
                frequencies_ghz=[1.0, np.inf],
                split_labels=["train", "train", "val", "test"],
            )
