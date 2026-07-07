"""
Tests for output naming helpers.
"""

from datetime import datetime, timezone

import pytest

from sparam_surrogate.outputs.naming import (
    created_at_from_run_id,
    format_run_timestamp,
    get_run_id,
    slugify_model_name,
)


class TestOutputNaming:
    """
    Tests for shared output naming conventions.
    """

    def test_slugify_model_name_returns_stable_slug(self) -> None:
        """
        Model names become lowercase underscore-separated slugs.
        """
        assert slugify_model_name("Scalar Ridge") == "scalar_ridge"
        assert slugify_model_name("Polynomial/Neural MLP") == (
            "polynomial_neural_mlp"
        )

    def test_slugify_model_name_rejects_blank_names(self) -> None:
        """
        Model slugs require at least one alphanumeric value.
        """
        with pytest.raises(ValueError, match="alphanumeric"):
            slugify_model_name(" -- ")

    def test_format_run_timestamp_accepts_datetime_and_timestamp_string(
        self,
    ) -> None:
        """
        Run timestamps use the compact UTC directory format.
        """
        timestamp = datetime(2026, 7, 5, 15, 30, tzinfo=timezone.utc)

        assert format_run_timestamp(timestamp) == "20260705T153000Z"
        assert format_run_timestamp("20260705T153000Z") == "20260705T153000Z"

    def test_format_run_timestamp_rejects_invalid_strings(self) -> None:
        """
        Timestamp strings must already match the run ID timestamp form.
        """
        with pytest.raises(ValueError, match="YYYYMMDDTHHMMSSZ"):
            format_run_timestamp("2026-07-05")

    def test_get_run_id_uses_shared_timestamp_and_slug_format(self) -> None:
        """
        Run IDs combine the formatted timestamp and model-name slug.
        """
        run_id = get_run_id(
            "Scalar Ridge",
            timestamp=datetime(2026, 7, 5, 15, 30, tzinfo=timezone.utc),
        )

        assert run_id == "20260705T153000Z_scalar_ridge"

    def test_created_at_from_run_id_returns_iso_timestamp(self) -> None:
        """
        Timestamped run IDs expose an ISO-style creation timestamp.
        """
        assert created_at_from_run_id("20260705T153000Z_scalar_ridge") == (
            "2026-07-05T15:30:00Z"
        )
        assert created_at_from_run_id("scalar_ridge") is None
