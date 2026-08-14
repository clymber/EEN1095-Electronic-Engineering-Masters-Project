"""
Naming helpers for output run IDs and registry keys.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

# Required run timestamp form: YYYYMMDDTHHMMSSZ.
TIMESTAMP_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")

# Timestamp prefix extracted from run IDs such as YYYYMMDDTHHMMSSZ_model_name.
RUN_ID_TIMESTAMP_PATTERN = re.compile(r"^(\d{8}T\d{6}Z)_")


def slugify_model_name(model_name: str) -> str:
    """
    Return a stable lowercase slug for a model name.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("model_name must contain at least one alphanumeric value.")
    return slug


def format_run_timestamp(timestamp: datetime | str | None) -> str:
    """
    Return a UTC timestamp formatted as ``YYYYMMDDTHHMMSSZ``.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    if isinstance(timestamp, str):
        if not TIMESTAMP_PATTERN.match(timestamp):
            raise ValueError("timestamp must match YYYYMMDDTHHMMSSZ.")
        return timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def get_run_id(
    model_name: str,
    *,
    timestamp: datetime | str | None = None,
) -> str:
    """
    Return the timestamped run ID for a model name.
    """
    return f"{format_run_timestamp(timestamp)}_{slugify_model_name(model_name)}"


def created_at_from_run_id(run_id: str) -> str | None:
    """
    Return an ISO timestamp derived from a timestamped run ID.
    """
    match = RUN_ID_TIMESTAMP_PATTERN.match(run_id)
    if match is None:
        return None
    created_at = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
    created_at = created_at.replace(tzinfo=timezone.utc)
    return created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
