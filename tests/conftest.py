"""Shared pytest fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.config import settings


@pytest.fixture
def allow_local_parquet_paths(monkeypatch):
    """Opt a test module into caller-controlled Parquet paths under the temp root."""
    root = Path(tempfile.gettempdir()).resolve()
    monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
    monkeypatch.setattr(settings, "LOCAL_PARQUET_SOURCE_ROOT", root.as_posix())
    return root
