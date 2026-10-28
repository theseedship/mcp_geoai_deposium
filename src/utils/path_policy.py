"""Deployment policy for caller-controlled local Parquet paths."""

from __future__ import annotations

from pathlib import Path

from src.config import settings


def resolve_local_parquet_path(value: str, *, role: str) -> Path:
    """Resolve a trusted local path or raise when local access is not confined."""
    if not settings.ALLOW_LOCAL_PARQUET_SOURCE:
        raise ValueError(f"local {role} is disabled by ALLOW_LOCAL_PARQUET_SOURCE")

    raw_root = settings.LOCAL_PARQUET_SOURCE_ROOT.strip()
    if not raw_root:
        raise ValueError(
            "LOCAL_PARQUET_SOURCE_ROOT is required when local parquet paths are enabled"
        )

    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        raise ValueError("LOCAL_PARQUET_SOURCE_ROOT must be an absolute path")
    root = root.resolve(strict=False)
    if root == Path(root.anchor):
        raise ValueError("LOCAL_PARQUET_SOURCE_ROOT must not be the filesystem root")

    path = Path(value).expanduser().resolve(strict=False)
    if not path.is_relative_to(root):
        raise ValueError(f"local {role} must be under LOCAL_PARQUET_SOURCE_ROOT ({root})")
    return path
