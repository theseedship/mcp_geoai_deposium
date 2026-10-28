"""Tests for the mine_patterns CLI (FIERCE DoD O5, issue #23)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.cli.mine_patterns import build_parser, run
from src.patterns.library_schema import PatternLibrary

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "scripts" / "sample" / "tabular_data.parquet"
_HAS_SO4GP = importlib.util.find_spec("so4gp") is not None


def test_parser_defaults_graank():
    args = build_parser().parse_args(["--source", "x.parquet", "--zone-id", "montpellier"])
    assert args.algorithm == "graank" and args.zone_id == "montpellier"


def test_tgrad_requires_target_and_order_by_fails_fast(capsys):
    # No I/O, no so4gp needed — the guard runs before any import/load.
    args = build_parser().parse_args(
        ["--source", "x.parquet", "--zone-id", "z", "--algorithm", "tgrad"]
    )
    assert run(args) == 2
    assert "requires --target-column and --order-by" in capsys.readouterr().err


def test_min_support_out_of_range_fails_fast(capsys):
    # Threshold validation runs before any I/O / so4gp import.
    args = build_parser().parse_args(
        ["--source", "x.parquet", "--zone-id", "z", "--min-support", "1.5"]
    )
    assert run(args) == 2
    assert "min-support must be in [0, 1]" in capsys.readouterr().err


@pytest.mark.skipif(not _HAS_SO4GP, reason="needs the 'patterns' extra (so4gp)")
def test_graank_smoke_writes_valid_library(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # belt and braces: never litter pytest's cwd if confinement breaks
    out = tmp_path / "patterns.json"
    args = build_parser().parse_args(
        [
            "--source",
            str(SAMPLE),
            "--zone-id",
            "montpellier",
            "--algorithm",
            "graank",
            "--min-support",
            "0.3",
            "-o",
            str(out),
        ]
    )
    assert run(args) == 0
    # the emitted JSON validates against the frozen exchange schema
    lib = PatternLibrary.model_validate_json(out.read_text())
    assert lib.zone_id == "montpellier" and lib.algorithm == "graank"
    for p in lib.patterns:
        assert p.support >= 0.0 and p.confidence is None
    # round-trips as JSON too
    assert isinstance(json.loads(out.read_text())["patterns"], list)


@pytest.mark.skipif(not _HAS_SO4GP, reason="needs the 'patterns' extra (so4gp)")
@pytest.mark.parametrize(
    "extra_args",
    [
        ["--algorithm", "graank", "--min-support", "0.3"],
        [
            "--algorithm",
            "tgrad",
            "--min-support",
            "0.3",
            "--order-by",
            "date",
            "--target-column",
            "ndvi_mean",
        ],
    ],
    ids=["graank", "tgrad"],
)
def test_mining_leaves_caller_cwd_clean_and_keeps_relative_paths(tmp_path, monkeypatch, extra_args):
    # #214: so4gp writes GRAANK_* / TGrad_* into the cwd with no off switch. The CLI must
    # confine that, while a relative --source / -o still resolve against the CALLER's cwd.
    caller = tmp_path / "caller"
    caller.mkdir()
    (caller / "in.parquet").write_bytes(SAMPLE.read_bytes())
    monkeypatch.chdir(caller)

    args = build_parser().parse_args(
        ["--source", "in.parquet", "--zone-id", "montpellier", *extra_args, "-o", "out.json"]
    )
    assert run(args) == 0

    assert sorted(p.name for p in caller.iterdir()) == ["in.parquet", "out.json"]
    assert Path.cwd() == caller
    lib = PatternLibrary.model_validate_json((caller / "out.json").read_text())
    assert lib.zone_id == "montpellier"
