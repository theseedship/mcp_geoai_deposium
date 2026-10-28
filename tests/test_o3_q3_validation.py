"""Network-free tests for scripts/o3_q3_validation.py."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "o3_q3_validation.py"
_spec = importlib.util.spec_from_file_location("o3_q3_validation", _SCRIPT_PATH)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


def _args(*values: str):
    return script.build_parser().parse_args(list(values))


def test_parser_defaults_to_full_panel_q3_both_algorithms():
    args = _args()
    assert args.zone_set == "full-panel"
    assert args.window_label == ["2023_07", "2023_08", "2023_09"]
    assert args.algorithm == ["graank", "tgrad"]
    assert args.source_template == "climate/holdout/unified/{window_label}/{zone_id}.parquet"


def test_preflight_excludes_zones_with_too_few_rows():
    def fake_loader(uri: str, *_args):
        zone_id = uri.rsplit("/", 1)[-1].removesuffix(".parquet")
        rows = 1 if zone_id == "kerguelen" else 2
        return pd.DataFrame({"ndvi_mean": list(range(rows)), "temperature_2m": list(range(rows))})

    eligible, rows = script._preflight_sources(
        zone_ids=["montpellier", "kerguelen"],
        window_labels=["2023_07"],
        source_template=script.DEFAULT_SOURCE_TEMPLATE,
        source_kind="parquet",
        columns=["ndvi_mean", "temperature_2m"],
        order_by="date",
        min_rows_per_zone=2,
        frame_loader=fake_loader,
    )

    assert eligible == ["montpellier"]
    assert rows[0]["status"] == "eligible"
    assert rows[1]["status"] == "excluded_insufficient_rows"


def test_run_async_invokes_cv_with_eligible_zones_and_caveats(monkeypatch):
    monkeypatch.setattr(
        script,
        "_resolve_zone_ids",
        lambda _args: ["z1", "z2", "z3", "z4", "z5", "kerguelen"],
    )
    calls: list[dict[str, Any]] = []

    def fake_loader(uri: str, *_args):
        zone_id = uri.rsplit("/", 1)[-1].removesuffix(".parquet")
        label = uri.rsplit("/", 2)[-2]
        rows = 1 if zone_id == "kerguelen" and label == "2023_08" else 0
        if zone_id != "kerguelen":
            rows = 3
        return pd.DataFrame({"ndvi_mean": list(range(rows)), "temperature_2m": list(range(rows))})

    async def fake_tool(**kwargs):
        calls.append(kwargs)
        return {
            "patterns": [{"pattern_id": 0, "pattern": "p"}],
            "folds": [],
            "output": f"s3://deposium/{kwargs['output_key']}",
            "statistics": {"n_candidate_patterns": 1, "n_validated_patterns": 1},
            "metadata": {"truncated_sources": kwargs["truncated_sources"]},
        }

    args = _args("--algorithm", "graank", "--output-dir", "/tmp/unused")
    summary = asyncio.run(script.run_async(args, tool_runner=fake_tool, frame_loader=fake_loader))

    assert summary["eligible_zone_ids"] == ["z1", "z2", "z3", "z4", "z5"]
    assert summary["excluded_zone_ids"] == ["kerguelen"]
    assert summary["truncated_sources"]["kerguelen"] == ["insufficient_q3_rows"]
    assert calls[0]["zone_ids"] == ["z1", "z2", "z3", "z4", "z5"]
    assert calls[0]["output_key"] == "climate/o3/holdout/q3_2023/graank_validated.parquet"


def test_run_writes_summary_and_full_results(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(script, "_resolve_zone_ids", lambda _args: ["z1", "z2", "z3", "z4", "z5"])

    def fake_loader(_uri: str, *_args):
        return pd.DataFrame({"ndvi_mean": [1, 2], "temperature_2m": [3, 4]})

    async def fake_run_async(args, **_kwargs):
        return {
            "kind": "o3_q3_validation_summary",
            "zone_ids": ["z1", "z2", "z3", "z4", "z5"],
            "eligible_zone_ids": ["z1", "z2", "z3", "z4", "z5"],
            "excluded_zone_ids": [],
            "truncated_sources": {},
            "preflight": [],
            "runs": {
                "graank": {
                    "patterns": [{"pattern_id": 0}, {"pattern_id": 1}],
                    "output": "s3://deposium/climate/o3/holdout/q3_2023/graank_validated.parquet",
                    "statistics": {"n_candidate_patterns": 2, "n_validated_patterns": 2},
                }
            },
        }

    monkeypatch.setattr(script, "run_async", fake_run_async)
    args = _args("--algorithm", "graank", "--output-dir", tmp_path.as_posix(), "--top-n", "1")
    assert script.run(args) == 0

    summary = json.loads((tmp_path / "o3-q3-validation-summary.json").read_text())
    full_result = json.loads((tmp_path / "graank-result.json").read_text())
    assert summary["runs"]["graank"]["n_patterns"] == 2
    assert len(summary["runs"]["graank"]["top_patterns"]) == 1
    assert "patterns" not in summary["runs"]["graank"]
    assert len(full_result["patterns"]) == 2
    assert "OK=True" in capsys.readouterr().out
