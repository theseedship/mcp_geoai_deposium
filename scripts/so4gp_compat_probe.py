#!/usr/bin/env python3
"""Probe the so4gp 1.1.2 CPU contract, retaining legacy API diagnostics.

Required GRAANK and public TGRAANK calls use CPU and deterministic Apriori;
temporal mining explicitly selects all transformations rather than AMI.
Legacy TGrad and optional optimizer diagnostics remain available. This tiny
fixture checks API/output compatibility, not scientific skill or calibration.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import io
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO


@dataclass(frozen=True)
class AlgorithmProbe:
    name: str
    class_name: str
    required: bool = False
    temporal: bool = False
    stochastic: bool = False


ALGORITHM_PROBES: tuple[AlgorithmProbe, ...] = (
    AlgorithmProbe("graank", "GRAANK", required=True),
    AlgorithmProbe("tgrad", "TGrad", required=True, temporal=True),
    AlgorithmProbe("aco", "AntGRAANK", stochastic=True),
    AlgorithmProbe("ga", "GeneticGRAANK", stochastic=True),
    AlgorithmProbe("pso", "ParticleGRAANK", stochastic=True),
)

REQUIRED_OUTPUT_SHAPE = "patterns_json"


def _dist_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _demo_frame(*, temporal: bool):
    import pandas as pd

    if temporal:
        return pd.DataFrame(
            [
                ["2020-01-01", 10, 0.30, 25],
                ["2020-01-08", 20, 0.31, 24],
                ["2020-01-15", 35, 0.40, 22],
                ["2020-01-22", 50, 0.55, 20],
                ["2020-01-29", 40, 0.62, 21],
                ["2020-02-05", 60, 0.58, 19],
                ["2020-02-12", 80, 0.70, 17],
                ["2020-02-19", 70, 0.85, 18],
                ["2020-02-26", 90, 0.80, 16],
                ["2020-03-04", 100, 0.92, 15],
            ],
            columns=["date", "precipitation", "ndvi", "temperature"],
        )

    return pd.DataFrame(
        {
            "precipitation": [45, 52, 38, 25, 18, 42, 55, 48, 62, 70],
            "temperature": [8, 9, 11, 14, 17, 15, 13, 12, 10, 9],
            "ndvi": [0.72, 0.68, 0.65, 0.58, 0.52, 0.61, 0.69, 0.74, 0.78, 0.83],
        }
    )


def _signature_accepts_kwargs(cls: type) -> bool:
    try:
        params = inspect.signature(cls.__init__).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params)


def _filter_supported_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    if _signature_accepts_kwargs(cls):
        return kwargs
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def _constructor_candidates(
    cls: type,
    frame: Any,
    *,
    probe: AlgorithmProbe,
    min_support: float,
    min_rep: float,
    encode_equal: bool,
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    kwargs: dict[str, Any] = {
        "data_source": frame,
        "min_sup": min_support,
        "eq": encode_equal,
        "device": "cpu",
    }
    if probe.temporal:
        kwargs.update({"target_col": 2, "min_rep": min_rep})

    filtered = _filter_supported_kwargs(cls, kwargs)
    positional_kwargs = _filter_supported_kwargs(
        cls, {key: value for key, value in kwargs.items() if key != "data_source"}
    )

    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((), filtered),
        ((frame,), positional_kwargs),
    ]
    if probe.temporal:
        candidates.extend(
            [
                ((frame,), {"min_sup": min_support, "target_col": 2, "min_rep": min_rep}),
                ((frame, min_support), {"target_col": 2, "min_rep": min_rep}),
            ]
        )
    else:
        candidates.extend(
            [
                ((frame,), {"min_sup": min_support, "eq": encode_equal}),
                ((frame, min_support), {"eq": encode_equal}),
            ]
        )
    return candidates


def _instantiate(
    cls: type,
    frame: Any,
    *,
    probe: AlgorithmProbe,
    min_support: float,
    min_rep: float,
    encode_equal: bool,
) -> Any:
    errors: list[str] = []
    for args, kwargs in _constructor_candidates(
        cls,
        frame,
        probe=probe,
        min_support=min_support,
        min_rep=min_rep,
        encode_equal=encode_equal,
    ):
        try:
            return cls(*args, **kwargs)
        except TypeError as exc:
            errors.append(str(exc))
            continue
    joined = " | ".join(errors[-3:])
    raise TypeError(
        f"could not instantiate {probe.class_name}; attempted signatures failed: {joined}"
    )


def _discover(miner: Any, *, probe: AlgorithmProbe) -> Any:
    if probe.temporal:
        method = getattr(miner, "discover_tgp", None)
        if method is not None:
            try:
                return method(parallel=False)
            except TypeError:
                return method()
        method = getattr(miner, "discover", None)
        if method is None:
            raise AttributeError(f"{probe.class_name}.discover() is missing")
        return method(
            target_col=2,
            transformations="all",
            search_algorithm="apriori",
            max_iteration=3,
            save_results=False,
        )

    method = getattr(miner, "discover", None)
    if method is None:
        raise AttributeError(f"{probe.class_name}.discover() is missing")
    params = inspect.signature(method).parameters
    choices = {"search_type": "apriori", "target_col": None, "save_results": False}
    if probe.stochastic:
        choices["max_iteration"] = 3
    return method(**{key: value for key, value in choices.items() if key in params})


def _json_payload(raw: Any) -> Any:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def classify_output(raw: Any) -> dict[str, Any]:
    """Return a compact, stable description of a so4gp output payload."""
    raw_type = type(raw).__name__
    try:
        payload = _json_payload(raw)
    except (TypeError, json.JSONDecodeError):
        preview = str(raw)
        return {
            "raw_type": raw_type,
            "shape": "text",
            "n_patterns": None,
            "keys": [],
            "algorithm": None,
            "preview": preview[:300],
        }

    if isinstance(payload, dict):
        keys = sorted(str(key) for key in payload)
        if isinstance(payload.get("Patterns"), list):
            shape = "patterns_json"
            n_patterns = len(payload["Patterns"])
        elif isinstance(payload.get("Best Patterns"), list):
            shape = "best_patterns_json"
            n_patterns = len(payload["Best Patterns"])
        else:
            shape = "json_object"
            n_patterns = None
        return {
            "raw_type": raw_type,
            "shape": shape,
            "n_patterns": n_patterns,
            "keys": keys,
            "algorithm": payload.get("Algorithm"),
            "preview": None,
        }

    if isinstance(payload, list):
        return {
            "raw_type": raw_type,
            "shape": "json_list",
            "n_patterns": len(payload),
            "keys": [],
            "algorithm": None,
            "preview": None,
        }

    return {
        "raw_type": raw_type,
        "shape": type(payload).__name__,
        "n_patterns": None,
        "keys": [],
        "algorithm": None,
        "preview": str(payload)[:300],
    }


def _run_inside_tempdir(fn):
    previous = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="so4gp_compat_") as tmp:
        os.chdir(tmp)
        try:
            return fn()
        finally:
            os.chdir(previous)


def _one_algorithm_run(
    algorithms_module: ModuleType | Any,
    probe: AlgorithmProbe,
    *,
    min_support: float,
    min_rep: float,
    encode_equal: bool,
) -> dict[str, Any]:
    class_name = probe.class_name
    cls = getattr(algorithms_module, class_name, None)
    if cls is None and probe.temporal and class_name == "TGrad":
        class_name = "TGRAANK"
        cls = getattr(algorithms_module, class_name, None)
    if cls is None:
        return {
            "status": "missing",
            "class_name": class_name,
            "elapsed_seconds": 0.0,
            "output": None,
            "error": f"{class_name} not found under so4gp.algorithms",
        }

    frame = _demo_frame(temporal=probe.temporal)
    started = time.perf_counter()
    captured_logs = io.StringIO()
    try:
        with redirect_stdout(captured_logs), redirect_stderr(captured_logs):
            raw = _run_inside_tempdir(
                lambda: _discover(
                    _instantiate(
                        cls,
                        frame,
                        probe=probe,
                        min_support=min_support,
                        min_rep=min_rep,
                        encode_equal=encode_equal,
                    ),
                    probe=probe,
                )
            )
        payload = _json_payload(raw)
        if isinstance(payload, dict) and payload.get("Error"):
            raise ValueError(f"so4gp discovery failed: {payload['Error']}")
        return {
            "status": "ok",
            "class_name": class_name,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "output": classify_output(raw),
            "logs": captured_logs.getvalue()[:1000],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic probe must report every failure.
        return {
            "status": "error",
            "class_name": class_name,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "output": None,
            "logs": captured_logs.getvalue()[:1000],
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }


def _summarize_algorithm(probe: AlgorithmProbe, runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok_runs = [run for run in runs if run["status"] == "ok"]
    shapes = sorted({run["output"]["shape"] for run in ok_runs if run.get("output")})
    required_shape_ok_runs = [
        run for run in ok_runs if (run.get("output") or {}).get("shape") == REQUIRED_OUTPUT_SHAPE
    ]
    if probe.required:
        contract_ok = len(required_shape_ok_runs) == len(runs)
        if contract_ok:
            status = "ok"
        elif ok_runs:
            status = "contract_mismatch"
        else:
            status = runs[-1]["status"]
    else:
        contract_ok = True
        if len(ok_runs) == len(runs):
            status = "ok"
        elif ok_runs:
            status = "degraded"
        else:
            status = runs[-1]["status"]
    return {
        "name": probe.name,
        "class_name": runs[0]["class_name"],
        "required": probe.required,
        "temporal": probe.temporal,
        "stochastic": probe.stochastic,
        "status": status,
        "contract_ok": contract_ok,
        "required_shape": REQUIRED_OUTPUT_SHAPE if probe.required else None,
        "required_shape_ok_runs": len(required_shape_ok_runs) if probe.required else None,
        "ok_runs": len(ok_runs),
        "total_runs": len(runs),
        "output_shapes": shapes,
        "runs": runs,
    }


def run_probe_suite(
    *,
    algorithms_module: ModuleType | Any | None = None,
    min_support: float = 0.4,
    min_rep: float = 0.5,
    encode_equal: bool = False,
    repeats: int = 1,
    include_optimizers: bool = True,
) -> dict[str, Any]:
    if algorithms_module is None:
        from so4gp import algorithms as algorithms_module

    selected = [probe for probe in ALGORITHM_PROBES if include_optimizers or not probe.stochastic]
    algorithms: list[dict[str, Any]] = []
    for probe in selected:
        run_count = repeats if probe.stochastic else 1
        runs = [
            _one_algorithm_run(
                algorithms_module,
                probe,
                min_support=min_support,
                min_rep=min_rep,
                encode_equal=encode_equal,
            )
            for _ in range(max(1, run_count))
        ]
        algorithms.append(_summarize_algorithm(probe, runs))

    required_summaries = [item for item in algorithms if item["required"]]
    required_passed = bool(required_summaries) and all(
        item["contract_ok"] for item in required_summaries
    )
    optimizer_status = {item["name"]: item["status"] for item in algorithms if item["stochastic"]}
    return {
        "schema_version": 1,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            "so4gp": _dist_version("so4gp"),
            "pandas": _dist_version("pandas"),
            "numpy": _dist_version("numpy"),
            "scikit-learn": _dist_version("scikit-learn"),
            "scikit-fuzzy": _dist_version("scikit-fuzzy"),
        },
        "parameters": {
            "min_support": min_support,
            "min_rep": min_rep,
            "encode_equal": encode_equal,
            "repeats": repeats,
            "include_optimizers": include_optimizers,
        },
        "required_passed": required_passed,
        "optimizer_status": optimizer_status,
        "algorithms": algorithms,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-support", type=float, default=0.4)
    parser.add_argument("--min-rep", type=float, default=0.5)
    parser.add_argument("--encode-equal", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--skip-optimizers", action="store_true")
    parser.add_argument("--fail-on-required", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    algorithms_module: ModuleType | Any | None = None,
    stdout: TextIO | None = None,
) -> int:
    args = _parse_args(argv)
    try:
        result = run_probe_suite(
            algorithms_module=algorithms_module,
            min_support=args.min_support,
            min_rep=args.min_rep,
            encode_equal=args.encode_equal,
            repeats=args.repeats,
            include_optimizers=not args.skip_optimizers,
        )
    except ImportError as exc:
        result = {
            "schema_version": 1,
            "required_passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "python": {"version": platform.python_version(), "executable": sys.executable},
            "packages": {"so4gp": _dist_version("so4gp")},
        }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    (stdout or sys.stdout).write(text + "\n")

    if args.fail_on_required and not result.get("required_passed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
