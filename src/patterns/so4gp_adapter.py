"""Adapter around so4gp gradual pattern mining.

This is the ONLY module that imports so4gp. The default runtime pins the real
1.1.2 public CPU API (verified by ``tests/test_so4gp_contract.py``) and normalizes so4gp's
output — which is a JSON *string*, not a dict — into plain Python objects.
The earlier 1.0.9 CPU replay keeps its exact-version scope. Legacy invocation
is retained for captured compatibility tests, separate from the default API.

Historical so4gp API (0.9.1, captured output):
    from so4gp.algorithms import GRAANK
    GRAANK(data_source=<pandas.DataFrame | csv path>, min_sup=0.5, eq=False)
    obj.discover()  -> JSON string:
        {"Algorithm": "GRAANK", "Run-time": "1.36 seconds",
         "Patterns": [["precip+, ndvi+", "sup=0.821", "density=0.821",
                       "avg_dev=2.913", "dispersion=1.586", "connect=1",
                       "singularity_scr=0.12"], ...],
         "Invalid Count": "11"}

Each pattern is a co-monotonic gradual itemset (a set of (attribute,
direction) items that rise/fall together across rows) — NOT an
antecedent->consequent association rule, and NOT temporal. See the ADR
docs/adr/0001-so4gp-pattern-mining-integration.md for why the
terra-cognita patterns schema (antecedent/consequent/lag_days) does not
map onto plain GRAANK and needs a temporal variant (so4gp TGrad/TGradAMI).
"""

from __future__ import annotations

import inspect
import json
import math
import re
from collections.abc import Callable

# Map of the algorithm key (as exposed by discover_patterns) to the so4gp
# class name. ONLY GRAANK is wired: it is the only variant whose discover()
# output shape (the "Patterns" payload) is verified by the contract test and
# handled by parse_discover_output. The optimization variants (AntGRAANK,
# GeneticGRAANK, ParticleGRAANK, HillClimbingGRAANK, RandomGRAANK) document a
# different "Best Patterns" result shape — they will be added once their
# output is contract-tested (FIERCE Palier 3, cross-zone + 4 algos).
SUPPORTED_ALGORITHMS = {
    "graank": "GRAANK",
}

# Temporal variants (FIERCE Palier 2). These mine *temporal* gradual patterns:
# a target gradual item plus other gradual items that lead/trail it by a time
# lag, e.g. "ndvi+, (precipitation+) +7.0 days". The public TGRAANK facade
# takes target_col at discovery; legacy TGrad takes it at construction.
# Both return lag-bearing "Patterns" and are wired separately from GRAANK.
# TGradAMI (auto target selection via mutual information) is deferred — its
# constructor still requires target_col in 0.9.1 and its lag-estimation path
# is not contract-tested yet.
SUPPORTED_TEMPORAL_ALGORITHMS = {
    "tgrad": "TGrad",
}

# so4gp's TimeDelay.format_time() renders the lag in the largest sensible unit
# (so4gp/gradual_patterns.py). Normalize every unit to days so downstream
# consumers compare a single scalar. Months/years use the average-length
# constants so4gp itself uses (2.628e6 s/month, derived 365.25 d/year).
_LAG_UNIT_TO_DAYS = {
    "seconds": 1.0 / 86400.0,
    "minutes": 1.0 / 1440.0,
    "hours": 1.0 / 24.0,
    "days": 1.0,
    "weeks": 7.0,
    "months": 2.628e6 / 86400.0,  # ≈30.42 days, matches so4gp's month constant
    "years": 365.25,
}

# A temporal gradual item: "(precipitation+) +7.0 days" / "(temp-) -2.0 weeks".
_TEMPORAL_ITEM_RE = re.compile(r"\(\s*([^)]+?)\s*\)\s*([+-])\s*([0-9.]+)\s*([A-Za-z]+)")


class So4gpNotInstalled(RuntimeError):
    """Raised when mining is attempted without the ``patterns`` extra."""


def _resolve_algorithm(algorithm: str, registry: dict[str, str], *, kind_label: str):
    """Resolve a so4gp algorithm class from ``registry``, failing LOUD if not installed.

    Never silently falls back to mock output — fabricated patterns feeding downstream
    alerts are a worse failure than an explicit error. ``kind_label`` ("algorithm" for the
    contemporaneous registry, "temporal algorithm" for the TGrad registry) is interpolated
    into the unsupported-key error so the two paths keep their distinct messages.
    """
    key = algorithm.lower()
    if key not in registry:
        raise ValueError(f"Unsupported {kind_label} '{algorithm}'. Supported: {sorted(registry)}")
    try:
        from so4gp import algorithms as so4gp_algorithms
    except ImportError as exc:  # pragma: no cover - exercised via contract test
        raise So4gpNotInstalled(
            "so4gp is not installed. Install the optional extra on Python "
            ">=3.14:  uv sync --extra patterns"
        ) from exc
    class_name = registry[key]
    if class_name == "TGrad" and not hasattr(so4gp_algorithms, class_name):
        # Public 1.x exposes the temporal facade as TGRAANK. Prefer the original
        # TGrad whenever present so the pinned legacy contract stays intact.
        class_name = "TGRAANK"
    return getattr(so4gp_algorithms, class_name)


def _parse_kv(token: str) -> tuple[str, float | str]:
    """Parse a 'key=value' metric token; value coerced to float when possible."""
    key, _, raw = token.partition("=")
    key = key.strip()
    raw = raw.strip()
    try:
        return key, float(raw)
    except ValueError:
        return key, raw


def _split_attr_dir(token: str) -> dict:
    """Split 'precip+' -> {attribute: 'precip', direction: '+'}.

    Direction is the trailing '+' (increasing) or '-' (decreasing). Column
    names are assumed not to end in '+'/'-' (true for the climate/NDVI
    schema). A token without a recognized direction keeps direction None so
    nothing is silently dropped.
    """
    token = token.strip()
    if token and token[-1] in "+-":
        return {"attribute": token[:-1], "direction": token[-1]}
    return {"attribute": token, "direction": None}


def _parse_gradual_items(pattern_str: str) -> list[dict]:
    """Parse 'precip+, temp-, ndvi+' into [{attribute, direction}, ...]."""
    items: list[dict] = []
    for token in pattern_str.split(","):
        if token.strip():
            items.append(_split_attr_dir(token))
    return items


def _lag_to_days(value: float, unit: str) -> float | None:
    """Normalize a so4gp time-lag (value + unit) to days; None if unit unknown."""
    factor = _LAG_UNIT_TO_DAYS.get(unit.strip().lower())
    return None if factor is None else value * factor


def _leading_target_token(pattern_str: str) -> str:
    """Return the leading target token — the text before the first TOP-LEVEL comma.

    Commas *inside* parentheses (e.g. a column name ``co2(ppm,avg)``) are ignored so the
    target name (and its trailing direction) is not truncated; the lagged items follow
    after that top-level comma.
    """
    depth = 0
    for i, ch in enumerate(pattern_str):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            return pattern_str[:i]
    return pattern_str


def _parse_temporal_items(pattern_str: str, *, target_name: str | None = None) -> list[dict]:
    """Parse a TGrad pattern string into target + lagged gradual items.

    'ndvi+, (precipitation+) +7.0 days, (temperature-) +2.0 weeks' ->
        [{attribute: 'ndvi', direction: '+', role: 'target',
          lag_days: 0.0, lag_raw: None},
         {attribute: 'precipitation', direction: '+', role: 'lagged',
          lag_days: 7.0, lag_raw: '+7.0 days'},
         {attribute: 'temperature', direction: '-', role: 'lagged',
          lag_days: 14.0, lag_raw: '+2.0 weeks'}]

    The target gradual item (the so4gp ``target_col``) is rendered first with
    no parentheses/lag; every other item is parenthesized and carries a signed
    time lag. The sign is kept verbatim in ``lag_raw`` and applied to
    ``lag_days`` ('+' = later, '-' = before, per so4gp's TimeDelay).
    """
    items: list[dict] = []

    # The target is the leading token before the first TOP-LEVEL comma; the lagged
    # items follow as "(attr±) ±N unit". Use the paren-aware top-level comma boundary
    # (NOT the first "(" or a naive split) so a target name containing parentheses —
    # including a comma inside them, e.g. "co2(ppm,avg)" — is not truncated.
    lead = _leading_target_token(pattern_str).strip()
    base = _split_attr_dir(lead)
    # The authoritative target name is threaded from the miner (``df.columns[target_col]``);
    # the so4gp envelope's "Target Column" is a numeric INDEX, not a name. Fall back to the
    # parsed leading token when no name is supplied. Direction always comes from the token.
    attribute = target_name if target_name else base["attribute"]
    if attribute:
        items.append(
            {
                "attribute": attribute,
                "direction": base["direction"],
                "role": "target",
                "lag_days": 0.0,
                "lag_raw": None,
            }
        )

    for match in _TEMPORAL_ITEM_RE.finditer(pattern_str):
        attr_dir, sign, raw_value, unit = match.groups()
        days = _lag_to_days(float(raw_value), unit)
        if days is not None and sign == "-":
            days = -days
        items.append(
            {
                **_split_attr_dir(attr_dir),
                "role": "lagged",
                "lag_days": days,
                "lag_raw": f"{sign}{raw_value} {unit}",
            }
        )
    return items


def parse_discover_output(
    raw: str, *, is_temporal: bool = False, target_name: str | None = None
) -> dict:
    """Normalize so4gp's discover() JSON *string* into a clean dict.

    Pure function — no so4gp/pandas import — so it is unit-testable in the
    base runtime against a captured sample.

    Args:
        raw: the JSON string returned by ``discover()`` / ``discover_tgp()``.
        is_temporal: whether the patterns are temporal (TGrad). This is set
            explicitly by the caller (``mine_gradual_patterns`` → False,
            ``mine_temporal_patterns`` → True), NOT inferred from the pattern
            text. A content heuristic (e.g. "is there a '(' in the string?")
            would misfire on plain GRAANK output whose column names contain a
            parenthesis — ``"co2(ppm)+, ndvi+"`` would be mis-parsed as a
            temporal pattern and silently drop items.

    Returns:
        {
          "algorithm": str,
          "runtime_seconds": float | None,
          "invalid_count": int | None,
          "patterns": [
            {"gradual_items": [{"attribute","direction"}, ...],
             "support": float,
             "metrics": {density, avg_dev, dispersion, connect, singularity_scr, ...}},
            ...
          ],
        }
    """
    data = json.loads(raw)
    if data.get("Error"):
        raise ValueError(f"so4gp discovery failed: {data['Error']}")

    runtime = data.get("Run-time")
    runtime_seconds = None
    if isinstance(runtime, str):
        # "1.357852 seconds" -> 1.357852
        head = runtime.strip().split()
        if head:
            try:
                runtime_seconds = float(head[0])
            except ValueError:
                runtime_seconds = None

    invalid_count = None
    if "Invalid Count" in data:
        try:
            invalid_count = int(data["Invalid Count"])
        except (TypeError, ValueError):
            invalid_count = None

    patterns: list[dict] = []
    for entry in data.get("Patterns", []):
        if not entry:
            continue
        pattern_str = entry[0]
        gradual_items = (
            _parse_temporal_items(pattern_str, target_name=target_name)
            if is_temporal
            else _parse_gradual_items(pattern_str)
        )
        metrics: dict[str, float | str] = {}
        support: float | None = None
        for token in entry[1:]:
            if not isinstance(token, str) or "=" not in token:
                continue
            # Some outputs combine assignments ("sup=0.8, conf=0.9"). Split
            # only before another metric key so string values such as "a,b" survive.
            for assignment in re.split(r",\s*(?=[A-Za-z_][A-Za-z0-9_]*\s*=)", token):
                key, value = _parse_kv(assignment)
                if key == "sup":
                    support = value if isinstance(value, float) else None
                else:
                    metrics[key] = value
        patterns.append(
            {
                "gradual_items": gradual_items,
                "support": support,
                "metrics": metrics,
                "is_temporal": is_temporal,
            }
        )

    # Temporal (TGrad) envelopes carry the target column + min representativity.
    target_column = data.get("Target Column")
    min_representativity = None
    if "Minimum Representation" in data:
        try:
            min_representativity = float(data["Minimum Representation"])
        except (TypeError, ValueError):
            min_representativity = None

    return {
        "algorithm": data.get("Algorithm"),
        "runtime_seconds": runtime_seconds,
        "invalid_count": invalid_count,
        "target_column": target_column,
        "min_representativity": min_representativity,
        "patterns": patterns,
    }


def mine_gradual_patterns(
    df,
    *,
    min_support: float = 0.5,
    encode_equal: bool = False,
    algorithm: str = "graank",
    capture_raw_output: Callable[[str], None] | None = None,
) -> dict:
    """Run so4gp on a DataFrame and return normalized gradual patterns.

    Args:
        df: pandas DataFrame of NUMERIC columns. so4gp accepts only a
            DataFrame or a CSV path; non-numeric columns are rejected by
            so4gp itself. Order rows by time before calling for temporal
            interpretation of the (contemporaneous) co-monotonic patterns.
        min_support: minimum support threshold (so4gp ``min_sup``), 0-1.
        encode_equal: so4gp ``eq`` — treat equal consecutive values as
            gradual.
        algorithm: one of SUPPORTED_ALGORITHMS keys.
        capture_raw_output: optional callback receiving the exact native JSON
            before normalization, including native Error envelopes. Callback
            failures propagate to the caller.

    Returns:
        The normalized dict from :func:`parse_discover_output`, plus the
        echoed ``min_support`` and ``algorithm``.

    Raises:
        So4gpNotInstalled: if the ``patterns`` extra is not installed.

    Note:
        The pinned public CPU API disables file output by default. Legacy
        implementations can write ``GRAANK_<timestamp>.csv/.txt`` to cwd;
        a direct caller owns cwd confinement. The MCP tools and
        the ``mine_patterns`` CLI run it through ``_mine_in_tempdir``
        (``src/tools/patterns.py``). See the ADR.
    """
    cls = _resolve_algorithm(algorithm, SUPPORTED_ALGORITHMS, kind_label="algorithm")
    kwargs = {"data_source": df, "min_sup": min_support, "eq": encode_equal}
    if "device" in inspect.signature(cls).parameters:
        kwargs["device"] = "cpu"
    miner = cls(**kwargs)
    # Leave target_col unset: the public API's default None mines unrestricted
    # itemsets. Its native conf metric is retained, not calibrated by GeoAI.
    raw = miner.discover()
    if capture_raw_output is not None:
        capture_raw_output(raw)
    result = parse_discover_output(raw, is_temporal=False)
    result["min_support"] = min_support
    result["algorithm_key"] = algorithm
    return result


def _require_temporal_clock(df, *, target_col: int):
    """Require one clock and finite attributes without upstream row/column removal.

    Native TGRAANK sums multiple clocks, and its cleaner can shift a positional
    target by dropping text columns. Typed timestamps are normalized in a copy.
    """
    import pandas as pd

    error = "Temporal mining requires exactly one complete, parseable date-time column and finite numeric attributes"
    clock_index = None
    clock_dates = None
    for index in range(len(df.columns)):
        column = df.iloc[:, index]
        if column.empty or column.isna().any():
            raise ValueError(error)
        numeric = pd.to_numeric(column, errors="coerce")
        if not pd.api.types.is_datetime64_any_dtype(column.dtype) and numeric.notna().all():
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(error)
            continue
        # Numeric-looking strings above are attributes, not implicit years.
        dates = pd.to_datetime(column, errors="coerce", format="mixed", utc=True)
        if dates.isna().any() or clock_index is not None:
            raise ValueError(error)
        clock_index, clock_dates = index, dates
    if clock_index is None:
        raise ValueError(error)
    if target_col == clock_index:
        raise ValueError("target_col must identify a numeric attribute, not the date-time column")
    if all(isinstance(value, str) for value in df.iloc[:, clock_index]):
        return df
    normalized = df.copy()
    normalized.isetitem(clock_index, clock_dates.astype(str))
    return normalized


def mine_temporal_patterns(
    df,
    *,
    target_col: int,
    min_support: float = 0.5,
    min_rep: float = 0.5,
    encode_equal: bool = False,
    algorithm: str = "tgrad",
    parallel: bool = False,
    capture_raw_output: Callable[[str], None] | None = None,
) -> dict:
    """Mine *temporal* gradual patterns from a DataFrame (FIERCE Palier 2).

    Unlike :func:`mine_gradual_patterns`, this surfaces a time lag between the
    target attribute and the other (gradual) attributes, e.g.
    "ndvi+, (precipitation+) +7.0 days".

    Args:
        df: pandas DataFrame with exactly one complete, parseable date-time
            column and finite numeric attributes. Typed timestamps are normalized to
            UTC strings for the public CPU API; legacy TGrad requires strings.
        target_col: integer index of the target/reference column (must not be
            the date-time column). Resolve a column NAME to its index at the
            call site.
        min_support: so4gp ``min_sup`` (0-1).
        min_rep: so4gp ``min_rep`` — minimum representativity; bounds how many
            temporal transformation steps (lags) are explored.
        encode_equal: so4gp ``eq``.
        algorithm: one of SUPPORTED_TEMPORAL_ALGORITHMS keys.
        parallel: pass-through to legacy ``discover_tgp`` multiprocessing;
            unsupported by the public TGRAANK CPU path.
        capture_raw_output: optional callback receiving the exact native JSON
            before normalization, including native Error envelopes. Callback
            failures propagate to the caller.

    Returns:
        The normalized dict from :func:`parse_discover_output` (patterns carry
        ``is_temporal=True`` with per-item ``role``/``lag_days``), plus the
        echoed ``min_support``, ``min_rep``, ``target_col`` and ``algorithm``.

    Raises:
        So4gpNotInstalled: if the ``patterns`` extra is not installed.

    Note:
        The public CPU call explicitly disables file output. Legacy
        ``discover_tgp()`` writes ``TGrad_<timestamp>.csv/.txt`` to cwd;
        a direct caller owns cwd confinement. The
        MCP tools and the ``mine_patterns`` CLI run it through
        ``_mine_in_tempdir`` (``src/tools/patterns.py``).
    """
    cls = _resolve_algorithm(
        algorithm, SUPPORTED_TEMPORAL_ALGORITHMS, kind_label="temporal algorithm"
    )
    legacy_api = hasattr(cls, "discover_tgp")
    if legacy_api:
        miner = cls(
            data_source=df,
            min_sup=min_support,
            eq=encode_equal,
            target_col=target_col,
            min_rep=min_rep,
        )
    else:
        if parallel:
            raise ValueError("parallel=True is unsupported by the TGRAANK CPU API")
        temporal_df = _require_temporal_clock(df, target_col=target_col)
        miner = cls(
            data_source=temporal_df,
            min_sup=min_support,
            min_rep=min_rep,
            eq=encode_equal,
            device="cpu",
        )
    # Resolve the REAL target name (the so4gp envelope's "Target Column" is a numeric
    # index, not a name); thread it so a column whose name contains "(" is not truncated.
    try:
        target_name = str(df.columns[target_col])
    except (IndexError, KeyError):
        target_name = None
    if legacy_api:
        raw = miner.discover_tgp(parallel=parallel)
    else:
        raw = miner.discover(
            target_col=target_col,
            transformations="all",
            search_algorithm="apriori",
            save_results=False,
        )
    if capture_raw_output is not None:
        capture_raw_output(raw)
    result = parse_discover_output(raw, is_temporal=True, target_name=target_name)
    for pattern in result["patterns"]:
        lagged = [item for item in pattern["gradual_items"] if item.get("role") == "lagged"]
        if not lagged or any(
            item["lag_days"] is None or not math.isfinite(item["lag_days"]) for item in lagged
        ):
            raise ValueError(
                "Temporal mining returned a lag without a supported physical time unit"
            )
    result["min_support"] = min_support
    result["min_rep"] = min_rep
    result["target_col"] = target_col
    result["algorithm_key"] = algorithm
    return result
