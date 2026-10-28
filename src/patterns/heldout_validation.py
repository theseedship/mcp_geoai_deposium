"""Offline observation scoring; independent of MCP and producer confidence.

Pairs never cross zones. Both orientations of each unordered pair are examined,
but strict inequalities allow at most one orientation to satisfy a rule. Thus
joint support uses n*(n-1)/2; confidence uses the fixed antecedent's count.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely.geometry import shape

TARGET = "ndvi_mean"
CLIMATE = (
    "temperature_2m",
    "precipitation",
    "humidity",
    "evapotranspiration",
    "wind_speed",
    "solar_radiation",
)


@dataclass(frozen=True)
class Rule:
    antecedent: tuple[tuple[str, str], ...]
    target_direction: str
    lag_steps: int = 0

    def __post_init__(self):
        if (
            not self.antecedent
            or any(a not in CLIMATE or d not in ("+", "-") for a, d in self.antecedent)
            or len({a for a, _ in self.antecedent}) != len(self.antecedent)
            or self.target_direction not in ("+", "-")
            or type(self.lag_steps) is not int
            or self.lag_steps < 0
        ):
            raise ValueError("Expected climate antecedents, NDVI target and nonnegative lag steps")
        object.__setattr__(self, "antecedent", tuple(sorted(self.antecedent)))

    @property
    def id(self):
        return json.dumps(
            [self.antecedent, [TARGET, self.target_direction], self.lag_steps],
            separators=(",", ":"),
        )

    def as_dict(self):
        return {
            "rule_id": self.id,
            "antecedent_items": [list(i) for i in self.antecedent],
            "consequent": [TARGET, self.target_direction],
            "lag_steps": self.lag_steps,
        }


@dataclass(frozen=True)
class Counts:
    pairs: int = 0
    antecedent: int = 0
    joint: int = 0

    def __post_init__(self):
        if (
            any(type(v) is not int for v in (self.pairs, self.antecedent, self.joint))
            or not 0 <= self.joint <= self.antecedent <= self.pairs
        ):
            raise ValueError("Invalid pair counts")

    def __add__(self, other):
        return Counts(
            self.pairs + other.pairs, self.antecedent + other.antecedent, self.joint + other.joint
        )

    def as_dict(self, min_support=0.5, min_confidence=0.5):
        support = self.joint / self.pairs if self.pairs else None
        confidence = self.joint / self.antecedent if self.antecedent else None
        status = (
            "non_evaluable"
            if confidence is None
            else ("pass" if support >= min_support and confidence >= min_confidence else "fail")
        )
        return {
            "pairs": self.pairs,
            "antecedent": self.antecedent,
            "joint": self.joint,
            "support": support,
            "confidence": confidence,
            "status": status,
        }


def candidate_universe(columns=CLIMATE, lags=(0,)):
    """Predeclared exhaustive itemsets; observations cannot change the search space."""
    columns = tuple(sorted(columns))
    if len(set(columns)) != len(columns) or not set(columns) <= set(CLIMATE):
        raise ValueError("Invalid climate columns")
    return [
        Rule(tuple((a, d) for a, d in zip(columns, signs, strict=True) if d), target, lag)
        for lag in sorted(set(lags))
        for signs in itertools.product((None, "+", "-"), repeat=len(columns))
        if any(signs)
        for target in ("+", "-")
    ]


def transform(frame, lag_steps):
    """Shift climate by frozen observation steps, with target at the later date.

    Call on a period-restricted frame: the first `lag_steps` target windows are
    removed, so neither their climate nor their target can cross the boundary.
    This is an observation-step experiment, not a fuzzy calendar-day TGrad lag.
    """
    if type(lag_steps) is not int or lag_steps < 0:
        raise ValueError("Invalid lag steps")
    result = frame.sort_values("date", kind="stable").reset_index(drop=True).copy()
    climate = [c for c in CLIMATE if c in result]
    result[climate] = result[climate].shift(lag_steps)
    return result.iloc[lag_steps:].reset_index(drop=True)


def temporal_split(frame):
    dates = pd.to_datetime(frame["date"], errors="raise")
    return (
        frame.loc[dates.between("2021-01-01", "2022-12-31")].copy(),
        frame.loc[dates.between("2023-01-01", "2024-12-31")].copy(),
    )


class _Population:
    def __init__(self, frame, lag):
        frame = transform(frame, lag)
        self.pairs = len(frame) * (len(frame) - 1) // 2
        self.masks = {}
        for column in [TARGET, *(c for c in CLIMATE if c in frame)]:
            if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(
                frame[column]
            ):
                raise ValueError(f"Expected finite numeric observations: {column}")
            values = frame[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Expected finite numeric observations: {column}")
            for sign in ("+", "-"):
                mask = (
                    values[:, None] < values[None, :]
                    if sign == "+"
                    else values[:, None] > values[None, :]
                )
                self.masks[column, sign] = int.from_bytes(np.packbits(mask).tobytes(), "little")

    def score(self, rule):
        mask = self.masks[rule.antecedent[0]]
        for item in rule.antecedent[1:]:
            mask &= self.masks[item]
        return Counts(
            self.pairs,
            mask.bit_count(),
            (mask & self.masks[TARGET, rule.target_direction]).bit_count(),
        )


def score(frame, rule):
    return _Population(frame, rule.lag_steps).score(rule)


def geometry_folds(geojson, zone_ids, n_folds=5):
    """Connected components of intersecting footprints, balanced by zone count."""
    ids = sorted(zone_ids)
    if len(ids) != len(set(ids)) or n_folds < 2:
        raise ValueError("Unique zones and at least two folds required")
    features = {f["properties"]["zone_id"]: f for f in geojson["features"]}
    geometries = {z: shape(features[z]["geometry"]) for z in ids}
    if any(g.is_empty or not g.is_valid for g in geometries.values()):
        raise ValueError("Invalid footprint")
    remaining = set(ids)
    groups = []
    while remaining:
        group, queue = [], [min(remaining)]
        remaining.remove(queue[0])
        while queue:
            zone = queue.pop()
            group.append(zone)
            neighbors = sorted(z for z in remaining if geometries[zone].intersects(geometries[z]))
            remaining.difference_update(neighbors)
            queue.extend(neighbors)
        groups.append(sorted(group))
    if len(groups) < n_folds:
        raise ValueError("Too few independent footprint groups for requested folds")
    folds = [[] for _ in range(n_folds)]
    for group in sorted(groups, key=lambda g: (-len(g), g)):
        index = min(range(n_folds), key=lambda i: (len(folds[i]), i))
        folds[index].extend(group)
    return [sorted(f) for f in folds]


def _counts_by_rule(frames, rules):
    populations = {
        (z, lag): _Population(frame, lag)
        for z, frame in frames.items()
        for lag in {r.lag_steps for r in rules}
    }
    return {r.id: {z: populations[z, r.lag_steps].score(r) for z in sorted(frames)} for r in rules}


def _pool(counts):
    result = Counts()
    for count in counts:
        result += count
    return result


def _select(counts, min_support):
    return counts.pairs > 0 and counts.antecedent > 0 and counts.joint / counts.pairs >= min_support


def spatial_validation(frames, folds, rules, *, min_support=0.5, min_confidence=0.5, koppen=None):
    if (
        len(folds) != 5
        or any(not f for f in folds)
        or sorted(z for f in folds for z in f) != sorted(frames)
    ):
        raise ValueError("Five disjoint exhaustive folds required")
    _validate_thresholds(min_support, min_confidence)
    rules = sorted(rules, key=lambda r: r.id)
    counts = _counts_by_rule(frames, rules)
    selected = {}
    donor_rows = []
    for i, heldout in enumerate(folds, 1):
        for r in rules:
            donor = _pool(c for z, c in counts[r.id].items() if z not in heldout)
            selected[i, r.id] = _select(donor, min_support)
            if selected[i, r.id]:
                donor_rows.append(
                    {"fold": i, "rule_id": r.id, **donor.as_dict(min_support, min_confidence)}
                )
    candidates = [r for r in rules if any(selected[i, r.id] for i in range(1, 6))]
    rows, zones, classes, library = [], [], [], []
    for r in candidates:
        passing = 0
        evaluated_counts = []
        for i, heldout in enumerate(folds, 1):
            if not selected[i, r.id]:
                rows.append(
                    {
                        "fold": i,
                        "rule_id": r.id,
                        "selected": False,
                        "status": "not_selected",
                        "pairs": None,
                        "antecedent": None,
                        "joint": None,
                        "support": None,
                        "confidence": None,
                    }
                )
                continue
            pooled = _pool(counts[r.id][z] for z in heldout)
            evaluated_counts.append(pooled)
            metrics = pooled.as_dict(min_support, min_confidence)
            passing += metrics["status"] == "pass"
            rows.append({"fold": i, "rule_id": r.id, "selected": True, **metrics})
            for z in heldout:
                zones.append(
                    {
                        "fold": i,
                        "rule_id": r.id,
                        "zone_id": z,
                        "koppen": (koppen or {}).get(z),
                        **counts[r.id][z].as_dict(min_support, min_confidence),
                    }
                )
            if koppen:
                for label in sorted({koppen[z] for z in heldout}):
                    members = [z for z in heldout if koppen[z] == label]
                    classes.append(
                        {
                            "fold": i,
                            "rule_id": r.id,
                            "koppen": label,
                            "zone_ids": members,
                            **_pool(counts[r.id][z] for z in members).as_dict(
                                min_support, min_confidence
                            ),
                        }
                    )
        library.append(
            {
                **r.as_dict(),
                **_pool(evaluated_counts).as_dict(min_support, min_confidence),
                "passing_folds": passing,
                "stability": passing / 5,
                "validated": passing >= 4,
            }
        )
    return {
        "folds": [
            {"fold": i, "train_zones": sorted(set(frames) - set(f)), "validation_zones": f}
            for i, f in enumerate(folds, 1)
        ],
        "candidates": library,
        "donor_scores": donor_rows,
        "fold_scores": rows,
        "zone_scores": zones,
        "koppen_scores": classes,
    }


def _validate_thresholds(support, confidence):
    if not 0 <= support <= 1 or not 0 <= confidence <= 1:
        raise ValueError("Thresholds must lie between zero and one")


def _lag_elapsed_days(frame, lag):
    dates = pd.to_datetime(frame["date"]).sort_values()
    intervals = (dates - dates.shift(lag)).dt.days.dropna()
    return {
        "min": int(intervals.min()) if len(intervals) else None,
        "max": int(intervals.max()) if len(intervals) else None,
    }


def temporal_validation(frames, rules, *, min_support=0.5, min_confidence=0.5, koppen=None):
    _validate_thresholds(min_support, min_confidence)
    splits = {z: temporal_split(f) for z, f in frames.items()}
    donors = _counts_by_rule({z: f[0] for z, f in splits.items()}, rules)
    candidates = [r for r in rules if _select(_pool(donors[r.id].values()), min_support)]
    # Evaluation is computed only after the donor candidate library is frozen.
    tests = _counts_by_rule({z: f[1] for z, f in splits.items()}, candidates)
    rows, zone_rows, class_rows = [], [], []
    for r in candidates:
        rows.append(
            {"rule_id": r.id, **_pool(tests[r.id].values()).as_dict(min_support, min_confidence)}
        )
        for z, c in tests[r.id].items():
            zone_rows.append(
                {
                    "rule_id": r.id,
                    "zone_id": z,
                    "koppen": (koppen or {}).get(z),
                    **c.as_dict(min_support, min_confidence),
                }
            )
        if koppen:
            for label in sorted({koppen[z] for z in frames}):
                members = sorted(z for z in frames if koppen[z] == label)
                class_rows.append(
                    {
                        "rule_id": r.id,
                        "koppen": label,
                        "zone_ids": members,
                        **_pool(tests[r.id][z] for z in members).as_dict(
                            min_support, min_confidence
                        ),
                    }
                )
    return {
        "candidates": [r.as_dict() for r in candidates],
        "donor_scores": [
            {"rule_id": r.id, **_pool(donors[r.id].values()).as_dict(min_support, min_confidence)}
            for r in candidates
        ],
        "scores": rows,
        "zone_scores": zone_rows,
        "koppen_scores": class_rows,
        "periods": {
            "train": ["2021-01-01", "2022-12-31"],
            "evaluate": ["2023-01-01", "2024-12-31"],
        },
        "coverage": [
            {
                "zone_id": z,
                "train_rows": len(d),
                "evaluation_rows": len(t),
                "lag_windows_removed": {
                    str(lag): {"train": min(lag, len(d)), "evaluation": min(lag, len(t))}
                    for lag in sorted({r.lag_steps for r in rules})
                },
                "lag_elapsed_days": {
                    str(lag): {
                        "train": _lag_elapsed_days(d, lag),
                        "evaluation": _lag_elapsed_days(t, lag),
                    }
                    for lag in sorted({r.lag_steps for r in rules})
                },
            }
            for z, (d, t) in sorted(splits.items())
        ],
    }
