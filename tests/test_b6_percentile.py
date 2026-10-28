"""Unit tests for FIERCE B6 level-1 percentile thresholds — the pure, network-free core of
`vegetation_stress` (`percentile_thresholds`, `_resolve_ndvi_threshold`,
`_analyze_vegetation_stress_observations`)."""

from __future__ import annotations

from src.tools.change import (
    _analyze_vegetation_stress_observations,
    _resolve_ndvi_threshold,
)
from src.utils.stats import percentile_thresholds


def _obs(values: list[float]) -> list[dict]:
    return [{"date": f"2024-01-{i + 1:02d}", "ndvi_mean": v} for i, v in enumerate(values)]


# 10 points descending 0.9 -> 0.45; split 75/25 -> recent = last 3 -> recent_mean = 0.5
DESCENDING = _obs([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45])


class TestPercentileThresholds:
    def test_selected_threshold_honours_level(self):
        vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        assert (
            percentile_thresholds(vals, level=10)["selected_threshold"]
            == (percentile_thresholds(vals, level=10)["p10"])
        )
        assert (
            percentile_thresholds(vals, level=90)["selected_threshold"]
            == (percentile_thresholds(vals, level=90)["p90"])
        )
        p = percentile_thresholds(vals, level=50)
        assert p["p10"] <= p["p50"] <= p["p90"]

    def test_arbitrary_level_not_limited_to_decile_set(self):
        p = percentile_thresholds([0.0, 1.0], level=25)
        assert p["selected_level"] == 25
        assert 0.0 <= p["selected_threshold"] <= 1.0


class TestResolveThreshold:
    def test_static_mode_uses_static_threshold(self):
        eff, meta = _resolve_ndvi_threshold(
            DESCENDING,
            static_threshold=0.3,
            threshold_mode="static",
            percentile_level=10,
            min_points=8,
        )
        assert eff == 0.3
        assert meta["threshold_source"] == "static"
        assert meta["effective_ndvi_threshold"] == 0.3
        assert meta["percentile_values"] is None
        assert meta["fallback_reason"] is None

    def test_percentile_mode_with_enough_points(self):
        eff, meta = _resolve_ndvi_threshold(
            DESCENDING,
            static_threshold=0.3,
            threshold_mode="percentile",
            percentile_level=90,
            min_points=8,
        )
        expected = percentile_thresholds([o["ndvi_mean"] for o in DESCENDING], level=90)
        assert meta["threshold_source"] == "percentile"
        assert meta["trigger_value"] == "recent_mean"
        assert eff == expected["selected_threshold"]
        assert meta["effective_ndvi_threshold"] == eff
        assert meta["percentile_values"]["p90"] == expected["p90"]

    def test_percentile_mode_falls_back_below_min_points(self):
        eff, meta = _resolve_ndvi_threshold(
            DESCENDING[:5],
            static_threshold=0.3,
            threshold_mode="percentile",
            percentile_level=90,
            min_points=8,
        )
        assert eff == 0.3
        assert meta["threshold_source"] == "static_fallback"
        assert meta["fallback_reason"] == "insufficient_points"
        assert meta["n_points"] == 5

    def test_empty_static_records_provenance(self):
        # zero observations: the static path still echoes the mode + a None trigger (nothing compared)
        eff, meta = _resolve_ndvi_threshold(
            [], static_threshold=0.3, threshold_mode="static", percentile_level=10, min_points=8
        )
        assert eff == 0.3
        assert meta["threshold_mode"] == "static"
        assert meta["threshold_source"] == "static"
        assert meta["n_points"] == 0
        assert meta["trigger_value"] is None  # no observation compared (Codex #36)

    def test_empty_percentile_records_fallback_provenance(self):
        # zero observations in percentile mode: provenance must stay distinguishable from static —
        # it records the requested mode/level and the static fallback that WOULD apply (Codex #36)
        eff, meta = _resolve_ndvi_threshold(
            [], static_threshold=0.3, threshold_mode="percentile", percentile_level=90, min_points=8
        )
        assert eff == 0.3
        assert meta["threshold_mode"] == "percentile"
        assert meta["percentile_level"] == 90
        assert meta["threshold_source"] == "static_fallback"
        assert meta["fallback_reason"] == "insufficient_points"
        assert meta["n_points"] == 0
        assert meta["trigger_value"] is None


class TestAnalyze:
    def test_static_default_no_low_ndvi_when_recent_above_threshold(self):
        cur, base, change, alerts, meta = _analyze_vegetation_stress_observations(
            DESCENDING, ndvi_threshold=0.3, drop_threshold=0.2
        )
        types = {a["type"] for a in alerts}
        assert cur == 0.5  # recent_mean
        assert "low_ndvi" not in types  # 0.5 >= static 0.3
        assert (
            "declining_trend" in types
        )  # change -0.25 < -drop_threshold (static, unaffected by mode)
        assert meta["threshold_source"] == "static"
        assert meta["baseline_observations"] + meta["recent_observations"] == len(DESCENDING)
        assert meta["ndvi_decline_rule"] == "absolute_drop_with_max_gap"
        assert meta["ndvi_decline_max_gap_days"] == 32

    def test_percentile_mode_fires_low_ndvi_via_p90(self):
        # static would NOT fire (recent_mean 0.5 >= 0.3); percentile P90 (~0.85) DOES (0.5 < 0.85)
        cur, base, change, alerts, meta = _analyze_vegetation_stress_observations(
            DESCENDING,
            ndvi_threshold=0.3,
            drop_threshold=0.2,
            threshold_mode="percentile",
            percentile_level=90,
        )
        assert "low_ndvi" in {a["type"] for a in alerts}
        assert meta["threshold_source"] == "percentile"
        assert meta["trigger_value"] == "recent_mean"
        assert meta["effective_ndvi_threshold"] == meta["percentile_values"]["p90"]

    def test_percentile_falls_back_below_min_points(self):
        _, _, _, alerts, meta = _analyze_vegetation_stress_observations(
            DESCENDING[:5],
            ndvi_threshold=0.3,
            drop_threshold=0.2,
            threshold_mode="percentile",
            percentile_level=90,
        )
        assert meta["threshold_source"] == "static_fallback"
        assert "low_ndvi" not in {a["type"] for a in alerts}  # static 0.3, recent_mean >= 0.3

    def test_single_observation_snapshot(self):
        cur, base, change, alerts, meta = _analyze_vegetation_stress_observations(
            _obs([0.2]), ndvi_threshold=0.3, drop_threshold=0.2
        )
        assert cur == 0.2 and base is None and change is None
        assert "low_ndvi" in {a["type"] for a in alerts}  # 0.2 < 0.3
        assert meta["n_points"] == 1
        assert meta["trigger_value"] == "current_ndvi"  # raw obs, not "recent_mean" (Greptile #36)
