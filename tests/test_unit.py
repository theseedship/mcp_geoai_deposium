"""Unit tests for pure logic helpers — no network, no MCP server.

Run with:
    pytest tests/test_unit.py -v -o addopts=""
"""

import numpy as np
import pytest

from src.tools.change import (
    _classify_ndvi_change,
    _detect_profile_stress,
    _first_rapid_decline,
    _low_ndvi_alert,
    _profile_statistics,
    _resample_by_interval,
)
from src.utils.raster import compute_ndvi

# ---------------------------------------------------------------------------
# _classify_ndvi_change — recalibrated formula (sprint fix-geoai-correctness)
# ---------------------------------------------------------------------------


class TestClassifyNdviChange:
    """Confidence saturates at |dNDVI|=0.2; with default threshold 0.7,
    changes >= ~0.14 NDVI are flagged. Ensures we don't miss real
    vegetation losses (typical 0.10-0.20 range)."""

    def test_noise_below_floor_is_no_change(self):
        # 0.04 NDVI delta is below the noise floor regardless of confidence
        cls, conf = _classify_ndvi_change(-0.04, confidence_threshold=0.0)
        assert cls == "no_change"

    def test_small_real_loss_is_detected(self):
        # 0.15 NDVI drop: confidence = 0.75, above default 0.7 threshold
        cls, conf = _classify_ndvi_change(-0.15, confidence_threshold=0.7)
        assert cls == "vegetation_loss"
        assert conf == pytest.approx(0.75, abs=0.01)

    def test_small_real_gain_is_detected(self):
        cls, conf = _classify_ndvi_change(0.15, confidence_threshold=0.7)
        assert cls == "vegetation_gain"
        assert conf == pytest.approx(0.75, abs=0.01)

    def test_borderline_at_default_threshold(self):
        # 0.14 NDVI drop: confidence = 0.70, exactly at default threshold
        cls, conf = _classify_ndvi_change(-0.14, confidence_threshold=0.7)
        assert cls == "vegetation_loss"

    def test_just_below_threshold(self):
        # 0.13 NDVI drop: confidence = 0.65, below default threshold
        cls, conf = _classify_ndvi_change(-0.13, confidence_threshold=0.7)
        assert cls == "no_change"

    def test_high_threshold_rejects_medium_change(self):
        # 0.15 NDVI drop with strict 0.95 threshold → noise
        cls, conf = _classify_ndvi_change(-0.15, confidence_threshold=0.95)
        assert cls == "no_change"

    def test_confidence_caps_at_one(self):
        cls, conf = _classify_ndvi_change(-0.5, confidence_threshold=0.7)
        assert cls == "vegetation_loss"
        assert conf == 1.0

    def test_zero_diff_is_no_change(self):
        cls, conf = _classify_ndvi_change(0.0, confidence_threshold=0.7)
        assert cls == "no_change"
        assert conf == 0.0

    def test_old_formula_regression(self):
        # Regression guard: under the OLD formula (divisor 0.5), a 0.20
        # NDVI loss gave confidence=0.4 → no_change. With the new formula
        # it must be detected.
        cls, conf = _classify_ndvi_change(-0.20, confidence_threshold=0.7)
        assert cls == "vegetation_loss", "0.20 NDVI loss must trigger detection"


# ---------------------------------------------------------------------------
# _resample_by_interval — Phase 3
# ---------------------------------------------------------------------------


def _obs(date: str, ndvi: float = 0.5) -> dict:
    """Build a minimal profile entry."""
    return {"date": date, "ndvi_mean": ndvi}


class TestResampleByInterval:
    def test_disabled_when_interval_zero(self):
        profile = [_obs("2024-06-01"), _obs("2024-06-02"), _obs("2024-06-03")]
        assert _resample_by_interval(profile, 0) == profile

    def test_disabled_when_interval_negative(self):
        profile = [_obs("2024-06-01"), _obs("2024-06-02")]
        assert _resample_by_interval(profile, -5) == profile

    def test_passthrough_for_single_obs(self):
        profile = [_obs("2024-06-01")]
        assert _resample_by_interval(profile, 30) == profile

    def test_drops_close_observations(self):
        # 6 obs at 5-day spacing; with interval=14, expect first + every
        # 3rd one (5d gaps too small, 10d still too small, 15d clears it)
        profile = [
            _obs("2024-06-01"),
            _obs("2024-06-06"),
            _obs("2024-06-11"),
            _obs("2024-06-16"),
            _obs("2024-06-21"),
            _obs("2024-06-26"),
        ]
        result = _resample_by_interval(profile, 14)
        kept_dates = [r["date"] for r in result]
        assert kept_dates == ["2024-06-01", "2024-06-16"]

    def test_keeps_all_when_already_spaced(self):
        profile = [_obs("2024-01-01"), _obs("2024-03-01"), _obs("2024-05-01")]
        result = _resample_by_interval(profile, 30)
        assert len(result) == 3

    def test_drops_entries_with_none_date(self):
        # When resampling is active, entries with no date can't be placed
        # in the temporal sequence and are dropped.
        profile = [
            _obs("2024-06-01"),
            {"date": None, "ndvi_mean": 0.4},
            _obs("2024-07-01"),
        ]
        result = _resample_by_interval(profile, 14)
        assert len(result) == 2
        assert all(r["date"] is not None for r in result)

    def test_supports_full_iso_datetime(self):
        # STAC dates often include time component
        profile = [
            _obs("2024-06-01T10:00:00+00:00"),
            _obs("2024-06-15T10:00:00+00:00"),
            _obs("2024-06-29T10:00:00+00:00"),
        ]
        result = _resample_by_interval(profile, 16)
        # Day 1 → kept, day 15 → 14d gap, drop. Day 29 → 28d gap from day 1, kept.
        assert len(result) == 2


# ---------------------------------------------------------------------------
# compute_ndvi — pure numpy helper
# ---------------------------------------------------------------------------


class TestComputeNdvi:
    def test_basic_ndvi(self):
        red = np.array([[1000.0, 2000.0]])
        nir = np.array([[3000.0, 2000.0]])
        mean, std, vmin, vmax = compute_ndvi(red, nir)
        assert vmin == pytest.approx(0.0)
        assert vmax == pytest.approx(0.5)
        assert mean == pytest.approx(0.25)

    def test_zero_denominator_is_excluded(self):
        # red + nir == 0 must not raise nor pollute the stats
        red = np.array([[0.0, 1000.0]])
        nir = np.array([[0.0, 3000.0]])
        stats = compute_ndvi(red, nir)
        assert stats is not None
        mean, _, vmin, vmax = stats
        assert mean == vmin == vmax == pytest.approx(0.5)

    def test_all_nan_returns_none(self):
        red = np.full((2, 2), np.nan)
        nir = np.full((2, 2), np.nan)
        assert compute_ndvi(red, nir) is None

    def test_nan_pixels_are_ignored(self):
        red = np.array([[np.nan, 1000.0]])
        nir = np.array([[3000.0, 3000.0]])
        mean, _, _, _ = compute_ndvi(red, nir)
        assert mean == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _profile_statistics
# ---------------------------------------------------------------------------


class TestProfileStatistics:
    def test_empty_profile_returns_none_fields(self):
        assert _profile_statistics([]) == {
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
        }

    def test_aggregates(self):
        profile = [_obs("2024-06-01", 0.2), _obs("2024-06-20", 0.6)]
        stats = _profile_statistics(profile)
        assert stats["mean"] == pytest.approx(0.4)
        assert stats["min"] == pytest.approx(0.2)
        assert stats["max"] == pytest.approx(0.6)
        assert stats["std"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# _detect_profile_stress
# ---------------------------------------------------------------------------


class TestDetectProfileStress:
    def test_no_stress_on_healthy_profile(self):
        profile = [_obs("2024-06-01", 0.7), _obs("2024-06-20", 0.68)]
        detected, period, drop, gap_days = _detect_profile_stress(profile)
        assert detected is False
        assert period is None
        assert drop is None
        assert gap_days is None

    def test_low_ndvi_triggers_stress(self):
        # Below DEFAULT_NDVI_STRESS_THRESHOLD (0.3)
        profile = [_obs("2024-06-01", 0.25)]
        detected, period, drop, gap_days = _detect_profile_stress(profile)
        assert detected is True
        assert period == "2024-06-01"
        assert drop is None
        assert gap_days is None

    def test_rapid_drop_triggers_stress_and_reports_drop(self):
        # Drop of 0.25 > DEFAULT_NDVI_DROP_THRESHOLD (0.2)
        profile = [_obs("2024-06-01", 0.8), _obs("2024-06-20", 0.55)]
        detected, period, drop, gap_days = _detect_profile_stress(profile)
        assert detected is True
        assert period == "2024-06-20"
        assert drop == pytest.approx(0.25)
        assert gap_days == 19

    def test_same_drop_is_rapid_only_within_maximum_gap(self):
        short_gap = [_obs("2024-06-01", 0.8), _obs("2024-07-03", 0.55)]
        long_gap = [_obs("2024-06-01", 0.8), _obs("2024-07-04", 0.55)]

        assert _detect_profile_stress(short_gap) == (True, "2024-07-03", 0.25, 32)
        assert _detect_profile_stress(long_gap) == (False, None, None, None)

    def test_irregular_spacing_uses_each_consecutive_gap(self):
        profile = [
            _obs("2024-06-01", 0.8),
            _obs("2024-07-10", 0.55),  # 39-day gap: not rapid
            _obs("2024-07-20", 0.5),  # 10-day gap, but below drop threshold
            _obs("2024-07-30", 0.25),  # 10-day gap and qualifying drop
        ]
        assert _detect_profile_stress(profile) == (True, "2024-07-30", 0.25, 10)

    def test_non_increasing_or_invalid_dates_do_not_make_a_rapid_drop(self):
        for dates in [
            ("2024-06-01", "2024-06-01"),
            ("2024-06-02", "2024-06-01"),
            ("2024-06-01T23:30:00-02:00", "2024-06-02T02:00:00+00:00"),
            ("not-a-date", "2024-06-20"),
        ]:
            profile = [_obs(dates[0], 0.8), _obs(dates[1], 0.55)]
            assert _detect_profile_stress(profile) == (False, None, None, None)

    def test_empty_profile(self):
        assert _detect_profile_stress([]) == (False, None, None, None)


# ---------------------------------------------------------------------------
# vegetation_stress alert helpers
# ---------------------------------------------------------------------------


class TestStressAlertHelpers:
    def test_low_ndvi_alert_below_threshold(self):
        alerts = _low_ndvi_alert(0.2, 0.3)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "low_ndvi"
        assert alerts[0]["ndvi_value"] == pytest.approx(0.2)

    def test_low_ndvi_alert_above_threshold(self):
        assert _low_ndvi_alert(0.5, 0.3) == []

    def test_first_rapid_decline_found(self):
        observations = [
            _obs("2024-06-01", 0.8),
            _obs("2024-06-10", 0.75),
            _obs("2024-06-20", 0.4),  # drop 0.35
            _obs("2024-06-30", 0.1),  # drop 0.3 — must NOT be reported
        ]
        alert = _first_rapid_decline(observations, drop_threshold=0.2)
        assert alert is not None
        assert alert["type"] == "rapid_decline"
        assert alert["date_to"] == "2024-06-20"
        assert alert["ndvi_drop"] == pytest.approx(0.35)
        assert alert["gap_days"] == 10
        assert alert["max_gap_days"] == 32
        assert alert["rule"] == "absolute_drop_with_max_gap"

    def test_first_rapid_decline_ignores_drop_after_maximum_gap(self):
        observations = [_obs("2024-06-01", 0.8), _obs("2024-07-04", 0.55)]
        assert _first_rapid_decline(observations, drop_threshold=0.2) is None

    def test_profile_and_stress_alert_paths_apply_same_gap_rule(self):
        observations = [_obs("2024-06-01", 0.8), _obs("2024-06-20", 0.55)]
        rapid_alert = _first_rapid_decline(observations, drop_threshold=0.2)
        stress_detected, stress_period, drop, gap_days = _detect_profile_stress(observations)

        assert rapid_alert is not None
        assert (stress_detected, stress_period, drop, gap_days) == (
            True,
            rapid_alert["date_to"],
            rapid_alert["ndvi_drop"],
            rapid_alert["gap_days"],
        )

    def test_first_rapid_decline_none(self):
        observations = [_obs("2024-06-01", 0.8), _obs("2024-06-10", 0.75)]
        assert _first_rapid_decline(observations, drop_threshold=0.2) is None
