"""The Koppen derivation used to cross-check the panel's hand-written labels (#186).

Committed with tests because the script's whole value is being re-runnable: it answers
"are these 15 constants true" today, and it will answer it again when the normals change
and when #194 needs a midpoint baseline to measure a polygon result against. A derivation
nobody can re-run is a screenshot.

The classifier is pure, so these are constructed climates with unambiguous expected
classes — not a re-assertion of the 15 panel results, which would pin the DATA rather than
the rule and would have to be rewritten every time a zone's normals are refreshed.
"""

from __future__ import annotations

import pytest

from scripts.koppen_crosscheck import (
    BASELINE_YEARS,
    classify_koppen,
    monthly_totals,
    window_years,
)


def _flat(value: float) -> dict[int, float]:
    return dict.fromkeys(range(1, 13), value)


def _by_half(first_half: float, second_half: float) -> dict[int, float]:
    """Jan-Mar + Oct-Dec at one value, Apr-Sep at the other — the NH winter/summer split."""
    return {m: (second_half if 4 <= m <= 9 else first_half) for m in range(1, 13)}


class TestTheMonthlyTotalRecoversRealMonthLengths:
    def test_february_is_not_thirty_days(self):
        """8 leap years over 1991-2020, so the mean February is 28.27 days, not 28 or 30.

        A flat 30 would inflate every February total by 6%, which matters at a threshold.
        """
        assert monthly_totals(1.0, 848) == pytest.approx(28.27, abs=0.01)

    def test_a_thirty_one_day_month_is_exact(self):
        assert monthly_totals(2.5596, 930) == pytest.approx(79.3, abs=0.1)

    def test_the_window_length_is_a_parameter(self):
        assert monthly_totals(1.0, 310, years=10) == pytest.approx(31.0)


class TestTheTemperatureDrivenClasses:
    """These are the ones that agreed on all 15 zones, so they are the rule's load-bearing half."""

    def test_a_polar_zone_above_freezing_in_summer_is_tundra(self):
        code, _ = classify_koppen(_by_half(-5.0, 8.0), _flat(50.0))
        assert code == "ET"

    def test_a_zone_that_never_thaws_is_frost(self):
        code, _ = classify_koppen(_flat(-10.0), _flat(50.0))
        assert code == "EF"

    def test_the_c_d_boundary_is_the_zero_isotherm(self):
        """The convention `generate_zones.py` declares. One degree either side flips the class."""
        just_above, _ = classify_koppen(_by_half(0.5, 20.0), _flat(80.0))
        just_below, _ = classify_koppen(_by_half(-0.5, 20.0), _flat(80.0))
        assert just_above.startswith("C")
        assert just_below.startswith("D")


class TestAridityOverridesTemperature:
    def test_a_hot_zone_dry_enough_is_arid_not_tropical(self):
        """B is tested first on purpose: an arid zone is arid whatever its temperature says.

        Without the precedence this climate reads `Aw` — tropical savanna — on a zone
        receiving 12 mm a year.
        """
        code, _ = classify_koppen(_flat(28.0), _flat(1.0))
        assert code == "BWh"

    def test_steppe_and_desert_split_on_half_the_threshold(self):
        wetter, decided = classify_koppen(_flat(28.0), _flat(30.0))
        drier, _ = classify_koppen(_flat(28.0), _flat(10.0))
        assert wetter == "BSh"
        assert drier == "BWh"
        # The same threshold decides both, so the split is one number and not two rules.
        assert decided["threshold"] == pytest.approx(2 * 28.0 + 14)

    def test_the_cold_variant_is_below_eighteen_degrees(self):
        code, _ = classify_koppen(_flat(10.0), _flat(1.0))
        assert code == "BWk"


class TestSeasonality:
    def test_a_dry_summer_temperate_zone_is_mediterranean(self):
        """Csa: warm, with the rain falling in the half-year that is not summer."""
        code, _ = classify_koppen(_by_half(8.0, 24.0), _by_half(120.0, 10.0))
        assert code == "Csa"

    def test_summer_is_the_warmer_half_year_and_not_the_calendar(self):
        """Peel's rule, and the one this module used to state but not implement.

        Identical PRECIPITATION in both climates — dry in Apr-Sep, wet in Oct-Mar. Only the
        temperatures differ, and they are what decides which half is summer. The first climate
        is warm in Apr-Sep, so its dry half IS summer and the letter is `s`. The second is warm
        in Oct-Mar, so the same dry months are its winter and `s` cannot apply.

        A latitude-driven rule got this right for all 15 panel zones and would get it wrong for
        a near-equatorial or bimodal climate, which is exactly where the letter is fragile.
        """
        precip = _by_half(120.0, 10.0)
        dry_summer, _ = classify_koppen(_by_half(8.0, 24.0), precip)
        dry_winter, _ = classify_koppen(_by_half(24.0, 8.0), precip)

        assert dry_summer[1] == "s"
        assert dry_winter[1] != "s"

    def test_latitude_is_not_an_input(self):
        """It was a parameter, and removing it is the point: the temperatures already say it."""
        import inspect

        assert "latitude" not in inspect.signature(classify_koppen).parameters

    def test_a_year_round_wet_temperate_zone_has_no_dry_season_letter(self):
        code, _ = classify_koppen(_by_half(6.0, 18.0), _flat(70.0))
        assert code.startswith("Cf")


class TestTheDecidedOnValuesExplainTheVerdict:
    """A bare class says nothing about WHY a zone diverges, and every divergence measured so
    far has been one threshold away from agreeing."""

    def test_the_intermediates_are_returned_alongside_the_code(self):
        _code, decided = classify_koppen(_by_half(8.0, 24.0), _by_half(120.0, 10.0))
        assert decided["t_cold"] == 8.0
        assert decided["t_hot"] == 24.0
        assert decided["map"] == pytest.approx(780.0)
        assert decided["p_driest"] == 10.0

    def test_the_baseline_window_is_a_named_constant(self):
        assert BASELINE_YEARS == 30


class TestTheWindowIsReadFromTheDataNotAssumed:
    """The divisor was the literal 30. A five-year bundle then understated every monthly
    precipitation total roughly sixfold, which moves the aridity threshold silently."""

    def test_whole_calendar_years_are_counted_inclusively(self):
        assert window_years("1991-01-01", "2020-12-31") == 30
        assert window_years("2021-01-01", "2025-12-31") == 5

    @pytest.mark.parametrize(
        "start,end",
        [
            ("1991-03-01", "2020-12-31"),  # does not start on 1 January
            ("1991-01-01", "2020-08-31"),  # does not end on 31 December
        ],
    )
    def test_a_partial_window_is_refused_rather_than_approximated(self, start, end):
        """`n / years` means "days in this calendar month" only if every month appears equally.

        A window starting in March does not satisfy that, and returning a wrong total quietly
        is the failure the check exists to prevent.
        """
        with pytest.raises(ValueError, match="whole calendar years"):
            window_years(start, end)

    def test_the_five_year_case_scales_differently_from_the_thirty_year_one(self):
        """The bug, in one line: the same stored row means six times more rain over 5 years."""
        over_thirty = monthly_totals(2.0, 930, years=30)
        over_five = monthly_totals(2.0, 155, years=5)
        assert over_thirty == pytest.approx(over_five)
        assert monthly_totals(2.0, 155, years=30) == pytest.approx(over_five / 6, rel=1e-6)
