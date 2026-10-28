"""Network-free tests for the shared calendar helper.

`src/utils/dates.py` exists to be importable without the raster/STAC stack, so its tests import
nothing beyond it.
"""

from __future__ import annotations

from src.utils.dates import month_ranges


class TestTheIsoUpperBound:
    """`9999-12-31` is a legal ISO date and must not be an error.

    Reaching the end of a month through "the day before the 1st of next month" needs
    `date(cur.year + 1, 1, 1)`, which for December 9999 is outside `datetime`'s range. The helper
    raised `ValueError: year must be in 1..9999, not 10000`, and callers wrapped it — guarding a
    helper that should not have been raising.
    """

    def test_the_last_representable_year_slices_into_twelve_months(self):
        slices = month_ranges("9999-01-01", "9999-12-31")

        assert len(slices) == 12
        assert slices[0] == ("9999-01-01", "9999-01-31")
        assert slices[-1] == ("9999-12-01", "9999-12-31")

    def test_the_last_representable_month_alone(self):
        assert month_ranges("9999-12-01", "9999-12-31") == [("9999-12-01", "9999-12-31")]

    def test_the_very_last_representable_day(self):
        assert month_ranges("9999-12-31", "9999-12-31") == [("9999-12-31", "9999-12-31")]

    def test_the_ordinary_cases_are_unchanged(self):
        """The December branch is spelled out now, so December must still behave like any month."""
        assert month_ranges("2024-11-15", "2025-01-10") == [
            ("2024-11-15", "2024-11-30"),
            ("2024-12-01", "2024-12-31"),
            ("2025-01-01", "2025-01-10"),
        ]
