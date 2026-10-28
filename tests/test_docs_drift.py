"""Pin the climate storage layout in shipped Python source."""

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_never_advertise_nested_climate_zone_prefix():
    """Source checkout validation also works before the first Git commit."""
    tracked = sorted(
        path for directory in ("src", "scripts") for path in (_REPO_ROOT / directory).rglob("*.py")
    )
    assert tracked, "expected Python sources"
    nested = re.compile(r"climate/\{zone_id\}/")
    offenders = [
        f"{path.relative_to(_REPO_ROOT)}:{number}: {line.strip()}"
        for path in tracked
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if nested.search(line)
    ]
    assert not offenders, "Python sources advertise a noncanonical climate path:\n" + "\n".join(
        offenders
    )
