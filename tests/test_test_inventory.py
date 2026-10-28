"""Preserve live acceptance harness coverage and hermetic CI discovery."""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TESTS = _REPO_ROOT / "tests"
_HARNESS = _TESTS / "live_acceptance.py"
_WORKFLOW = _REPO_ROOT / ".github/workflows/ci.yml"


def _harness_source() -> str:
    return _HARNESS.read_text(encoding="utf-8")


def _live_case_count() -> int:
    """Every ``check(...)`` call in the harness, counted on the AST.

    Not a grep: a `check(` inside a string or a comment is not a case, and a call split
    across lines is. The number on the page has to be the number a reader would get by
    running the thing.
    """
    tree = ast.parse(_harness_source())
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "check"
    )


def test_the_live_harness_is_not_collected_by_pytest():
    """The exclusion is structural, which is the whole point of the rename.

    Under `tests/test_integration.py` the file matched pytest's discovery pattern, collected
    zero tests, and had to be excluded by a `--ignore` flag — one YAML edit away from being
    pulled into the gate. A name pytest does not discover cannot be pulled in by accident.
    """
    assert _HARNESS.exists(), "the live harness moved again; update this guard and the docs"
    assert not _HARNESS.name.startswith("test_")
    assert not _HARNESS.name.endswith("_test.py")


def test_the_old_name_is_gone():
    assert not (_TESTS / "test_integration.py").exists()


def test_the_workflow_carries_no_exclusion_for_it():
    """A `--ignore` would mean the gate is a configured list again, not a rule.

    Read off the executable lines only. The comment above the step explains that the flag is
    gone, and a guard that scanned the prose too would fail on its own explanation.
    """
    executable = [
        line
        for line in _WORKFLOW.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not [line for line in executable if "--ignore" in line]
    assert not [line for line in executable if "test_integration.py" in line]


def test_every_branch_of_the_harness_accounts_for_all_56_cases():
    """The sharp one: twelve cases only run when the live response carries data.

    Before #171's review they were simply not executed, and the summary printed
    ALL TESTS PASSED over a run of 44 — while the criterion cites 56. Each now ends in
    ``skip(label, reason)``. This walks every combination of the harness's conditionals and
    asserts the outcomes always total 56, so no if/else branch can drop a case silently (an
    ``if`` without ``else`` is not simulated).
    """
    tree = ast.parse(_harness_source())
    run_tests = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_tests"
    )

    def outcomes(nodes, taken):
        total = 0
        for node in nodes:
            if isinstance(node, ast.If):
                branch = taken.get(ast.unparse(node.test)[:70], True)
                total += outcomes(node.body if branch else node.orelse, taken)
            elif isinstance(node, ast.For):
                total += len(getattr(node.iter, "elts", [])) * outcomes(node.body, taken)
            elif isinstance(node, ast.AsyncWith | ast.With | ast.Try):
                total += outcomes(node.body, taken)
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ("check", "skip")
            ):
                total += 1
        return total

    conditions = sorted(
        {
            ast.unparse(node.test)[:70]
            for node in ast.walk(run_tests)
            if isinstance(node, ast.If) and node.orelse
        }
    )
    totals = {
        outcomes(run_tests.body, dict(zip(conditions, combo, strict=True)))
        for combo in itertools.product([True, False], repeat=len(conditions))
    }
    assert totals == {_live_case_count()}


def test_the_harness_declares_the_inventory_it_reports_against():
    """The summary's denominator must be the inventoried figure, not "whatever ran"."""
    source = _harness_source()
    assert f"INVENTORIED_CASES = {_live_case_count()}" in source
    assert "of {INVENTORIED_CASES} inventoried cases" in source
    assert "skipped" in source
