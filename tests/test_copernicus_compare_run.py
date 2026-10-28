"""Real timeout and input-fidelity contracts, without installed miners."""

import math
import sys

import pytest

from benchmarks.copernicus_compare_run import matrix_digest, run_bounded, validate_matrix


def test_actual_child_timeout_preserves_partial_stdout():
    outcome = run_bounded(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(3)"],
        timeout=0.4,
    )
    assert outcome["status"] == "timeout"
    assert outcome["returncode"] is None
    assert "started" in outcome["stdout"]
    assert outcome["elapsed_seconds"] < 3


def test_child_failure_is_not_successful_empty_mining():
    outcome = run_bounded([sys.executable, "-c", "raise ImportError('comparator missing')"], 3)
    assert outcome["status"] == "error"
    assert outcome["returncode"] != 0
    assert "comparator missing" in outcome["stderr"]


def test_success_preserves_actual_stdout():
    outcome = run_bounded([sys.executable, "-c", "print('actual result')"], 3)
    assert outcome["status"] == "ok"
    assert outcome["returncode"] == 0
    assert outcome["stdout"].strip() == "actual result"


@pytest.mark.parametrize("timeout", [0, -1, 61, math.nan, math.inf, True])
def test_timeout_cannot_expand_the_agreed_budget(timeout):
    with pytest.raises(ValueError):
        run_bounded([sys.executable, "-c", "print('must not start')"], timeout)


@pytest.mark.parametrize(
    "matrix",
    [
        {"columns": ["x", "y"], "rows": [[1, 2], [3]]},
        {"columns": ["x", "y"], "rows": [[1, 2], [True, 3]]},
        {"columns": ["x", "y"], "rows": [[1, 2], [None, 3]]},
        {"columns": ["x", "y"], "rows": [[1, 2], [math.nan, 3]]},
        {"columns": ["x", "x"], "rows": [[1, 2], [3, 4]]},
        {"columns": ["x"], "rows": [[1]] * 65},
    ],
)
def test_invalid_matrix_is_refused_without_row_or_column_repair(matrix):
    with pytest.raises(ValueError):
        validate_matrix(matrix)


def test_matrix_identity_preserves_order_labels_and_signed_zero():
    matrix = {"columns": ["x", "y"], "rows": [[1.0, -0.0], [2.0, 3.0]]}
    validate_matrix(matrix)
    first = matrix_digest(matrix)
    assert first == matrix_digest(dict(matrix))
    assert first != matrix_digest({**matrix, "rows": list(reversed(matrix["rows"]))})
    assert first != matrix_digest({**matrix, "columns": ["y", "x"]})
    assert first != matrix_digest({**matrix, "rows": [[1.0, 0.0], [2.0, 3.0]]})
