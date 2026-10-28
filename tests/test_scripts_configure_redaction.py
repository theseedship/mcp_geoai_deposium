"""#161 review — a script that drives tool handlers in-process must install the redactor.

The processor added in #148 is installed by `setup_logging()`, and only `src/server.py` called it.
structlog's DEFAULT chain has no redactor, so in a script run every `log.error(..., error=str(e))`
these handlers reach went to stderr verbatim — all 17 of them in `src/tools/`, not just the one
the review happened to land on.

Patching the one call site under review would have been the per-call-site fix
`src/utils/mcp_redaction.py` argues against in its own module docstring. This pins the boundary
instead: the eighth script to import `src.` fails here rather than leaking quietly.

#202 extends the same guards to `src/cli/*.py`: its `python -m src.cli.X` entry points drive the
same handlers in-process and had been left out.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

# Anchored on this file, not on the working directory: a runner that starts pytest from anywhere
# but the repo root would otherwise find no scripts, and `test_the_candidate_set_is_not_empty`
# is the only thing standing between that and a suite of guards silently parametrized over
# nothing.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = sorted((_REPO_ROOT / "scripts").glob("*.py"))
CLI_MODULES = sorted((_REPO_ROOT / "src" / "cli").glob("*.py"))


def _imports_src(tree: ast.Module) -> bool:
    return any(
        (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src"))
        or (isinstance(node, ast.Import) and any(a.name.startswith("src") for a in node.names))
        for node in ast.walk(tree)
    )


def _has_entry_point(source: str) -> bool:
    return '__name__ == "__main__"' in source or "__name__ == '__main__'" in source


def _entry_points(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    return [
        path
        for path in paths
        if _imports_src(ast.parse(path.read_text())) and _has_entry_point(path.read_text())
    ]


SCRIPT_CANDIDATES = _entry_points(SCRIPTS)
CLI_CANDIDATES = _entry_points(CLI_MODULES)
CANDIDATES = SCRIPT_CANDIDATES + CLI_CANDIDATES


def test_the_candidate_set_is_not_empty():
    # Without this, a refactor that moved or renamed the scripts would turn the guard below into
    # a parametrize over nothing — green, and guarding nothing. One floor per directory, so
    # losing all of one cannot hide behind the other's count.
    assert len(SCRIPT_CANDIDATES) >= 7, (
        f"expected the script entry points, found {SCRIPT_CANDIDATES}"
    )
    assert len(CLI_CANDIDATES) >= 8, f"expected the src/cli entry points, found {CLI_CANDIDATES}"


@pytest.mark.parametrize("path", CANDIDATES, ids=lambda p: p.name)
def test_every_script_entry_point_installs_the_redaction_processor(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "setup_logging")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "setup_logging")
        )
    ]
    assert calls, (
        f"{path} drives src/ handlers in-process but never calls setup_logging(), so structlog "
        "runs its default chain with no redact_processor and every error=str(e) in src/tools/ "
        "reaches stderr verbatim"
    )


def _is_main_guard(node: ast.stmt) -> bool:
    """``if __name__ == "__main__":`` — the one module-level branch that does NOT run on import."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _statements_executed_on_import(body: list[ast.stmt]):
    """Every statement an ``import`` of this module would run.

    Descends module-level branches — ``if``, ``try``, ``with``, loops — because a call in any of
    them executes at import just as surely as one at top level. Does NOT descend function or class
    bodies (deferred), nor the ``__main__`` guard (never true on import).
    """
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        if _is_main_guard(node):
            # Skipped entirely rather than yielded-then-not-recursed: the caller walks each
            # yielded statement's subtree, so yielding this one would reach into the very branch
            # that is exempt.
            continue
        yield node
        for field in ("body", "orelse", "finalbody"):
            nested = getattr(node, field, None)
            if isinstance(nested, list):
                yield from _statements_executed_on_import(nested)
        for handler in getattr(node, "handlers", []) or []:
            yield from _statements_executed_on_import(handler.body)


def _calls_not_deferred(node: ast.AST):
    """Every ``Call`` in this statement, without descending into a function or class body.

    A ``def`` nested in a module-level ``if`` still only *defines* the call — walking into it
    would flag the perfectly correct ``def main(): setup_logging()`` as an import-time call.
    """
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return
    if isinstance(node, ast.Call):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from _calls_not_deferred(child)


@pytest.mark.parametrize("path", CANDIDATES, ids=lambda p: p.name)
def test_the_call_does_not_run_at_import_time(path: pathlib.Path):
    # `setup_logging()` reconfigures structlog, root logging and (after #159) the `fastmcp`
    # logger, all process-wide. Anywhere that runs on import, it would fire whenever a test
    # imports the script and rewire logging for the rest of the session.
    #
    # The name matters here, and this guard has now been narrowed-then-widened twice. It began as
    # "the call is inside a function" while only rejecting a bare top-level expression, so a call
    # under `if SOME_FLAG:` passed. Widening the statement walk was not enough either: matching
    # only `ast.Expr` with an `ast.Name` callee still let through `geoai_logging.setup_logging()`
    # and `x = setup_logging()`, both of which execute on import just the same.
    #
    # So the match is now on the CALL, wherever it sits in an import-executed statement, by either
    # spelling of the callee. `if __name__ == "__main__":` stays exempt because it genuinely does
    # not run on import — the property being pinned is "does not run on import", not "is lexically
    # inside a function".
    tree = ast.parse(path.read_text())
    on_import = [
        call
        for statement in _statements_executed_on_import(tree.body)
        for call in _calls_not_deferred(statement)
        if (isinstance(call.func, ast.Name) and call.func.id == "setup_logging")
        or (isinstance(call.func, ast.Attribute) and call.func.attr == "setup_logging")
    ]
    assert not on_import, f"{path} calls setup_logging() on import"


class TestTheProcessorActuallyRedactsOnceAScriptInstallsIt:
    """The other half: calling `setup_logging()` has to be worth calling.

    The guards above pin that every script installs the processor; this pins that the processor
    then scrubs the record the #161 review pointed at — a partial STAC failure whose exception
    quotes a credential-bearing STAC_URL.
    """

    @pytest.fixture(autouse=True)
    def _restore_global_logging(self, monkeypatch):
        import logging as stdlib_logging

        import structlog

        from src.utils import logging as geoai_logging
        from src.utils import s3_endpoint

        monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", "rustfs-admin", raising=False)
        monkeypatch.setattr(
            s3_endpoint.settings, "S3_SECRET_KEY", "rustfs-admin-secret", raising=False
        )
        monkeypatch.setattr(
            s3_endpoint.settings, "S3_ENDPOINT", "http://rustfs:9000", raising=False
        )
        monkeypatch.setattr(
            s3_endpoint.settings,
            "STAC_URL",
            "https://stac-reader:s3cret-token@stac.internal/v1",
            raising=False,
        )
        for name in s3_endpoint.AMBIENT_STORE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)

        root = stdlib_logging.getLogger()
        saved = (root.handlers[:], root.level, structlog.get_config())
        yield geoai_logging
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
        structlog.configure(**saved[2])

    def test_the_partial_failure_warning_is_scrubbed_in_a_script_run(
        self, _restore_global_logging, capfd
    ):
        import structlog

        _restore_global_logging.setup_logging()
        structlog.get_logger("probe").warning(
            "ndvi_monthly_partial_failure",
            months=3,
            failed=1,
            error=("APIError: Failed to parse: https://stac-reader:s3cret-token@stac.internal/v1"),
        )
        captured = capfd.readouterr()
        stream = captured.err + captured.out
        assert "s3cret-token" not in stream, "the script run emitted the STAC credential verbatim"
        assert "ndvi_monthly_partial_failure" in stream, "the diagnostic itself was lost"


class TestSetupLoggingSurvivesBeingCalledLate:
    """The defect the script fix exposed, and the reason it is worth its own guard.

    `PrintLoggerFactory(file=sys.stderr)` captured the stream object at configure time. Harmless
    while `src/server.py` at import was the only caller; fatal once a script's `main()` calls it
    mid-process. Under pytest, `main()` configured structlog against the per-test capture buffer,
    and every later log write in the session raised `ValueError: I/O operation on closed file` —
    69 unrelated tests failed from one snapshotted stream, none of them near the change.

    The same shape hits any process that reopens or redirects stderr after startup, so this pins
    the property rather than the symptom.
    """

    @pytest.fixture(autouse=True)
    def _restore(self):
        import logging as stdlib_logging

        import structlog

        root = stdlib_logging.getLogger()
        saved = (root.handlers[:], root.level, structlog.get_config())
        yield
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
        structlog.configure(**saved[2])

    def test_a_logger_configured_against_a_dead_stream_still_writes(self):
        import io
        import sys

        import structlog

        from src.utils import logging as geoai_logging

        doomed = io.StringIO()
        real_stderr = sys.stderr
        sys.stderr = doomed
        try:
            geoai_logging.setup_logging()  # configured while stderr is the doomed buffer
        finally:
            sys.stderr = real_stderr
        doomed.close()  # exactly what pytest does to a capture buffer when its test ends

        replacement = io.StringIO()
        sys.stderr = replacement
        try:
            structlog.get_logger("probe").warning("still_alive")
        finally:
            sys.stderr = real_stderr

        assert "still_alive" in replacement.getvalue(), (
            "the logger is stranded on the stream present when setup_logging() ran"
        )

    def test_the_stdlib_handler_also_survives_a_dead_stream(self):
        # The half the first fix missed, caught by the #164 review. `StreamHandler.emit` does not
        # raise on a dead stream — it routes to `handleError`, which prints the RAW record:
        #
        #     --- Logging error ---
        #     ValueError: I/O operation on closed file
        #     Message: 'failed pw=rustfs-admin-secret'
        #
        # So the failure is not a lost line, it is the formatter being bypassed and the
        # unformatted record published. Once the stdlib bridge routes these through
        # redact_processor, that is a redaction bypass.
        import io
        import logging as stdlib_logging
        import sys

        from src.utils import logging as geoai_logging

        doomed = io.StringIO()
        real_stderr = sys.stderr
        sys.stderr = doomed
        try:
            geoai_logging.setup_logging()
        finally:
            sys.stderr = real_stderr
        doomed.close()

        replacement = io.StringIO()
        sys.stderr = replacement
        try:
            stdlib_logging.getLogger("third_party").warning("still_alive_stdlib")
        finally:
            sys.stderr = real_stderr

        written = replacement.getvalue()
        assert "Logging error" not in written, "the stdlib handler is stranded on a dead stream"
        assert "still_alive_stdlib" in written
