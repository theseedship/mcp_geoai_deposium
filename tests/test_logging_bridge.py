"""#159 — stdlib logging must pass through the redactor too.

`redact_processor` (#148) only sees records that go through structlog. FastMCP, pystac_client,
botocore, urllib3 and httpx all log through the stdlib module, and `logging.basicConfig` put them
on stderr untouched. The live case is FastMCP's own `logger.exception(f"Error calling tool ...")`,
which runs immediately before the raise that #158 scrubs — so without this bridge the credential
still leaves the process, just through fd 2 instead of through the client.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import sys

import pytest
import structlog
from fastmcp import Client, FastMCP

from src.utils import logging as geoai_logging
from src.utils import s3_endpoint
from src.utils.mcp_redaction import CredentialRedactionMiddleware

ACCESS_KEY = "rustfs-admin"
SECRET_KEY = "rustfs-admin-secret"
ENDPOINT = "http://rustfs:9000"
HTTPFS_ERROR = f"IO Error: HTTP HEAD to 'http://{ACCESS_KEY}:{SECRET_KEY}@rustfs:9000/b/k'"


@pytest.fixture(autouse=True)
def _store_configured_and_logging_restored(monkeypatch):
    """Configure the store, then put global logging back exactly as it was.

    `setup_logging()` rewrites root's handlers, structlog's config and the `fastmcp` logger. A
    test that leaves any of those changed silently rewires logging for the rest of the session.

    Restoring a saved handler is only sound because `setup_logging()` closes just the handlers it
    created (#165 review). While it closed everything it detached, this fixture could hand back a
    handler someone else — pytest's own logging capture, for one — was still holding, in a closed
    state. The fix belongs in `_detach_handlers`, not here: a fixture cannot reopen what it did
    not close.
    """
    for module in (s3_endpoint, geoai_logging):
        monkeypatch.setattr(module.settings, "S3_ACCESS_KEY", ACCESS_KEY, raising=False)
        monkeypatch.setattr(module.settings, "S3_SECRET_KEY", SECRET_KEY, raising=False)
        monkeypatch.setattr(module.settings, "S3_ENDPOINT", ENDPOINT, raising=False)
    for name in s3_endpoint.AMBIENT_STORE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    root = logging.getLogger()
    fastmcp_logger = logging.getLogger("fastmcp")
    saved = (
        root.handlers[:],
        root.level,
        fastmcp_logger.handlers[:],
        fastmcp_logger.propagate,
        structlog.get_config(),
    )
    yield
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    fastmcp_logger.handlers[:] = saved[2]
    fastmcp_logger.propagate = saved[3]
    structlog.configure(**saved[4])


class TestStdlibRecordsAreRedacted:
    def test_a_third_party_traceback_does_not_reach_stderr_with_the_credential(self, capfd):
        # `.exception()` renders the traceback, and the credential is on its LAST line — the
        # exception's own str(), not the log message. A fix that only rewrote record.msg would
        # pass a weaker version of this test while leaking exactly as before.
        geoai_logging.setup_logging()
        try:
            raise RuntimeError(HTTPFS_ERROR)
        except RuntimeError:
            logging.getLogger("some_third_party").exception("Error calling tool 'stac_search'")

        captured = capfd.readouterr().err
        assert SECRET_KEY not in captured, "the rendered traceback carried the store password"
        assert "Error calling tool" in captured, "the diagnostic itself was lost"

    def test_identifiers_in_a_stdlib_record_still_survive(self, capfd):
        # The same scoping guarantee structlog records get: a redactor that rewrites identifiers
        # produces logs nobody can correlate, and gets turned off.
        geoai_logging.setup_logging()
        logging.getLogger("some_third_party").warning("ingest failed for zone montpellier")
        assert "montpellier" in capfd.readouterr().err

    @pytest.mark.asyncio
    async def test_a_raising_tool_leaks_nothing_through_fastmcps_own_logger(self, capfd):
        # The end-to-end case #159 exists for. FastMCP logs the escaping exception through a
        # stdlib logger that sets propagate=False and owns its own handlers, so a bridge on root
        # alone would not have covered it.
        geoai_logging.setup_logging()

        mcp = FastMCP("test")
        mcp.add_middleware(CredentialRedactionMiddleware())

        @mcp.tool
        def boom() -> dict:
            raise RuntimeError(HTTPFS_ERROR)

        async with Client(mcp) as client:
            await client.call_tool("boom", {}, raise_on_error=False)

        captured = capfd.readouterr().err
        assert SECRET_KEY not in captured
        assert f"{ACCESS_KEY}:" not in captured


class TestTheFastmcpLoggerIsRerouted:
    def test_it_propagates_and_owns_no_handlers_after_setup(self):
        fastmcp_logger = logging.getLogger("fastmcp")
        fastmcp_logger.propagate = False
        fastmcp_logger.addHandler(logging.NullHandler())

        geoai_logging.setup_logging()

        assert fastmcp_logger.propagate is True
        assert fastmcp_logger.handlers == []

    def test_setup_is_idempotent_so_a_later_call_heals_a_reconfiguration(self):
        # fastmcp re-runs its own configure_logging() in some paths; calling setup_logging() again
        # must restore the reroute rather than stack a second handler on root.
        geoai_logging.setup_logging()
        before = len(logging.getLogger().handlers)
        geoai_logging.setup_logging()
        assert len(logging.getLogger().handlers) == before


class TestTheTemporaryLogLevelLandmineStaysDisarmed:
    """`fastmcp/server/mixins/transport.py` wraps the serve loop in `temporary_log_level(...)`.

    On a truthy `log_level` it re-runs fastmcp's `configure_logging()`, which restores
    `propagate=False` and fresh RichHandlers **for the life of the process** — silently undoing
    the reroute and returning stderr to an unredacted state. Nothing would look wrong; the leak
    would simply come back. `src/server.py` does not pass `log_level` today, and this pins it.
    """

    def test_server_never_passes_log_level_to_mcp_run(self):
        # Anchored on this file, not the working directory: a runner that starts pytest from
        # anywhere but the repo root would raise FileNotFoundError here, and — worse for a guard
        # of this kind — a `glob`-shaped version of the same mistake would simply find nothing
        # and pass.
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        source = (repo_root / "src" / "server.py").read_text()
        # Scoped to `mcp.run`, not to any `.run` (#165 review): an unrelated `.run(log_level=...)`
        # appearing in this file would otherwise fail a FastMCP-specific guard and read as a real
        # regression. Narrowing is safe here only because of the assertion below — if the server
        # object is ever renamed, this list goes empty and the guard says so instead of passing.
        run_calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "mcp"
        ]
        assert run_calls, "no mcp.run(...) call found — this guard has stopped guarding anything"
        offenders = [
            keyword.arg
            for call in run_calls
            for keyword in call.keywords
            if keyword.arg == "log_level"
        ]
        assert not offenders, (
            "mcp.run(log_level=...) re-runs fastmcp's configure_logging() and silently restores "
            "propagate=False, reverting the #159 reroute for the life of the process"
        )


def test_the_bridge_runs_format_exc_info_before_the_redactor():
    """Ordering, pinned separately because getting it wrong still produces plausible output.

    Before `format_exc_info`, a traceback is an `exc_info` tuple the redactor cannot read; the
    record renders afterwards with the credential intact. The bug would be invisible in any test
    that only checks the log *message*.
    """
    geoai_logging.setup_logging()
    formatter = logging.getLogger().handlers[0].formatter
    chain = list(formatter.foreign_pre_chain)
    # By identity, not by name: `structlog.processors.format_exc_info` is an `ExceptionRenderer`
    # INSTANCE in current structlog, so a name-based assertion silently checks nothing.
    assert structlog.processors.format_exc_info in chain
    assert geoai_logging.redact_processor in chain
    assert chain.index(structlog.processors.format_exc_info) < chain.index(
        geoai_logging.redact_processor
    )


class TestReconfiguringDoesNotLeakHandlers:
    """#163 review — `removeHandler` unlinks, it does not release.

    Invisible while `setup_logging()` ran once at server import. It now runs from every script
    entry point too, so anything a displaced handler holds — a `FileHandler`'s descriptor, a
    handler's thread — would accumulate per call.

    The #165 review then sharpened the contract: releasing must be limited to handlers this module
    created. A foreign handler belongs to whoever installed it and is still referenced by them.
    """

    def test_our_own_displaced_handler_is_closed(self):
        root = logging.getLogger()
        saved = (root.handlers[:], root.level, structlog.get_config())
        try:
            geoai_logging.setup_logging()
            ours = root.handlers[0]
            geoai_logging.setup_logging()  # displaces it
            assert ours not in root.handlers
            assert getattr(ours, "_closed", False), "our own displaced handler was never closed"
        finally:
            root.handlers[:] = saved[0]
            root.setLevel(saved[1])
            structlog.configure(**saved[2])

    def test_a_foreign_handler_is_detached_but_left_intact(self, tmp_path):
        # Closing someone else's handler is what the #165 review caught: pytest's own logging
        # capture is the concrete case, and closing it out from under the plugin means anything
        # that later restores it restores something broken. Detach, do not dispose.
        root = logging.getLogger()
        saved = (root.handlers[:], root.level, structlog.get_config())
        file_handler = logging.FileHandler(tmp_path / "foreign.log")
        root.addHandler(file_handler)
        try:
            geoai_logging.setup_logging()
            assert file_handler not in root.handlers, "the foreign handler is still attached"
            assert not getattr(file_handler, "_closed", False), (
                "a handler this module did not create was closed out from under its owner"
            )
            assert file_handler.stream is not None and not file_handler.stream.closed
        finally:
            file_handler.close()
            root.handlers[:] = saved[0]
            root.setLevel(saved[1])
            structlog.configure(**saved[2])

    def test_stderr_itself_is_never_closed(self):
        # The reason closing is safe: `StreamHandler` inherits `Handler.close`, which does not
        # touch the stream. If that ever changed, this would take stderr down with it.
        root = logging.getLogger()
        saved = (root.handlers[:], root.level, structlog.get_config())
        try:
            geoai_logging.setup_logging()
            geoai_logging.setup_logging()  # second call displaces the first handler
            assert not sys.stderr.closed
            logging.getLogger("probe").warning("still writable")
        finally:
            root.handlers[:] = saved[0]
            root.setLevel(saved[1])
            structlog.configure(**saved[2])
