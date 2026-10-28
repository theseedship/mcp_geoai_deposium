"""Structured logging configuration."""

import logging
import sys
from typing import Any

import structlog

from src.config import settings
from src.utils.s3_endpoint import redact_credentials, redact_text


def redact_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Punch every configured credential out of a log record before it is rendered.

    Only the startup event sanitised anything before this; every ``log.error(..., error=str(e))``
    site went to stderr raw, and there are 17 of those across ``src/tools/`` alone (#148 counted
    five). Patching call sites does not hold: the next one added is unredacted again, whereas a
    processor covers every site, present and future, including third-party libraries logging
    through structlog.

    Two passes, because the scoping rules differ by key. ``redact_credentials`` walks the record
    and rewrites only :data:`DIAGNOSTIC_FIELDS` — ``error``, ``errors``, ``exception``, ``detail``
    and friends — leaving ``zone_id`` and other identifiers verbatim, so a log line stays
    greppable. ``event`` is then redacted separately with ``redact_text``: it is structlog's own
    key for the message, so it is not in that set, yet it is unambiguously prose.

    Never raises. A processor that throws takes down logging for the whole process, which is a
    worse outcome than the leak it guards against — and it would fire precisely while something
    is already going wrong.

    And it fails open *silently*, which is a decision rather than an oversight. The obvious
    symmetry with :func:`src.utils.mcp_redaction._report_failure` — log that redaction failed —
    is a trap here: this processor runs on every record, so a log call from inside its own
    handler re-enters it, and the second failure logs a third, until a single ``log.error()``
    raises ``RecursionError`` out of the logging call and the process loses logging entirely.
    Staying silent is also what makes the middleware's telemetry safe to emit, since that line
    passes back through this processor while the shared primitive is broken.
    """
    try:
        redacted = redact_credentials(event_dict, settings.S3_ENDPOINT)
        event = redacted.get("event")
        if isinstance(event, str):
            redacted["event"] = redact_text(event, settings.S3_ENDPOINT)
        return redacted
    except Exception:  # pragma: no cover - defensive; logging must survive anything
        return event_dict


class _CurrentStderr:
    """``sys.stderr`` resolved at WRITE time rather than at configure time.

    ``PrintLoggerFactory(file=sys.stderr)`` captures the stream object when
    :func:`setup_logging` runs and holds it for the life of the process. That is invisible while
    the only caller is ``src/server.py`` at import, and becomes a real defect the moment anything
    calls :func:`setup_logging` later — which the script entry points now do (#161 review).

    Measured: with pytest capturing output, a script's ``main()`` configured structlog against the
    per-test capture buffer, and every subsequent log write in the session raised
    ``ValueError: I/O operation on closed file`` once that test ended — 69 unrelated failures from
    one snapshotted stream. The same shape would bite any process that reopens or redirects stderr
    after startup.
    """

    def write(self, message: str) -> int:
        return sys.stderr.write(message)

    def flush(self) -> None:
        sys.stderr.flush()


def setup_logging() -> None:
    """Configure structured logging with structlog.

    Safe to call more than once and at any point in a process's life: the stream is resolved per
    write (see :class:`_CurrentStderr`), so a later call cannot strand the logger on a stale
    stderr.
    """

    # Determine log level
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Configure structlog
    if settings.LOG_FORMAT == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # AFTER format_exc_info, which turns exc_info into the rendered `exception` string:
            # before it, the traceback is a tuple the redactor cannot read. BEFORE the renderer,
            # which serialises the record and is the last chance to touch a field.
            redact_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=_CurrentStderr()),
        cache_logger_on_first_use=True,
    )

    # Bridge STDLIB logging into the same processor chain (#159), on the same lazily-resolved
    # stream (#164 review), by explicit handler.
    #
    # The processor above only sees records that go through structlog; FastMCP, pystac_client,
    # botocore, urllib3 and httpx all log through the stdlib module, and
    # `logging.basicConfig(format="%(message)s")` — what stood here — put them on stderr
    # untouched. Not hypothetical: `fastmcp/server/server.py` runs
    # `logger.exception(f"Error calling tool {name!r}")` immediately before the raise that #158
    # scrubs, and `.exception()` renders the traceback, whose last line is the provider's own
    # message with the credential in it. Measured: the store password appeared on fd 2 in full.
    #
    # `format_exc_info` before `redact_processor` in the foreign chain for the same reason it is
    # ordered that way above — before it, the traceback is a tuple the redactor cannot read.
    #
    # `_CurrentStderr()` rather than `sys.stderr`, and an explicit handler rather than
    # `basicConfig`, because that call had two defects: it snapshots the stream, and it is a
    # no-op once root already owns a handler, so a second `setup_logging()` could not repair the
    # first one's stale binding. Those two matter MORE here than they did before this bridge
    # existed. `StreamHandler.emit` does not raise on a dead stream — it routes to `handleError`,
    # which prints the RAW record:
    #
    #     --- Logging error ---
    #     ValueError: I/O operation on closed file
    #     Message: 'failed pw=rustfs-admin-secret'
    #
    # which is to say it bypasses the formatter. With the formatter now carrying the redaction,
    # a stale stream is not a cosmetic fault but a redaction bypass.
    stdlib_handler = logging.StreamHandler(_CurrentStderr())
    stdlib_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.format_exc_info,
                redact_processor,
            ],
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )
    setattr(stdlib_handler, _OWNED, True)
    root_logger = logging.getLogger()
    _detach_handlers(root_logger)
    root_logger.addHandler(stdlib_handler)
    root_logger.setLevel(log_level)

    _reroute_fastmcp_logger()


_OWNED = "_geoai_owned_handler"


def _detach_handlers(logger: logging.Logger) -> None:
    """Remove every handler from ``logger``, and close only the ones this module created.

    ``removeHandler`` alone unlinks without releasing, so a ``FileHandler`` keeps its descriptor
    and some handlers keep a thread. :func:`setup_logging` now runs from every script entry point
    rather than once at server import, so what leaks per call accumulates — which is why closing
    is done at all.

    Closing only OUR handlers is the second half, and the review of #163 is what surfaced it: a
    handler installed by someone else is someone else's resource, and they still hold a reference
    to it. pytest's logging capture is the concrete case — closing it out from under the plugin
    means anything that later restores it restores something broken. Foreign handlers are
    therefore detached and left intact; their owner reattaches or disposes of them.

    Failures are swallowed: a handler that objects to being closed must not take down logging
    configuration, which is the thing standing between an exception and an unredacted stderr.
    """
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        if not getattr(handler, _OWNED, False):
            continue
        try:
            handler.close()
        except Exception:  # pragma: no cover - defensive
            pass


def _reroute_fastmcp_logger() -> None:
    """Send FastMCP's own records through the root handler, and therefore through the redactor.

    ``fastmcp.utilities.logging.configure_logging`` sets ``propagate = False`` and installs its
    own ``RichHandler``s at import, so a bridge on the root logger alone never sees a FastMCP
    record — which is precisely the logger that emits the credential-bearing tool-error traceback.

    Two things this costs, both accepted: rich colour tracebacks in dev become a rendered
    ``exception`` field, which is more greppable and, at ``LOG_FORMAT=json``, the only correct
    option; and a ``RichHandler`` ignores ``record.exc_text``, so the filter-based alternatives
    could not have worked here anyway.

    **Landmine.** ``fastmcp/server/mixins/transport.py`` wraps the serve loop in
    ``temporary_log_level(log_level)``, which re-runs ``configure_logging()`` and would silently
    restore ``propagate=False`` plus fresh handlers for the life of the process. It only fires on
    a truthy ``log_level``, and ``src/server.py`` does not pass one — the guard in
    ``tests/test_logging_bridge.py`` pins that, because the failure would be a silent regression
    to an unredacted stderr rather than anything visible.
    """
    fastmcp_logger = logging.getLogger("fastmcp")
    _detach_handlers(fastmcp_logger)
    fastmcp_logger.propagate = True
