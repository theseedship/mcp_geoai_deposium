"""Credential redaction on the MCP tool boundary (#148, error path from the #157 review).

The structlog processor in :mod:`src.utils.logging` covers what this server *logs* through
structlog. It does not cover what this server *returns*: a tool that catches an exception and
answers ``{"error": str(e)}`` hands that string to the client, and the exceptions in question come
back from DuckDB httpfs and ``pystac_client``, which quote the configured access key and a private
``STAC_URL`` in full.

One middleware rather than seventeen call sites. ``src/tools/`` alone has 17 ``error=str(e)``
constructions today and gains one with every new stage, so a per-site fix is unredacted again at
the next commit. This is the single place every tool result — returned *or raised* — passes
through.

Scope, stated honestly because the previous wording overclaimed: this covers ``on_call_tool``.
Resources (``@mcp.resource``) and prompts (``@mcp.prompt``) are registered on the same server and
bypass it entirely. Nothing leaks there today — no resource handler reads a credential-bearing
setting — but the day one does, this module is not what saves it.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware

from src.config import settings
from src.utils.s3_endpoint import redact_credentials, redact_text

log = structlog.get_logger(__name__)


def _redact_text_block(block: Any) -> Any:
    """One content block, scrubbed the same way its structured twin is.

    A block's ``text`` is normally the JSON serialization of ``structured_content``. Parsing it
    back and running the SAME ``redact_credentials`` keeps the two channels agreeing: scrubbing
    the text with the unscoped :func:`redact_text` while the structured half is scoped would make
    a client reading one see a different document from a client reading the other.

    A block that is not JSON is prose by elimination, so the unscoped leaf is right for it.
    """
    text = getattr(block, "text", None)
    if not isinstance(text, str) or not text:
        return block
    try:
        parsed = json.loads(text)
    except ValueError:
        scrubbed_text = redact_text(text, settings.S3_ENDPOINT)
        return block if scrubbed_text == text else block.model_copy(update={"text": scrubbed_text})
    scrubbed = redact_credentials(parsed, settings.S3_ENDPOINT)
    if scrubbed == parsed:
        return block
    return block.model_copy(update={"text": json.dumps(scrubbed)})


def _scalar_return_text(structured: Any, content: Any) -> str | None:
    """The string a tool returned as a bare scalar, or ``None`` if this is a real object.

    FastMCP wraps a non-object return in ``{"result": value}``. ``result`` is not a
    :data:`DIAGNOSTIC_FIELDS` key, so the scoped pass is a no-op on that wrapper while the text
    block — bare prose, not JSON — is scrubbed by the unscoped leaf. Measured on the shipped
    middleware, a ``-> str`` tool therefore returned ``content='pw=[REDACTED]'`` alongside
    ``structured_content={'result': 'pw=rustfs-admin-secret'}``: the two channels disagreeing, in
    the one way this module exists to prevent.

    The discriminator has to be exact, because widening the scoped pass to cover ``result`` would
    corrupt every tool that legitimately returns an object under that key. A *wrapped scalar* is
    the case where the single content block's raw text is byte-identical to the wrapper's value; a
    tool that genuinely returns ``{"result": "..."}`` serialises its block as JSON, so the texts
    differ and this returns ``None``.
    """
    if not isinstance(structured, dict) or set(structured) != {"result"}:
        return None
    value = structured["result"]
    if not isinstance(value, str):
        return None
    texts = [getattr(block, "text", None) for block in (content or [])]
    return value if len(texts) == 1 and texts[0] == value else None


class CredentialRedactionMiddleware(Middleware):
    """Punch configured credentials out of every tool result before it reaches the client."""

    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        try:
            result = await call_next(context)
        except ToolError as exc:
            # The raised path, which the first cut of this middleware left wide open: everything
            # below only runs on a value `call_next` RETURNS. FastMCP's default is
            # `mask_error_details=False`, and this repo never overrides it, so server.py's handler
            # does `raise ToolError(f"Error calling tool {name!r}: {e}")` — the provider's own
            # message, verbatim, to the client. Five of the fifteen tools reach a provider
            # outside any catch-all and take this path.
            #
            # `except ToolError` and not `except Exception`, deliberately. FastMCP converts inside
            # `call_next`, so by the time we are reached the 429 and timeout special-cases have
            # already become their fixed strings and cannot be defeated here. What still arrives
            # unconverted is a pydantic `ValidationError` from argument parsing — which quotes the
            # CALLER'S OWN argument, never a server-side secret, and which cannot be rebuilt from
            # a message anyway (`ValidationError.__new__()` requires `line_errors`; reconstructing
            # it raises TypeError). Letting it through untouched is both correct and safe.
            scrubbed = redact_text(str(exc), settings.S3_ENDPOINT)
            if scrubbed == str(exc):
                raise
            # `from None`, not `from exc`: the cause chain still holds the unscrubbed message, and
            # anything upstream that renders a traceback would republish it. Nothing is lost —
            # FastMCP already logged the full exception server-side before raising.
            raise ToolError(scrubbed, log_level=getattr(exc, "log_level", 40)) from None

        # Two passes, each guarded on its own. Sharing one `try` meant a failure in the structured
        # pass skipped the content pass entirely, sending BOTH channels out raw.
        structured = getattr(result, "structured_content", None)
        content = getattr(result, "content", None)
        try:
            if structured is not None:
                scalar = _scalar_return_text(structured, content)
                if scalar is not None:
                    scrubbed_scalar = redact_text(scalar, settings.S3_ENDPOINT)
                    if scrubbed_scalar != scalar:
                        result.structured_content = {"result": scrubbed_scalar}
                else:
                    scrubbed = redact_credentials(structured, settings.S3_ENDPOINT)
                    if scrubbed is not structured and scrubbed != structured:
                        result.structured_content = scrubbed
        except Exception as exc:  # pragma: no cover - defensive
            _report_failure(context, "structured", exc)
        try:
            if content:
                result.content = [_redact_text_block(block) for block in content]
        except Exception as exc:  # pragma: no cover - defensive
            _report_failure(context, "content", exc)
        return result


def _report_failure(context: Any, channel: str, exc: Exception) -> None:
    """Say that the boundary failed open, carrying no part of what it choked on.

    Deliberately never ``str(exc)``, ``repr(exc)`` or ``exc_info``: the exception raised by a
    broken redactor can quote the very payload it failed to scrub, and on exactly this path the
    processor in :mod:`src.utils.logging` is failing open too — so anything carried here would be
    emitted raw. The tool name, the channel and the exception *type* are enough for an operator to
    see which boundary is open and why, and none of them is attacker-influenced.

    Never raises. A middleware wrapped around every tool cannot afford telemetry that can throw.
    """
    try:
        message = getattr(context, "message", None)
        log.error(
            "credential_redaction_failed",
            tool=getattr(message, "name", None),
            channel=channel,
            error_type=type(exc).__name__,
        )
    except Exception:  # pragma: no cover - defensive
        pass
