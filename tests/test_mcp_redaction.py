"""#148 — credentials must not leave through a tool result or a log record.

Two boundaries, because a failing tool writes to both: `log.error(..., error=str(e))` goes to
stderr, and the same message is returned to the MCP client inside `{"error": ...}`. httpfs and
pystac_client quote the configured credential in full in those exceptions.
"""

import json

import httpx
import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from src.utils import logging as geoai_logging
from src.utils import s3_endpoint
from src.utils.mcp_redaction import CredentialRedactionMiddleware

ACCESS_KEY = "rustfs-admin"
# A superstring of the access key, so a shortest-first replacement would publish "-secret".
SECRET_KEY = "rustfs-admin-secret"
ENDPOINT = "http://rustfs:9000"


@pytest.fixture(autouse=True)
def _store_configured(monkeypatch):
    for module in (s3_endpoint, geoai_logging):
        monkeypatch.setattr(module.settings, "S3_ACCESS_KEY", ACCESS_KEY, raising=False)
        monkeypatch.setattr(module.settings, "S3_SECRET_KEY", SECRET_KEY, raising=False)
        monkeypatch.setattr(module.settings, "S3_ENDPOINT", ENDPOINT, raising=False)
    from src.utils import mcp_redaction

    monkeypatch.setattr(mcp_redaction.settings, "S3_ENDPOINT", ENDPOINT, raising=False)
    for name in s3_endpoint.AMBIENT_STORE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


HTTPFS_ERROR = (
    f"IO Error: HTTP HEAD to 'http://{ACCESS_KEY}:{SECRET_KEY}@rustfs:9000/b/k' "
    f"<AWSAccessKeyId>{ACCESS_KEY}</AWSAccessKeyId>"
)


class TestToolResultsCarryNoCredentials:
    @pytest.fixture
    def server(self):
        mcp = FastMCP("test")
        mcp.add_middleware(CredentialRedactionMiddleware())

        @mcp.tool
        def failing_tool() -> dict:
            # The shape every tool in src/tools/ uses when it catches an exception.
            return {"error": HTTPFS_ERROR, "zone_id": "montpellier"}

        return mcp

    @pytest.mark.asyncio
    async def test_neither_channel_leaks_the_credential(self, server):
        async with Client(server) as client:
            result = await client.call_tool("failing_tool", {})

        structured = json.dumps(result.structured_content)
        text = "".join(getattr(block, "text", "") for block in result.content)

        # BOTH channels: a client may read either, and they must agree.
        assert SECRET_KEY not in structured, "structured_content leaked the store password"
        assert SECRET_KEY not in text, "the text block leaked the store password"
        assert f"{ACCESS_KEY}:" not in structured
        assert f"{ACCESS_KEY}:" not in text

    @pytest.mark.asyncio
    async def test_identifiers_survive_so_the_answer_stays_usable(self, server):
        async with Client(server) as client:
            result = await client.call_tool("failing_tool", {})
        # Scoped redaction, not blanket: an identifier is not a credential, and a tool answer
        # whose zone_id has been rewritten is worse than useless to the caller.
        assert result.structured_content["zone_id"] == "montpellier"

    @pytest.mark.asyncio
    async def test_a_clean_result_is_returned_unchanged(self, server):
        @server.tool
        def clean_tool() -> dict:
            return {"rows": 3, "zone_id": "montpellier"}

        async with Client(server) as client:
            result = await client.call_tool("clean_tool", {})
        assert result.structured_content == {"rows": 3, "zone_id": "montpellier"}


class TestLogRecordsCarryNoCredentials:
    def test_the_processor_scrubs_diagnostics_and_keeps_identifiers(self):
        out = geoai_logging.redact_processor(
            None,
            "error",
            {"event": "era5_ingest_fetch_error", "zone_id": "montpellier", "error": HTTPFS_ERROR},
        )
        assert SECRET_KEY not in json.dumps(out)
        assert out["zone_id"] == "montpellier"

    def test_the_event_message_is_scrubbed_although_it_is_not_a_diagnostic_field(self):
        # `event` is structlog's own key for the message; it is not in DIAGNOSTIC_FIELDS, so
        # redact_credentials alone would leave it untouched.
        out = geoai_logging.redact_processor(None, "error", {"event": f"failed for {SECRET_KEY}"})
        assert SECRET_KEY not in out["event"]

    def test_a_rendered_traceback_is_scrubbed(self):
        # Placed after format_exc_info in the chain, so exc_info has already become this string.
        out = geoai_logging.redact_processor(
            None, "error", {"event": "boom", "exception": f"Traceback ... {SECRET_KEY}"}
        )
        assert SECRET_KEY not in out["exception"]

    def test_the_processor_never_raises(self, monkeypatch):
        # A processor that throws takes down logging for the whole process — a worse outcome than
        # the leak it guards, and it would fire exactly when something is already going wrong.
        monkeypatch.setattr(
            geoai_logging, "redact_credentials", lambda *a, **k: (_ for _ in ()).throw(RuntimeError)
        )
        record = {"event": "boom", "error": "x"}
        assert geoai_logging.redact_processor(None, "error", record) == record


class TestRaisedExceptionsCarryNoCredentials:
    """#157 review, Codex P1 — the merged middleware only touched what `call_next` RETURNED.

    FastMCP defaults to `mask_error_details=False` and this repo never overrides it, so an
    exception escaping a tool becomes `ToolError(f"Error calling tool {name!r}: {e}")` and the
    provider's own message goes to the client verbatim. Five of the fifteen tools reach a
    provider outside any catch-all (stac_search, stac_download, ndvi_profile, vegetation_stress,
    change_detect), so this is a live path, not a hypothetical one.
    """

    @pytest.fixture
    def server(self):
        mcp = FastMCP("test")
        mcp.add_middleware(CredentialRedactionMiddleware())

        @mcp.tool
        def raising_tool() -> dict:
            raise RuntimeError(HTTPFS_ERROR)

        @mcp.tool
        def raising_toolerror() -> dict:
            raise ToolError(HTTPFS_ERROR)

        @mcp.tool
        def rate_limited() -> dict:
            # The response carries the same request object, not a bare one: httpx raises
            # RuntimeError on `response.request` when it was never set, and several of httpx's
            # own error/message paths reach for it. The test passes either way today, which is
            # exactly why it is worth not leaving as a trap for whoever extends it.
            request = httpx.Request("GET", "http://x")
            raise httpx.HTTPStatusError(
                f"429 for {SECRET_KEY}",
                request=request,
                response=httpx.Response(429, request=request),
            )

        @mcp.tool
        def needs_int(n: int) -> dict:
            return {"n": n}

        return mcp

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", ["raising_tool", "raising_toolerror"])
    async def test_a_raised_exception_does_not_carry_the_credential(self, server, tool):
        async with Client(server) as client:
            result = await client.call_tool(tool, {}, raise_on_error=False)

        text = "".join(getattr(block, "text", "") for block in result.content)
        assert result.is_error, "the failure must still be reported as a failure"
        assert SECRET_KEY not in text, f"{tool} leaked the store password through the raised path"
        assert f"{ACCESS_KEY}:" not in text

    @pytest.mark.asyncio
    async def test_the_client_still_raises_and_the_raised_message_is_scrubbed(self, server):
        async with Client(server) as client:
            with pytest.raises(ToolError) as excinfo:
                await client.call_tool("raising_tool", {})
        assert SECRET_KEY not in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_an_actionable_error_is_not_flattened_into_something_useless(self, server):
        # FastMCP converts 429 and timeouts into fixed, actionable strings BEFORE any middleware
        # sees them. Scrubbing must leave those alone — the `scrubbed == str(exc)` bare re-raise.
        # Without this guard the next reader "simplifies" that branch away and the LLM stops being
        # told it was rate limited.
        async with Client(server) as client:
            result = await client.call_tool("rate_limited", {}, raise_on_error=False)
        text = "".join(getattr(block, "text", "") for block in result.content)
        assert "Rate limited by upstream API, please retry later" in text
        assert SECRET_KEY not in text

    @pytest.mark.asyncio
    async def test_an_argument_validation_error_is_left_intact(self, server):
        # pydantic's ValidationError reaches the middleware UNCONVERTED, and it cannot be rebuilt
        # from a message (`ValidationError.__new__()` requires `line_errors`). Catching
        # `Exception` and re-raising `type(exc)(scrubbed)` turns this into a TypeError. It also
        # quotes the caller's own argument, never a server-side secret, so there is nothing to
        # scrub. This guard pins `except ToolError` rather than `except Exception`.
        async with Client(server) as client:
            result = await client.call_tool("needs_int", {"n": "abc"}, raise_on_error=False)
        text = "".join(getattr(block, "text", "") for block in result.content)
        assert result.is_error
        assert "TypeError" not in text, "the middleware crashed rebuilding a ValidationError"
        assert "valid integer" in text, "the caller lost the diagnosis of their own bad argument"


class TestBothChannelsAgree:
    """The invariant this module claims: a client reading `content` and a client reading
    `structured_content` must see the same document."""

    @pytest.fixture
    def server(self):
        mcp = FastMCP("test")
        mcp.add_middleware(CredentialRedactionMiddleware())

        @mcp.tool
        def scalar_tool() -> str:
            # FastMCP wraps a non-object return as {"result": value}. `result` is not a
            # DIAGNOSTIC_FIELDS key, so the scoped pass is a no-op on it while the text block —
            # bare prose, not JSON — is scrubbed by the unscoped leaf. Measured on the shipped
            # middleware: content='pw=[REDACTED]' beside structured={'result': 'pw=rustfs-...'}.
            return HTTPFS_ERROR

        @mcp.tool
        def object_under_result() -> dict:
            # The discriminator must not fire here: a tool legitimately returning something under
            # `result` keeps the scoped pass, or an identifier that merely contains the access key
            # as a substring gets rewritten.
            return {"result": {"run_id": f"{ACCESS_KEY}-2026-08-21"}}

        return mcp

    @pytest.mark.asyncio
    async def test_a_scalar_return_is_scrubbed_in_both_channels(self, server):
        async with Client(server) as client:
            result = await client.call_tool("scalar_tool", {})
        text = "".join(getattr(block, "text", "") for block in result.content)
        structured = json.dumps(result.structured_content)
        assert SECRET_KEY not in text
        assert SECRET_KEY not in structured, "the scalar wrapper leaked while the text was scrubbed"

    @pytest.mark.asyncio
    async def test_an_object_returned_under_result_keeps_its_identifiers(self, server):
        async with Client(server) as client:
            result = await client.call_tool("object_under_result", {})
        assert result.structured_content["result"]["run_id"] == f"{ACCESS_KEY}-2026-08-21"


class TestTheBoundaryReportsWhenItFailsOpen:
    """#157 review, Copilot — failing open is right, failing open *silently* is not."""

    @pytest.fixture
    def server(self):
        mcp = FastMCP("test")
        mcp.add_middleware(CredentialRedactionMiddleware())

        @mcp.tool
        def ok_tool() -> dict:
            return {"error": HTTPFS_ERROR}

        return mcp

    @pytest.mark.asyncio
    async def test_a_broken_redactor_is_reported_without_carrying_the_payload(
        self, server, monkeypatch
    ):
        from src.utils import mcp_redaction

        def boom(*_a, **_k):
            # The hazard Copilot's suggestion walks into: the exception raised by a broken
            # redactor can quote the very payload it choked on.
            raise RuntimeError(f"choked on {SECRET_KEY}")

        monkeypatch.setattr(mcp_redaction, "redact_credentials", boom)
        records = []
        monkeypatch.setattr(
            mcp_redaction.log, "error", lambda event, **kw: records.append((event, kw))
        )

        async with Client(server) as client:
            result = await client.call_tool("ok_tool", {})

        assert result is not None, "the boundary must fail open, not drop the caller's answer"
        assert records, "the boundary failed open silently"
        event, kw = records[0]
        assert event == "credential_redaction_failed"
        assert kw["channel"] == "structured"
        assert kw["error_type"] == "RuntimeError"
        assert SECRET_KEY not in json.dumps(kw), "the telemetry became a new leak channel"

    @pytest.mark.asyncio
    async def test_a_failure_in_one_channel_does_not_skip_the_other(self, server, monkeypatch):
        # Sharing one `try` meant a structured-pass failure skipped the content pass, so BOTH
        # channels went out raw. Only the structured half may be lost.
        #
        # The fault is injected at `_scalar_return_text` and not at `redact_credentials` on
        # purpose: both passes call that primitive, so breaking it globally breaks both and
        # proves nothing about the split. What the split actually buys is survival of a fault
        # structural to ONE pass — this discriminator, or `model_copy` on a content block.
        from src.utils import mcp_redaction

        monkeypatch.setattr(
            mcp_redaction,
            "_scalar_return_text",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError),
        )
        monkeypatch.setattr(mcp_redaction.log, "error", lambda *a, **k: None)

        async with Client(server) as client:
            result = await client.call_tool("ok_tool", {})
        text = "".join(getattr(block, "text", "") for block in result.content)
        assert SECRET_KEY not in text, "the content pass was skipped by the structured failure"

    @pytest.mark.asyncio
    async def test_a_normal_result_does_not_enter_the_fail_open_branch(self, server, monkeypatch):
        from src.utils import mcp_redaction

        records = []
        monkeypatch.setattr(mcp_redaction.log, "error", lambda event, **kw: records.append(event))
        async with Client(server) as client:
            await client.call_tool("ok_tool", {})
        assert records == [], "the boundary is failing open on ordinary traffic"
