"""Edge cases of the shared ``S3_ENDPOINT`` parser the panel/CLI tests cannot reach.

The end-to-end guarantees live where they matter — no credential in a published run manifest
(``test_climate_baseline_panel.py``, ``test_ndvi_pilot_panel.py``, ``test_unified_pilot_panel.py``)
and no credential in a CLI error message (``test_forecast_ensemble_archive.py``). This file only
pins the shapes those callers do not exercise.
"""

from __future__ import annotations

import itertools
import json
import re
from urllib.parse import quote, urlsplit

import pytest

from src.utils import s3_endpoint
from src.utils.s3_endpoint import (
    REDACTED,
    STAC_REDUCTIONS,
    STAC_REFUSED,
    configured_store_secrets,
    effective_store_values,
    endpoint_credentials,
    publication_secrets,
    redact_credentials,
    redact_text,
    remember_effective_store_credentials,
    sanitize_endpoint,
    sanitize_stac_url,
    url_secrets,
)

# --------------------------------------------------------------------------- #
# The ACCEPT grammar: what is admitted, verbatim, and nothing else
# --------------------------------------------------------------------------- #

_ACCEPTED = [
    # scheme + host + port
    "http://rustfs:9000",
    "https://mesh.internal:9000",
    # scheme + host, no port
    "https://s3.example.com",
    # no scheme at all: DuckDB's own `SET s3_endpoint` takes a bare `host:port`
    "mesh.internal:9000",
    "mesh.internal",
    # protocol-relative, in its one normalised spelling
    "//mesh.internal:9000",
    # dotted IPv4, which the hostname production subsumes
    "http://127.0.0.1:9000",
    "127.0.0.1",
    # bracketed IPv6, balanced
    "http://[::1]:9000",
    "https://[2001:db8::1]",
    "[::1]:9000",
    # port bounds
    "http://host:1",
    "http://host:65535",
    # a hyphen inside a label, and a long dotted name
    "http://rustfs-admin-store.mesh.internal:9000",
]


@pytest.mark.parametrize("endpoint", _ACCEPTED)
def test_an_endpoint_inside_the_grammar_is_returned_verbatim(endpoint):
    """Refusing everything would satisfy every safety assertion and break every deployment."""
    assert sanitize_endpoint(endpoint) == (endpoint, False)


def test_the_scheme_is_case_insensitive_and_normalised_down():
    # RFC 3986 schemes are case-insensitive. A case-sensitive strip downstream left
    # 'HTTP://rustfs:9000' whole and handed httpfs a scheme as part of the hostname.
    assert sanitize_endpoint("HTTP://rustfs:9000") == ("http://rustfs:9000", True)
    assert sanitize_endpoint("HttPs://rustfs:9000") == ("https://rustfs:9000", True)


def test_the_flag_says_the_configured_value_differed():
    # `True` means the configured value is not the returned one: refused, its scheme case
    # normalised, or its single trailing slash dropped.
    assert sanitize_endpoint("http://admin:secret@mesh.internal:9000") == (None, True)
    assert sanitize_endpoint("http://mesh.internal:9000")[1] is False


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("http://rustfs:9000/", "http://rustfs:9000"),
        ("https://s3.example.com/", "https://s3.example.com"),
        ("HTTP://RustFS:9000/", "http://RustFS:9000"),
        ("//mesh.internal:9000/", "//mesh.internal:9000"),
        ("mesh.internal:9000/", "mesh.internal:9000"),
        ("http://127.0.0.1:9000/", "http://127.0.0.1:9000"),
        ("http://[::1]:9000/", "http://[::1]:9000"),
        ("http://rustfs/", "http://rustfs"),
    ],
)
def test_one_trailing_slash_after_a_valid_authority_is_accepted_and_stripped(endpoint, expected):
    """The single deliberate exception to "no path", and it is a deviation worth defending.

    `main`'s `duckdb_conn` did `endpoint.rstrip("/")`, so `S3_ENDPOINT=http://rustfs:9000/` is a
    deployment that RUNS today. The slash sits after an authority the grammar has already
    accepted, so there is no second reading of those bytes; and it cannot hide a credential,
    because a credential needs a `:` or an `@` and both are validated inside the authority first.
    Refusing it would turn a working, safe configuration into a startup failure.
    """
    assert sanitize_endpoint(endpoint) == (expected, True)


@pytest.mark.parametrize(
    "endpoint",
    [
        # More than one slash is not a spelling of the same address.
        "http://rustfs:9000//",
        "http://rustfs:9000///",
        # Anything after the slash is a path, with or without a slash of its own at the end.
        "http://rustfs:9000/bucket",
        "http://rustfs:9000/bucket/",
        "http://rustfs:9000/?x=1",
        "http://rustfs:9000/#f",
        # The strip runs AFTER the authority is accepted, so it cannot rescue a bad one.
        "http://admin:secret@rustfs:9000/",
        "http://admin:secret/",
        "http://rustfs:99999/",
        "http://rustfs:abc/",
        "http://[::1:9000/",
        "ftp://rustfs:9000/",
        "http:///",
        "/",
    ],
)
def test_everything_other_than_that_one_slash_stays_refused(endpoint):
    """The exception is bounded on every side: one slash, and only after a valid authority."""
    assert sanitize_endpoint(endpoint) == (None, True)


def test_credentials_are_offered_raw_and_decoded_for_redaction():
    # botocore prints the decoded form; the URL carries the encoded one. A redactor needs both.
    assert endpoint_credentials("http://admin:p%40ss@mesh.internal:9000") == (
        "admin",
        "admin",
        "p%40ss",
        "p@ss",
    )
    assert endpoint_credentials("http://mesh.internal:9000") == ()
    assert endpoint_credentials(None) == ()


# --------------------------------------------------------------------------- #
# Hostile endpoint shapes: the secret never reaches the output
# --------------------------------------------------------------------------- #

_ACCOUNT = "rustfs-admin"
_SECRET = "sup3r-s3cret"
# The two halves of the secret either side of a delimiter that RFC 3986 forbids in userinfo.
_SECRET_HEAD, _SECRET_TAIL = _SECRET[:5], _SECRET[5:]

# Every way the configured value has been seen — or can be crafted — to smuggle the userinfo past
# a parser that only knows "split the authority on the first slash, then split on the last @".
_HOSTILE_ENDPOINTS = [
    # The plain shape, for reference.
    f"http://{_ACCOUNT}:{_SECRET}@mesh.internal:9000",
    # Protocol-relative: no "://" separator, so the authority used to parse as a path.
    f"//{_ACCOUNT}:{_SECRET}@mesh.internal:9000",
    f"///{_ACCOUNT}:{_SECRET}@mesh.internal:9000",
    # Empty authority: the userinfo slid past the first "/" into the path.
    f"http:///{_ACCOUNT}:{_SECRET}@mesh.internal",
    f"http:////{_ACCOUNT}:{_SECRET}@mesh.internal",
    # A query or a fragment carrying a second copy of the credentials.
    f"http://{_ACCOUNT}:{_SECRET}@mesh.internal:9000?debug={_ACCOUNT}:{_SECRET}",
    f"http://{_ACCOUNT}:{_SECRET}@mesh.internal:9000#{_ACCOUNT}:{_SECRET}",
    f"http://mesh.internal:9000?token={_SECRET}",
    f"http://mesh.internal:9000/?token={_SECRET}",
    f"http://mesh.internal:9000#{_SECRET}",
    # Percent-encoded delimiter: reads as a bare hostname to a parser that only knows "@".
    f"http://{_ACCOUNT}:{_SECRET}%40mesh.internal:9000",
    f"//{_ACCOUNT}:{_SECRET}%40mesh.internal:9000",
    # Several "@": splitting on the first one would leave a credential inside the host.
    f"http://a:b@{_ACCOUNT}:{_SECRET}@mesh.internal:9000",
    f"http://{_ACCOUNT}:{_SECRET}@a:b@mesh.internal:9000",
    # Surrounding whitespace, as an .env file produces it.
    f"  http://{_ACCOUNT}:{_SECRET}@mesh.internal:9000\n",
    # Scheme case is not normalised anywhere else either.
    f"HTTP://{_ACCOUNT}:{_SECRET}@MESH.INTERNAL:9000",
    # No scheme at all: DuckDB's own `SET s3_endpoint` takes a bare `host:port`.
    f"{_ACCOUNT}:{_SECRET}@mesh.internal:9000",
    # Credentials spelled into the path rather than the userinfo.
    f"http://mesh.internal:9000/{_ACCOUNT}:{_SECRET}@bucket",
    # An unencoded "/", "?" or "#" INSIDE the password. RFC 3986 forbids all three in userinfo,
    # but an env var carries a raw password and nothing re-encodes it. The authority then ends
    # at that character, so `account:password-prefix` parsed as a host:port and got published.
    f"http://{_ACCOUNT}:{_SECRET_HEAD}/{_SECRET_TAIL}@mesh.internal:9000",
    f"http://{_ACCOUNT}:{_SECRET_HEAD}?{_SECRET_TAIL}@mesh.internal:9000",
    f"http://{_ACCOUNT}:{_SECRET_HEAD}#{_SECRET_TAIL}@mesh.internal:9000",
    f"{_ACCOUNT}:{_SECRET_HEAD}/{_SECRET_TAIL}@mesh.internal:9000",
    f"http://{_ACCOUNT}:{_SECRET_HEAD}/{_SECRET_TAIL}%40mesh.internal:9000",
    # Everything at once.
    f"http://{_ACCOUNT}:{_SECRET}@mesh.internal:9000/rustfs?x={_SECRET}#{_SECRET}",
]


# Shapes beyond the reported list, built against the ACCEPT grammar rather than against the
# recovery it replaced. Each was reachable through the old parser; the port ones are the hole it
# never had at all, because it validated the host and never the port.
_HOSTILE_ENDPOINTS += [
    # No "@" ANYWHERE, so the userinfo delimiter cannot help: the password reads as the port and
    # was configured into httpfs, which quotes `s3_endpoint` verbatim in every error it raises.
    f"http://{_ACCOUNT}:{_SECRET}",
    f"{_ACCOUNT}:{_SECRET}",
    f"//{_ACCOUNT}:{_SECRET}",
    f"https://{_ACCOUNT}:{_SECRET}",
    # An unknown scheme was neither stripped nor refused, so it reached httpfs whole -- and a
    # scheme-shaped credential prefix rode in with it.
    "ftp://rustfs:9000",
    f"s3://{_ACCOUNT}:{_SECRET}@mesh.internal:9000",
    f"{_ACCOUNT}:{_SECRET}://mesh.internal:9000",
    # A leading slash is not an address; the old parser stripped it and read the rest as a host.
    f"/{_SECRET}",
    f"//{_SECRET}/x",
    # Malformed IPv6: unbalanced, trailing bytes, or a scope id.
    f"[::1]{_SECRET}",
    "http://[::1:9000",
    "http://::1]:9000",
    "http://[fe80::1%eth0]:9000",
    "http://[not-an-address]:9000",
    # Ports that are not ports.
    "http://host:99999",
    "http://host:abc",
    "http://host:0",
    "http://host:-1",
    "http://host:65536",
    "http://host:09000",
    "http://host:9000abc",
    "http://host: 9000",
    # `str.isdigit()` is True for Arabic-Indic digits and `int()` then returns 9000 happily.
    "http://host:\u0669\u0660\u0660\u0660",
    # A path is a configuration mistake on an endpoint. The SINGLE trailing slash is the one
    # bounded exception and lives in its own test below; everything else with a path stays here.
    f"http://mesh.internal:9000/{_SECRET}",
    "http://mesh.internal:9000/bucket",
    "http://mesh.internal:9000//",
    "http://mesh.internal:9000/bucket/",
    # Hostname productions that must not be admitted.
    "http://mesh..internal:9000",
    "http://mesh.internal.:9000",
    "http://-mesh.internal:9000",
    "http://mesh.internal-:9000",
    "http://me sh.internal:9000",
    "http://mesh_internal:9000",
    f"http://mesh.internal:9000@{_ACCOUNT}",
]


@pytest.mark.parametrize("endpoint", _HOSTILE_ENDPOINTS)
def test_every_hostile_shape_is_refused_whole_rather_than_repaired(endpoint):
    """The posture, asserted directly: outside the grammar means no address at all.

    "Refused" and "sanitized to something secret-free" are not the same guarantee, and the
    difference is the whole review. Four rounds of recovery-style parsing each returned a
    shortened value that passed the secret-free assertion of its round, and each round's
    shortened value turned out to be a credential, a credential PREFIX, or an address the
    operator never configured.
    """
    address, differed = sanitize_endpoint(endpoint)

    assert address is None
    assert differed is True


@pytest.mark.parametrize("endpoint", _HOSTILE_ENDPOINTS)
def test_no_hostile_shape_leaks_the_secret_or_any_prefix_of_it(endpoint):
    sanitized, _differed = sanitize_endpoint(endpoint)

    assert _SECRET not in (sanitized or "")
    assert _ACCOUNT not in (sanitized or "")
    # A PREFIX, not only the whole secret. A password split across a "/" published its first
    # half as if it were a port — the whole secret is indeed absent, so an assertion that only
    # looks for the whole secret is blind to it.
    for length in range(4, len(_SECRET) + 1):
        assert _SECRET[:length] not in (sanitized or "")
        assert _ACCOUNT[:length] not in (sanitized or "")


@pytest.mark.parametrize(
    "endpoint",
    [
        e
        for e in _HOSTILE_ENDPOINTS
        if f"{_ACCOUNT}:{_SECRET}@" in e or f"{_ACCOUNT}:{_SECRET}%40" in e
    ],
)
def test_the_account_name_goes_with_the_password(endpoint):
    # The username alone identifies the store's account; publishing it is the same mistake.
    sanitized, removed = sanitize_endpoint(endpoint)

    assert _ACCOUNT not in (sanitized or "")
    assert removed is True


@pytest.mark.parametrize(
    ("endpoint", "note"),
    [
        (f"//{_ACCOUNT}:{_SECRET}@mesh.internal:9000", "protocol-relative userinfo"),
        (f"http:///{_ACCOUNT}:{_SECRET}@mesh.internal", "empty authority"),
        (f"http://{_ACCOUNT}:{_SECRET}%40mesh.internal:9000", "encoded delimiter"),
        (f"http://a:b@{_ACCOUNT}:{_SECRET}@mesh.internal:9000", "two userinfos"),
        (
            f"http://{_ACCOUNT}:{_SECRET}@mesh.internal:9000/rustfs?x={_SECRET}#{_SECRET}",
            "userinfo, path, query and fragment at once",
        ),
        (f"http://{_ACCOUNT}:{_SECRET}@mesh.internal:9000/rustfs@v2", "userinfo plus path @"),
        (f"http://{_ACCOUNT}:{_SECRET}@mesh.internal:9000/a@b?c@d#e@f", "@ everywhere"),
        ("http://mesh.internal:9000/rustfs", "a plain path, no credential anywhere"),
    ],
)
def test_a_recoverable_looking_address_is_refused_rather_than_recovered(endpoint, note):
    """These used to sanitize to `http://mesh.internal:9000` and be configured.

    The recovered value was even correct for most of them. It is refused anyway: "I can see
    which part of this is the address" is exactly the reasoning that published
    `http://rustfs-admin:sup3r` when the password held a slash, and the last row shows the price
    -- a credential-free endpoint with a path is refused too, because the grammar does not have
    a path in it and carving one off is the behaviour under review.
    """
    assert sanitize_endpoint(endpoint) == (None, True)


def test_a_split_secret_is_still_offered_to_the_redactors():
    """`endpoint_credentials` returned `()` for this shape, so every redactor was a no-op on it.

    httpfs and botocore both quote the configured value verbatim, so the string a redactor has
    to match is `rustfs-admin:sup3r/s3cret@...` — the components have to include the half the
    authority split off.
    """
    endpoint = f"http://{_ACCOUNT}:{_SECRET_HEAD}/{_SECRET_TAIL}@mesh.internal:9000"

    assert endpoint_credentials(endpoint) == (
        _ACCOUNT,
        _ACCOUNT,
        f"{_SECRET_HEAD}/{_SECRET_TAIL}",
        f"{_SECRET_HEAD}/{_SECRET_TAIL}",
    )
    scrubbed = redact_credentials({"error": f"HTTP HEAD to '{endpoint}/d/z1.parquet'"}, endpoint)
    assert _SECRET_HEAD not in scrubbed["error"]
    assert _ACCOUNT not in scrubbed["error"]


def test_an_endpoint_that_is_nothing_but_userinfo_is_null_not_empty():
    # "" would read like a configured address; there is no address left in "admin:secret@".
    assert sanitize_endpoint(f"{_ACCOUNT}:{_SECRET}@") == (None, True)


def test_a_percent_encoded_delimiter_still_yields_both_redaction_spellings():
    # The archive CLI redacts on these; missing the encoded delimiter meant redacting nothing.
    assert endpoint_credentials("http://admin:p%40ss%40mesh.internal:9000") == (
        "admin",
        "admin",
        "p%40ss",
        "p@ss",
    )


class TestRedactCredentials:
    """`sanitize_endpoint` cleans the address field; this cleans the fields that hold prose."""

    ENDPOINT = "http://rustfs-admin:sup3r-s3cret@mesh.internal:9000"

    def test_a_nested_payload_is_scrubbed_wherever_the_secret_hides(self):
        payload = {
            "errors": [{"zone_id": "z1", "error": f"HTTP HEAD to '{self.ENDPOINT}/d/z1.parquet'"}],
            "checks": {"note": ("tuple", f"{self.ENDPOINT}")},
        }

        scrubbed = redact_credentials(payload, self.ENDPOINT)

        # Serialized, because a credential hides in whatever key no assertion happens to read.
        assert "sup3r-s3cret" not in json.dumps(scrubbed, default=str)
        assert "rustfs-admin" not in json.dumps(scrubbed, default=str)
        assert "mesh.internal:9000" in scrubbed["errors"][0]["error"]

    def test_dict_keys_are_left_alone_so_a_short_credential_cannot_rewrite_the_schema(
        self, monkeypatch
    ):
        """Keys are code-controlled; scrubbing them buys nothing and costs the manifest schema.

        `S3_ACCESS_KEY` is whatever the deployment set it to, and it can collide with a key name
        this repo writes. Scrubbing keys made every key that CONTAINS one a different key: with
        `S3_ACCESS_KEY='s3_endpoint'` this payload came back as `{'[REDACTED]': …}`, losing the
        field outright. Those keys become COLUMN NAMES in the `COPY` that publishes a manifest, and
        `[REDACTED]_endpoint` does not match the `^[A-Za-z_][A-Za-z0-9_]*$` column check in
        `scripts/unified_pilot_panel.py` / `scripts/climate_baseline_panel.py` — so a short
        credential turned a redaction into a failed publication, or into a manifest whose
        consumer can no longer find `run_id`.

        A key can never hold the credential in the first place: they are literals in this repo's
        writers. The values under them are the prose, and those are still scrubbed, recursively.
        """
        # Eleven characters, so it clears `MIN_SECRET_LENGTH` and is a genuine pattern — and it
        # is character-for-character a key this repo's writers emit.
        monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", "s3_endpoint")
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", "")

        scrubbed = redact_credentials(
            {
                "s3_endpoint": "http://mesh.internal:9000",
                "s3_key": {"note": "sent to s3_endpoint twice"},
            },
            None,
        )

        assert list(scrubbed) == ["s3_endpoint", "s3_key"]
        # …and the recursion still reaches the diagnostic values under those untouched keys.
        assert scrubbed["s3_key"]["note"] == "sent to [REDACTED] twice"

    def test_a_structural_field_is_never_rewritten(self, monkeypatch):
        """The boundary is non-destructive: it redacts prose and touches nothing else.

        Measured on the dev store's own configuration, where `S3_ACCESS_KEY=deposium-dev` is a
        substring of the run id AND of every URI built from it. Rewriting a whole payload turned
        a redaction into a corruption: the run id stopped matching the run, and every URI in the
        manifest pointed at an object that does not exist.
        """
        monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", "deposium-dev")
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", "")
        payload = {
            "run_id": "deposium-dev-2026-08-21",
            "run_manifest_uri": (
                "s3://deposium/climate/runs/deposium-dev-2026-08-21/manifest.json"
            ),
            "code_ref": "deposium-dev@abc1234",
            "outputs": [
                {"uri": "s3://deposium/climate/unified/deposium-dev/z.parquet", "kind": "unified"}
            ],
            "inputs": [{"source": "https://deposium-dev.example/v1"}],
            "errors": [{"zone_id": "deposium-dev-zone", "error": "403 for deposium-dev"}],
        }

        scrubbed = redact_credentials(payload, None)

        assert scrubbed["run_id"] == "deposium-dev-2026-08-21"
        assert scrubbed["run_manifest_uri"] == payload["run_manifest_uri"]
        assert scrubbed["code_ref"] == "deposium-dev@abc1234"
        assert scrubbed["outputs"] == payload["outputs"]
        assert scrubbed["inputs"] == payload["inputs"]
        # The scope resets at each dict: `errors` opens it for `error`, not for the sibling id.
        assert scrubbed["errors"][0]["zone_id"] == "deposium-dev-zone"
        assert scrubbed["errors"][0]["error"] == "403 for [REDACTED]"

    def test_a_bare_list_of_diagnostics_is_covered(self):
        # A list inherits the scope its key opened; only a dict resets it.
        scrubbed = redact_credentials(
            {"warnings": ["retrying for sup3r-s3cret", "gave up"]}, self.ENDPOINT
        )

        assert scrubbed["warnings"] == ["retrying for [REDACTED]", "gave up"]

    def test_a_secret_in_a_nested_diagnostic_is_still_scrubbed(self):
        # Prose is where both httpfs and botocore quote the endpoint, at whatever depth.
        scrubbed = redact_credentials(
            {"checks": {"note": {"detail": "host_sup3r-s3cret"}}}, self.ENDPOINT
        )

        assert "sup3r-s3cret" not in json.dumps(scrubbed)

    def test_the_percent_encoded_spelling_is_caught_too(self):
        """botocore prints the userinfo decoded; the URL carries it encoded. Both must match."""
        endpoint = "http://rustfs-admin:sup3r%2Ds3cret@mesh.internal:9000"

        scrubbed = redact_credentials(
            {"error": "failed for sup3r-s3cret and sup3r%2Ds3cret"}, endpoint
        )

        assert "sup3r-s3cret" not in scrubbed["error"]
        assert "sup3r%2Ds3cret" not in scrubbed["error"]

    def test_non_string_leaves_keep_their_type(self):
        """The scrubbed payload is handed straight to DuckDB's parameter binding."""
        payload = {"rows": 12, "ok": True, "months": None, "ratio": 0.5}

        assert redact_credentials(payload, self.ENDPOINT) == payload

    def test_a_credential_free_endpoint_is_a_no_op(self):
        payload = {"error": "IO Error: HTTP HEAD to 'http://mesh.internal:9000/d/z1.parquet'"}

        assert redact_credentials(payload, "http://mesh.internal:9000") == payload
        assert redact_credentials(payload, "") == payload
        assert redact_credentials(payload, None) == payload


def test_an_account_name_that_prefixes_the_password_does_not_leave_its_tail():
    """This repo's own dev store: account `deposium-dev`, password `deposium-dev-secret`.

    Redacting in parser order rewrote the password to `[REDACTED]-secret` — the whole secret is
    gone, so every "is the secret still in there" assertion passes, while the tail is published.
    Longest match first is what actually removes it.
    """
    endpoint = "http://deposium-dev:deposium-dev-secret@127.0.0.1:9000"

    scrubbed = redact_credentials(
        {"error": "HTTP HEAD to 'http://deposium-dev:deposium-dev-secret@127.0.0.1:9000/x'"},
        endpoint,
    )

    assert scrubbed["error"] == "HTTP HEAD to 'http://[REDACTED]:[REDACTED]@127.0.0.1:9000/x'"
    assert "-secret" not in scrubbed["error"]


class TestTheNormalDeploymentShape:
    """`S3_ENDPOINT` with NO userinfo and the credentials supplied separately.

    That is what `docker-compose.yml` sets — `S3_ENDPOINT=http://rustfs:9000` plus
    `S3_ACCESS_KEY`/`S3_SECRET_KEY` — and it is the shape that actually runs. Keying the
    redactor on the endpoint's userinfo made it a complete no-op here: `endpoint_credentials`
    returns `()`, so nothing was scrubbed out of the whole payload, while DuckDB httpfs quotes
    the configured `s3_access_key_id` in its errors and those errors reach the published
    manifests verbatim through every `redact_credentials` caller.
    """

    ENDPOINT = "http://rustfs:9000"
    ACCESS_KEY = "deposium-dev"
    SECRET_KEY = "deposium-dev-secret"

    @pytest.fixture(autouse=True)
    def _configured(self, monkeypatch):
        monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", self.ACCESS_KEY)
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", self.SECRET_KEY)

    def test_the_endpoint_carries_no_credentials_yet_the_deployment_does(self):
        # The premise: this is exactly the shape the old redactor could not see.
        assert endpoint_credentials(self.ENDPOINT) == ()
        assert self.ACCESS_KEY in configured_store_secrets(self.ENDPOINT)
        assert self.SECRET_KEY in configured_store_secrets(self.ENDPOINT)

    def test_an_httpfs_error_naming_the_access_key_is_scrubbed(self):
        payload = {
            "errors": [
                {
                    "zone_id": "montpellier",
                    "error": (
                        "IOException: IO Error: Unable to connect to URL "
                        f"'{self.ENDPOINT}/deposium/climate/z.parquet': 403 (Forbidden) "
                        f"<AWSAccessKeyId>{self.ACCESS_KEY}</AWSAccessKeyId>"
                    ),
                }
            ],
            "run_manifest_errors": [{"error": f"signature for {self.SECRET_KEY} rejected"}],
        }

        scrubbed = redact_credentials(payload, self.ENDPOINT)

        serialized = json.dumps(scrubbed)
        assert self.ACCESS_KEY not in serialized
        assert self.SECRET_KEY not in serialized
        # The address itself is not a secret and must survive, or the manifest loses provenance.
        assert "rustfs:9000" in scrubbed["errors"][0]["error"]

    def test_the_access_key_prefixes_the_secret_and_the_tail_does_not_survive(self):
        # `deposium-dev` is a PREFIX of `deposium-dev-secret`. Redacting the short one first
        # rewrites the password to `[REDACTED]-secret`, which every "the whole secret is gone"
        # assertion passes while the tail is published.
        scrubbed = redact_credentials({"error": f"key={self.SECRET_KEY}"}, self.ENDPOINT)

        assert scrubbed["error"] == "key=[REDACTED]"
        assert "-secret" not in scrubbed["error"]

    def test_the_json_escaped_spelling_of_a_configured_key_is_caught(self, monkeypatch):
        """A 403 body quoted back into an exception arrives JSON-escaped, not raw.

        Neither the raw nor the URL-encoded spelling matches `dep\\u00f4sium-dev` — the value as
        it appears BETWEEN the quotes of a JSON string.
        """
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", 'dep\u00f4sium/"dev"')
        embedded = json.dumps({"AWSSecretAccessKey": 'dep\u00f4sium/"dev"'})

        scrubbed = redact_credentials({"error": f"body={embedded}"}, self.ENDPOINT)

        assert "sium" not in scrubbed["error"]
        assert REDACTED in scrubbed["error"]

    def test_the_percent_encoded_spelling_of_a_configured_key_is_caught(self, monkeypatch):
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", "s3cret/with+chars")

        scrubbed = redact_credentials(
            {"error": "tried s3cret/with+chars then s3cret%2Fwith%2Bchars"}, self.ENDPOINT
        )

        assert "s3cret" not in scrubbed["error"]


class TestThePrivateCatalogueCredential:
    """`STAC_URL` is a SECOND credential, and the redactors are the publication boundary.

    `configured_store_secrets` answers "what does this process reach the object store with", and
    a private STAC catalogue is not the object store. Measured on the pinned pystac_client, a
    URL-parse failure echoes the configured `STAC_URL` in full:

        APIError: Failed to parse: https://stac-reader:stac-s3cret@stac.internal:notaport/v1

    That string is `f"{type(exc).__name__}: {exc}"`, it lands in `errors[].error`, and the panels
    publish that field into the destination bucket. Sanitizing `inputs[].endpoint` is a different
    field and never touched it.
    """

    ENDPOINT = "http://rustfs:9000"
    STAC_URL = "https://stac-reader:stac-s3cret@stac.internal:notaport/v1"
    FAILURE = f"APIError: Failed to parse: {STAC_URL}"

    @pytest.fixture(autouse=True)
    def _configured(self, monkeypatch):
        monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", "deposium-dev")
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", "deposium-dev-secret")
        monkeypatch.setattr(s3_endpoint.settings, "STAC_URL", self.STAC_URL)

    def test_the_store_answer_does_not_know_the_catalogue_token(self):
        # The premise: the two credentials are unrelated, and the store answer is the narrow one.
        assert "stac-s3cret" not in configured_store_secrets(self.ENDPOINT)
        assert "stac-s3cret" in publication_secrets(self.ENDPOINT)
        # Every store secret is still in there — this widens the set, it does not replace it.
        assert set(configured_store_secrets(self.ENDPOINT)) <= set(
            publication_secrets(self.ENDPOINT)
        )

    def test_a_catalogue_failure_in_a_diagnostic_field_is_scrubbed(self):
        scrubbed = redact_credentials({"errors": [{"error": self.FAILURE}]}, self.ENDPOINT)

        serialized = json.dumps(scrubbed)
        assert "stac-s3cret" not in serialized
        assert "stac-reader" not in serialized
        # The catalogue host is provenance and survives.
        assert "stac.internal" in serialized

    def test_the_bare_string_entry_point_covers_it_too(self):
        assert "stac-s3cret" not in redact_text(self.FAILURE, self.ENDPOINT)

    def test_a_structural_field_is_still_left_alone(self):
        # Scoping is unchanged: only DIAGNOSTIC_FIELDS are rewritten, whatever the secret set is.
        payload = {"run_id": "stac-reader-2026-08-21"}
        assert redact_credentials(payload, self.ENDPOINT) == payload


def test_a_catalogue_without_a_credential_adds_no_pattern(monkeypatch):
    """The default deployment. `STAC_URL` is public, so it must contribute nothing at all."""
    monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", "")
    monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", "")
    monkeypatch.setattr(
        s3_endpoint.settings, "STAC_URL", "https://earth-search.aws.element84.com/v1"
    )

    assert publication_secrets("http://rustfs:9000") == ()


def test_unset_credentials_never_become_a_redaction_pattern(monkeypatch):
    """`""` is a substring of every position in every string; admitting one blanks the document."""
    monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", "")
    monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", "   ")
    payload = {"error": "IO Error: HTTP HEAD to 'http://mesh.internal:9000/d/z1.parquet'"}

    assert configured_store_secrets("http://mesh.internal:9000") == ()
    assert redact_credentials(payload, "http://mesh.internal:9000") == payload
    assert redact_credentials(payload, "") == payload


def test_the_archive_cli_and_the_manifest_redactor_share_one_secret_set(monkeypatch):
    """Two implementations of "what is configured" drift; the fifth leak was that drift."""
    from src.cli import forecast_ensemble_archive as archive

    monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", "deposium-dev")
    monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", "deposium-dev-secret")
    monkeypatch.setattr(s3_endpoint.settings, "S3_ENDPOINT", "http://rustfs:9000")

    store_secrets = set(configured_store_secrets(s3_endpoint.settings.S3_ENDPOINT))

    assert store_secrets <= archive._secret_values(None)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# `sanitize_stac_url`: a POLICY for a catalogue root, not a second endpoint parser
# --------------------------------------------------------------------------- #


class TestStacPolicy:
    """The STAC contract, component by component. See `sanitize_stac_url`'s table."""

    def test_a_query_is_dropped_and_SAID_so(self):
        """The defect this class exists for.

        `sanitize_endpoint("https://stac.example/v1?token=abc")` used to answer
        `("https://stac.example/v1", True)`: the query silently gone, one boolean raised, and
        nothing anywhere distinguishing "a token was dropped" from "the scheme was lowercased".
        A STAC deployment with a meaningful query was misrecorded in the manifest in silence.
        """
        assert sanitize_stac_url("https://stac.example/v1?token=abc") == (
            "https://stac.example/v1",
            ("query",),
        )

    def test_a_meaningful_query_is_dropped_too_and_the_cost_is_visible(self):
        """`?collections=` is real provenance, and losing it is a real cost. It is paid because
        `?token=` is how a catalogue takes a credential on the URL — and it is paid OUT LOUD."""
        assert sanitize_stac_url("https://stac.example:8443/v1?collections=s2") == (
            "https://stac.example:8443/v1",
            ("query",),
        )

    def test_the_path_is_kept_because_it_is_the_catalogue_root(self):
        """The endpoint grammar refuses a path outright, which published `null` for the DEFAULT
        deployment — the provenance the field exists to record, gone."""
        assert sanitize_endpoint("https://earth-search.aws.element84.com/v1") == (None, True)
        assert sanitize_stac_url("https://earth-search.aws.element84.com/v1") == (
            "https://earth-search.aws.element84.com/v1",
            (),
        )

    def test_userinfo_is_stripped_and_named(self):
        assert sanitize_stac_url("https://stac-reader:stac-s3cret@stac.internal/v1") == (
            "https://stac.internal/v1",
            ("userinfo",),
        )

    def test_a_fragment_is_dropped_and_named(self):
        assert sanitize_stac_url("https://stac.example/v1#frag") == (
            "https://stac.example/v1",
            ("fragment",),
        )

    def test_every_reduction_at_once_is_reported_in_URL_order(self):
        address, reductions = sanitize_stac_url("https://u:t@stac.example/v1/a@b?q=1#f")
        assert address == "https://stac.example"
        assert reductions == ("userinfo", "path", "query", "fragment")

    def test_a_path_with_an_at_sign_is_dropped_rather_than_interpreted(self):
        """This declines to decide whether that `@` is an email address inside a catalogue name
        or a credential spelled after the authority. The host survives either way."""
        assert sanitize_stac_url("https://stac.example/catalogs/user@example.com/v1") == (
            "https://stac.example",
            ("path",),
        )

    def test_an_untouched_url_reports_no_reductions(self):
        """`()` is the load-bearing answer: it is how a reader knows the recorded address IS the
        configured value, rather than a prefix of one."""
        assert sanitize_stac_url("https://stac.internal:8443/v1/catalog") == (
            "https://stac.internal:8443/v1/catalog",
            (),
        )

    def test_unset_and_refused_stay_distinguishable(self):
        assert sanitize_stac_url("") == (None, ())
        assert sanitize_stac_url(None) == (None, ())
        assert sanitize_stac_url("   ") == (None, ())
        # A scheme is mandatory on a catalogue URL, unlike on an endpoint.
        assert sanitize_stac_url("stac.example/v1") == (None, ("refused",))
        assert sanitize_stac_url("ftp://stac.example/v1") == (None, ("refused",))

    def test_it_inherits_every_authority_refusal_from_the_endpoint_grammar(self):
        """Above the path the two contracts are the SAME parser, so a catalogue URL cannot buy a
        looser host or port check by being a catalogue URL."""
        for url in (
            "https://stac.example:99999/v1",
            "https://stac.example:abc/v1",
            "https://[::1:9000/v1",
            "https://stac..example/v1",
            # The password with an unencoded `/` in it: the authority reads `user:sup3r`, whose
            # "port" is not numeric, so no reading of this value is offered at all.
            "https://user:sup3r/s3cret@stac.example:9000/v1",
        ):
            assert sanitize_stac_url(url) == (None, ("refused",)), url

    def test_no_reduction_name_can_be_confused_with_a_refusal(self):
        assert STAC_REFUSED not in STAC_REDUCTIONS


# `sanitize_stac_url`: the same grammar above the authority, a path below it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # The real setting. The path IS the provenance: it names the catalogue root.
        (
            "https://earth-search.aws.element84.com/v1",
            "https://earth-search.aws.element84.com/v1",
        ),
        ("http://stac.internal:8080/v1/collections", "http://stac.internal:8080/v1/collections"),
        # A private deployment reached with basic auth. The userinfo is delimited strictly --
        # the last "@" BEFORE the first "/?#", with the remainder required to be an authority --
        # so it can be dropped without guessing, and the address is still logged.
        ("https://stac-reader:sup3r-s3cret@stac.internal/v1", "https://stac.internal/v1"),
        ("https://stac-reader:sup3r-s3cret%40stac.internal/v1", "https://stac.internal/v1"),
        # A query and a fragment are dropped; the host and path survive.
        ("https://stac.example/v1?contact=user@example.com", "https://stac.example/v1"),
        ("https://stac.example/v1#user@example.com", "https://stac.example/v1"),
        # A path carrying an "@" is dropped rather than reasoned about. The host is the
        # provenance that matters, and this declines to decide whether that "@" is an address
        # inside a catalogue name or a credential someone spelled after the authority.
        ("https://stac.example/catalogs/user@example.com/v1", "https://stac.example"),
        ("https://stac.example/catalogs/user%40example.com/v1", "https://stac.example"),
    ],
)
def test_a_logged_url_keeps_its_path_and_loses_its_credentials(url, expected):
    assert sanitize_stac_url(url)[0] == expected


@pytest.mark.parametrize(
    "url",
    [
        # The ambiguity the old hand-written "does the authority hold a colon?" heuristic was
        # written for. The grammar answers it instead: the authority reads `stac-reader:sup3r`,
        # whose port is not numeric, so no reading of this value is offered at all.
        f"https://{_ACCOUNT}:{_SECRET_HEAD}/{_SECRET_TAIL}@stac.internal/v1",
        f"https://{_ACCOUNT}:{_SECRET_HEAD}?{_SECRET_TAIL}@stac.internal/v1",
        # Everything `sanitize_endpoint` refuses above the path, refused here too.
        "ftp://stac.internal/v1",
        "https://stac.internal:99999/v1",
        "https://stac.internal:abc/v1",
        "https://[::1:9000/v1",
        "https:///v1",
        # A scheme is mandatory: a catalogue URL is always a full URL, and a bare authority or a
        # protocol-relative value is not one.
        "//stac.example/a/user@example.com",
        "stac.example/v1",
    ],
)
def test_a_logged_url_outside_the_grammar_is_refused(url):
    assert sanitize_stac_url(url) == (None, (STAC_REFUSED,))


class TestACredentialBelowTheAuthority:
    """A credential does not only live in a userinfo, and neither half of that was covered.

    Two independent leaks, both measured on 6fbd599 before these tests existed:

    * `sanitize_stac_url("https://user:1234/5678@stac.example/v1")` answered
      `("https://user:1234", ("path",))`. The authority ends at the first `/`, so the intended
      userinfo `user:1234/5678` was cut in two, `user:1234` parsed as host+port, and the
      published address was the username plus the head of the password -- under a reduction
      list that never says the word userinfo;
    * `publication_secrets("")` answered `()` for
      `STAC_URL="https://stac.example/v1?token=abc123secret"`, so the token never entered the
      redaction set and `errors[].error` / `run_manifest_errors[].error` republished it in full.
      `sanitize_stac_url` correctly drops the query from the PUBLISHED address, which is what
      made this easy to miss: the structural field was clean the whole time.
    """

    NUMERIC_PASSWORD_URL = "https://stac-reader:1234/5678@stac.example/v1"
    QUERY_TOKEN = "abc123secret"
    QUERY_URL = f"https://stac.example/v1?collections=sentinel-2&token={QUERY_TOKEN}"

    def test_a_port_plus_an_at_sign_below_it_is_refused_rather_than_published(self):
        """`host:port` and `user:password` are the same two-token shape, and the `@` below the
        authority says which one was written. Measured before the guard: this published
        `https://stac-reader:1234` under a `path`-only reduction."""
        assert sanitize_stac_url(self.NUMERIC_PASSWORD_URL) == (None, (STAC_REFUSED,))

    def test_the_password_head_is_not_published_under_any_spelling(self):
        """The postcondition, stated over the value rather than over one expected string: no
        prefix of the credential survives into whatever is published."""
        address, _reductions = sanitize_stac_url(self.NUMERIC_PASSWORD_URL)
        assert address is None or ("stac-reader" not in address and "1234" not in address)

    def test_an_email_in_a_catalogue_path_still_keeps_its_host(self):
        """The bounded case the path-drop was written for, and it must not become collateral: a
        bare hostname cannot spell `user:password`, so there is no ambiguity to fail closed on."""
        assert sanitize_stac_url("https://stac.example/catalogs/user@example.com/v1") == (
            "https://stac.example",
            ("path",),
        )

    def test_a_query_token_enters_the_publication_secret_set(self, monkeypatch):
        monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", "")
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", "")
        monkeypatch.setattr(s3_endpoint.settings, "STAC_URL", self.QUERY_URL)
        assert self.QUERY_TOKEN in publication_secrets("")

    def test_a_query_token_is_punched_out_of_both_prose_fields(self, monkeypatch):
        """The fields that leak. Every panel writer copies exception text verbatim into these
        two, and `import-zip` then republishes them into the DESTINATION bucket."""
        monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", "")
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", "")
        monkeypatch.setattr(s3_endpoint.settings, "STAC_URL", self.QUERY_URL)
        quoted = f"APIError: Failed to open {self.QUERY_URL}"
        published = redact_credentials(
            {
                "errors": [{"error": quoted}],
                "run_manifest_errors": [{"uri": "s3://b/m.json", "error": quoted}],
            },
            "",
        )
        assert self.QUERY_TOKEN not in json.dumps(published)
        assert REDACTED in published["errors"][0]["error"]
        assert REDACTED in published["run_manifest_errors"][0]["error"]

    def test_the_bare_string_entry_point_covers_it_too(self, monkeypatch):
        """`redact_text` is what `climate_manifest_consumer.main` prints to stderr."""
        monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", "")
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", "")
        monkeypatch.setattr(s3_endpoint.settings, "STAC_URL", self.QUERY_URL)
        assert self.QUERY_TOKEN not in redact_text(f"APIError: {self.QUERY_URL}", "")

    def test_an_incoming_origin_token_is_collected_from_the_document(self):
        """The import side. The origin's catalogue token is not one of THIS process's configured
        secrets, but the manifest names its own catalogue, so it is recoverable from the document
        being imported -- which is what `_incoming_addresses` hands to `redact_credentials`."""
        assert "origin-tok" in url_secrets("https://stac.origin/v1?api_key=origin-tok")

    def test_a_fragment_credential_is_collected_too(self):
        assert "zzz" in url_secrets("https://stac.example/v1#api_key=zzz")

    @pytest.mark.parametrize(
        "name",
        [
            # `_` percent-encoded: the server reads `api_key`, the raw marker check read `api%5Fkey`.
            "api%5Fkey",
            # A letter encoded, so the raw name matches no marker at all.
            "%74oken",
            # Fully encoded, the spelling a strict encoder emits.
            "%73%65%63%72%65%74",
        ],
    )
    def test_a_percent_encoded_parameter_name_still_names_a_credential(self, name):
        """Measured before the guard: `url_secrets("...?api%5Fkey=abc12345")` answered `()`.

        The name is compared against the markers as the SERVER will read it, decoded. The raw
        spelling contains no marker, so the token never entered the secret set and was republished
        by every field that echoes the configured URL.
        """
        assert "abc12345" in url_secrets(f"https://stac.example/v1?{name}=abc12345")
        assert "abc12345" in url_secrets(f"https://stac.example/v1#{name}=abc12345")

    def test_a_parameter_that_names_no_credential_never_becomes_a_pattern(self):
        """The other failure mode. `?limit=1` in the secret set blanks every `1` in the
        document -- the destructive redactor `redact_credentials` was already scoped down once
        to stop being. Names, not every value."""
        assert url_secrets("https://stac.example/v1?collections=sentinel-2&limit=1") == ()

    def test_the_userinfo_spelling_is_still_collected(self):
        """The widening must not have replaced the case it started from."""
        assert "sup3r-s3cret" in url_secrets("https://reader:sup3r-s3cret@stac.internal/v1")

    def test_the_longest_secret_comes_first(self):
        """The ordering trap, on this function's own output: an account name routinely prefixes
        its own password, and replacing the short one first publishes the tail of the long one
        wearing a redaction."""
        secrets = url_secrets("https://deposium-dev:deposium-dev-secret@far.away:9000/v1")
        assert secrets == tuple(sorted(secrets, key=len, reverse=True))


def test_the_two_nulls_are_distinguishable():
    """`None` alone cannot say whether an endpoint was configured; the flag can."""
    assert sanitize_endpoint("") == (None, False)  # nothing configured
    assert sanitize_endpoint(f"{_ACCOUNT}:{_SECRET_HEAD}/{_SECRET_TAIL}@host:9000") == (
        None,
        True,  # configured, and unparseable — a misconfiguration, not an absence
    )


# --------------------------------------------------------------------------- #
# GUARD 1: the sanitizer never returns a truthy non-address
# --------------------------------------------------------------------------- #

# Every one of these is a NON-EMPTY `S3_ENDPOINT` that used to sanitize to the truthy scheme-only
# string `'http://'` (or `'//'`). Truthy means it passed `if address:` at every call site; the
# consumer then stripped the scheme and ran `SET s3_endpoint = ''`, which is DuckDB's spelling of
# "use s3.us-east-1.amazonaws.com" — with `S3_ACCESS_KEY`/`S3_SECRET_KEY` signed into the next
# request. A scheme is not an address, and neither is a bare port.
_SCHEME_ONLY_ENDPOINTS = [
    "http://",
    "https://",
    "//",
    "http:///",
    "http:////",
    "http://@",
    "http://user@",
    "http://user:pass@",
    "http://%40",
    "http://user:pass%40",
    "http://?@",
    "http://?",
    "http://#",
    "http:// ",
    "http://\t",
    "http:///@",
    "http://:@",
    "http://@@",
    "//@",
    # Added here rather than reported: a colon alone, and a port with no host in front of it.
    "http://:",
    "http://:/",
    "http://:9000",
    "//:9000",
    "://",
    "http://   ",
    "http://%40%40",
    "/",
    "///",
]


@pytest.mark.parametrize("endpoint", _SCHEME_ONLY_ENDPOINTS)
def test_a_value_with_no_host_in_it_is_refused_not_returned_as_an_address(endpoint):
    """The postcondition: the first element is None, or it contains a host. Never in between.

    This is guard 1 on its own — `src.utils.duckdb_conn` is not imported here, so nothing
    downstream can be covering for it.
    """
    assert sanitize_endpoint(endpoint) == (None, True)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://rustfs:9000",
        "https://s3.example.com",
        "mesh.internal:9000",
        "//mesh.internal:9000",
        "http://[::1]:9000",
        "http://127.0.0.1:9000",
    ],
)
def test_the_postcondition_holds_from_the_other_side_too(endpoint):
    # Refusing everything would satisfy "never a truthy non-address" and break every deployment.
    address, _removed = sanitize_endpoint(endpoint)

    assert address is not None
    host = address.split("://")[-1].lstrip("/").split("/")[0]
    host = host[1:].split("]")[0] if host.startswith("[") else host.split(":")[0]
    assert host


# --------------------------------------------------------------------------- #
# The postcondition over a generated space, not a hand-written list
# --------------------------------------------------------------------------- #

_PREFIXES = ["", "http://", "https://", "HTTP://", "ftp://", "//", "///", "/", "://", "s3://"]
_USERINFOS = [
    "",
    f"{_ACCOUNT}:{_SECRET}@",
    f"{_ACCOUNT}:{_SECRET}%40",
    f"{_ACCOUNT}@",
    # No delimiter at all: the password reads as a port.
    f"{_ACCOUNT}:{_SECRET}",
    # The delimiter outside the authority, which is where the password prefix used to survive.
    f"{_ACCOUNT}:{_SECRET_HEAD}/{_SECRET_TAIL}@",
    f"a:b@{_ACCOUNT}:{_SECRET}@",
]
_HOSTS = ["mesh.internal", "127.0.0.1", "[::1]", "[::1", "::1]", "", "me sh", "-h", "h-", "h..i"]
_PORTS = ["", ":9000", ":99999", ":abc", ":0", ":09000", ":٩٠٠٠", ":", ":65535"]
_TAILS = ["", "/", "/catalog", f"?x={_SECRET}", f"#{_SECRET}", f"/{_ACCOUNT}:{_SECRET}@b"]

# What `sanitize_endpoint` is allowed to return, re-derived here rather than imported: an
# assertion that reuses the grammar it is checking passes for any pair of matching mistakes.
_ADDRESS = re.compile(r"(https?://|//)?(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9.-]+)(:[1-9][0-9]{0,4})?")


def test_the_postcondition_holds_across_the_whole_generated_space():
    """37 800 shapes, every combination of the ways this value has ever been malformed.

    Four hand-written hostile tables preceded this one and each was incomplete — the reported
    list of the fourth round contained eight shapes none of the first three had thought of. A
    product is not a proof, but it is not a list of the mistakes already made either.

    Three claims per shape, for both parsers:

    * a returned address contains no secret and no PREFIX of one (a password split across a "/"
      published its first half, which every whole-secret assertion passes);
    * a returned `sanitize_endpoint` address matches the grammar — never a scheme alone, a bare
      port, a path, or anything else `if address:` would read as configured;
    * `None` for a non-empty value always carries `True`, so "unset" and "unusable" stay apart.
    """
    violations = []
    for prefix, userinfo, host, port, tail in itertools.product(
        _PREFIXES, _USERINFOS, _HOSTS, _PORTS, _TAILS
    ):
        value = f"{prefix}{userinfo}{host}{port}{tail}"
        for parse in (sanitize_endpoint, sanitize_stac_url):
            # The second element differs by parser -- a bool for one, a tuple of reduction names
            # for the other -- and only its TRUTH is read here: "the returned address is not the
            # configured value". `sanitize_stac_url` refuses with `("refused",)` and reports
            # "unset" with `()`, so the two nulls stay apart under exactly this test.
            address, differed = parse(value)
            if address is None:
                if not differed and value.strip():
                    violations.append(("configured value reported as unset", parse, value))
                continue
            leaked = next(
                (
                    secret[:length]
                    for secret in (_SECRET, _ACCOUNT)
                    for length in range(4, len(secret) + 1)
                    if secret[:length] in address
                ),
                None,
            )
            if leaked:
                violations.append((f"leaked {leaked!r}", parse, value, address))
            if parse is sanitize_endpoint and not _ADDRESS.fullmatch(address):
                violations.append(("not an address", parse, value, address))

    assert not violations, violations[:10]


# --------------------------------------------------------------------------- #
# The module docstring's justification for string parsing, measured rather than asserted
# --------------------------------------------------------------------------- #


def test_urlsplit_would_drop_the_host_of_a_scheme_less_endpoint():
    """Why this module parses strings instead of calling `urlsplit`.

    The grammar admits a scheme-less `host:port` because DuckDB's own `SET s3_endpoint` takes
    exactly that. `urlsplit` reads it as a scheme with an empty netloc, so the host disappears.
    """
    split = urlsplit("rustfs:9000")

    assert (split.scheme, split.netloc, split.path) == ("rustfs", "", "9000")


def test_boto3_refuses_the_scheme_less_spelling_that_duckdb_requires():
    """The docstring used to justify string parsing with "boto3 accepts a bare `host:9000`".

    It does not. Measured against the pinned botocore, `endpoint_url='rustfs:9000'` raises at
    client construction. The scheme-less spelling exists for DuckDB, not for boto3, and a
    measured-false justification has no business sitting in a security module — the next reader
    would build on it.
    """
    boto3 = pytest.importorskip("boto3")

    with pytest.raises(ValueError, match="Invalid endpoint"):
        boto3.client(
            "s3",
            endpoint_url="rustfs:9000",
            aws_access_key_id="a",
            aws_secret_access_key="b",
            region_name="us-east-1",
        )


class TestAShortSecretIsNotAPattern:
    """A redactor that mangles every diagnostic destroys the field it exists to keep readable.

    The filter used to be "non-empty", so a one-character configured value was a substring of
    ordinary prose everywhere it occurred. Both failures below were measured before the guard.
    """

    def _redacted(self, monkeypatch, token, prose):
        monkeypatch.setattr(s3_endpoint.settings, "STAC_URL", f"https://s.e/v1?token={token}")
        monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", "")
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", "")
        payload = s3_endpoint.redact_credentials(
            {"errors": [{"error": prose}]}, "http://rustfs:9000"
        )
        return payload["errors"][0]["error"]

    def test_a_one_character_token_does_not_blank_every_digit(self, monkeypatch):
        prose = "zone 1 of 15 failed after 1 retry"
        assert self._redacted(monkeypatch, "1", prose) == prose

    def test_a_two_character_token_does_not_shred_ordinary_words(self, monkeypatch):
        prose = "IOException: about to abort"
        assert self._redacted(monkeypatch, "ab", prose) == prose

    def test_a_credential_of_real_length_is_still_scrubbed(self, monkeypatch):
        """The guard must not be an excuse: everything a deployment can actually hold is covered.

        `deposium-dev` is twelve characters and an AWS access key id is twenty, both well above
        the floor.
        """
        out = self._redacted(monkeypatch, "abc123secret", "HTTP 401 for token abc123secret at stac")
        assert "abc123secret" not in out
        assert "[REDACTED]" in out

    def test_the_floor_is_stated_rather_than_implied(self):
        assert s3_endpoint.MIN_SECRET_LENGTH == 8


class TestTheFloorIsForUrlDerivedCandidatesOnly:
    """A configured store key is the credential whatever its length; only a URL token is noise.

    The floor above was applied to EVERY candidate, the explicitly configured
    `S3_ACCESS_KEY`/`S3_SECRET_KEY` included, and the archive CLI had just been rewired onto
    `configured_store_secrets`. Measured before the guard: with `S3_SECRET_KEY='s3cr3t7'` the set
    came back empty, and `_safe_error_message` republished the key that `main` had been redacting
    under its "non-empty" filter.
    """

    ACCESS_KEY = "acc-key"  # seven characters, one below `MIN_SECRET_LENGTH`
    SECRET_KEY = "s3cr3t7"  # seven characters, one below `MIN_SECRET_LENGTH`

    def _configure(self, monkeypatch, *, access_key="", secret_key="", stac_url=""):
        monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", access_key)
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", secret_key)
        monkeypatch.setattr(s3_endpoint.settings, "STAC_URL", stac_url)

    def test_the_two_keys_are_below_the_floor_by_construction(self):
        assert len(self.ACCESS_KEY) == len(self.SECRET_KEY) == s3_endpoint.MIN_SECRET_LENGTH - 1

    def test_a_short_configured_secret_key_is_still_a_pattern(self, monkeypatch):
        self._configure(monkeypatch, secret_key=self.SECRET_KEY)
        assert self.SECRET_KEY in configured_store_secrets("http://rustfs:9000")

    def test_a_short_configured_access_key_is_still_a_pattern(self, monkeypatch):
        self._configure(monkeypatch, access_key=self.ACCESS_KEY)
        assert self.ACCESS_KEY in configured_store_secrets("http://rustfs:9000")

    def test_a_short_configured_key_is_punched_out_of_published_prose(self, monkeypatch):
        self._configure(monkeypatch, access_key=self.ACCESS_KEY, secret_key=self.SECRET_KEY)
        payload = {"errors": [{"error": f"403 for {self.ACCESS_KEY} with {self.SECRET_KEY}"}]}
        published = redact_credentials(payload, "http://rustfs:9000")
        assert published["errors"][0]["error"] == f"403 for {REDACTED} with {REDACTED}"

    def test_a_url_token_of_the_same_length_is_still_below_the_floor(self, monkeypatch):
        """The exemption is for the configured keys and for them only: the prose-mangling the
        floor exists to stop is a URL-side failure, and it must keep stopping it."""
        self._configure(monkeypatch, stac_url=f"https://s.e/v1?token={self.SECRET_KEY}")
        assert publication_secrets("http://rustfs:9000") == ()

    def test_a_userinfo_password_of_the_same_length_is_still_below_the_floor(self, monkeypatch):
        self._configure(monkeypatch)
        assert configured_store_secrets(f"http://reader:{self.SECRET_KEY}@rustfs:9000") == ()

    def test_one_more_character_and_the_url_token_is_admitted(self, monkeypatch):
        """The boundary, so the test above cannot pass by the URL path being broken outright."""
        token = self.SECRET_KEY + "x"
        self._configure(monkeypatch, stac_url=f"https://s.e/v1?token={token}")
        assert token in publication_secrets("http://rustfs:9000")


class TestAmbientAwsCredentialsAreRedacted:
    """#147 — the AWS-default deployment shape: ``S3_*`` unset, ``AWS_*`` from an instance profile.

    DuckDB httpfs falls back to the ambient environment and quotes the access key straight back in
    its errors, which the panels copy verbatim into a published run manifest. The secret set knew
    nothing about those variables, so ``redact_credentials`` returned the payload untouched.
    """

    KEY = "AKIAAMBIENT12345"
    SECRET = "ambient-secret-abcdefgh"
    ENDPOINT = "http://rustfs:9000"

    @pytest.fixture(autouse=True)
    def _ambient_only(self, monkeypatch):
        # The premise of the whole issue: the store settings are UNSET and the credentials arrive
        # through the environment instead. With these set, the pre-existing paths would cover it.
        monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", "")
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", "")
        # Clear ALL FOUR first. A host that already holds assumed-role credentials — AWS-backed CI,
        # or a developer shell after `aws sso login` — exports AWS_SESSION_TOKEN alongside the key,
        # and an inherited token would join the secret set and make this class non-deterministic
        # (and the negative test below unfalsifiable).
        for name in s3_endpoint.AMBIENT_STORE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", self.KEY)
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", self.SECRET)

    def test_the_ambient_environment_reaches_both_secret_sets(self):
        # Both, because redact_credentials consumes publication_secrets while the archive CLI
        # consumes configured_store_secrets — the two used to duplicate their source list.
        for secrets in (
            configured_store_secrets(self.ENDPOINT),
            publication_secrets(self.ENDPOINT),
        ):
            assert self.KEY in secrets
            assert self.SECRET in secrets

    def test_an_httpfs_error_naming_the_ambient_key_is_scrubbed(self):
        payload = {
            "run_manifest_errors": [
                {
                    "zone_id": "montpellier",
                    "error": f'InvalidAccessKeyId ... Invalid Access Key: "{self.KEY}"',
                }
            ]
        }
        out = redact_credentials(payload, self.ENDPOINT)
        assert self.KEY not in json.dumps(out)
        # The diagnostic scoping must survive: a sibling identifier is not a credential.
        assert out["run_manifest_errors"][0]["zone_id"] == "montpellier"

    def test_the_session_token_is_covered_too(self, monkeypatch):
        token = "FwoGZXIvYXdzEBYaDAMBIENTSESSIONTOKEN"
        monkeypatch.setenv("AWS_SESSION_TOKEN", token)
        out = redact_credentials({"error": f"denied with token {token}"}, self.ENDPOINT)
        assert token not in json.dumps(out)

    def test_the_percent_encoded_spelling_is_caught(self, monkeypatch):
        # The raw reads this replaced had no spelling table, so an encoded form escaped them.
        secret = "ambient/secret+with=specials"
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
        encoded = quote(secret, safe="")
        assert encoded != secret, "the fixture must actually exercise re-encoding"
        out = redact_credentials({"error": f"url s3://k:{encoded}@host/b"}, self.ENDPOINT)
        assert encoded not in json.dumps(out)

    def test_the_lowercase_percent_spelling_is_caught_too(self, monkeypatch):
        # `quote` only emits uppercase hex, and the replacement is case-sensitive — so a producer
        # emitting `%2f` rather than `%2F` used to sail through an otherwise-covered secret.
        secret = "ambient/secret+specials"
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
        lowered = quote(secret, safe="").lower()
        assert "%2f" in lowered, "the fixture must actually exercise lowercase escapes"
        out = redact_credentials({"error": f"url s3://k:{lowered}@host/b"}, self.ENDPOINT)
        assert lowered not in json.dumps(out)

    def test_a_secret_whose_own_characters_spell_an_escape_folds_the_other_way(self, monkeypatch):
        # The mirror case: the credential itself contains the text "%2f", and a producer that
        # normalises escapes to uppercase — which RFC 3986 6.2.2.1 recommends — echoes it as
        # "%2F". Folding only towards lowercase left that spelling publishable.
        secret = "ambient%2fsecret%2bvalue"
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
        normalised = secret.replace("%2f", "%2F").replace("%2b", "%2B")
        assert normalised != secret, "the fixture must actually exercise the other direction"
        out = redact_credentials({"error": f"denied for {normalised}"}, self.ENDPOINT)
        assert normalised not in json.dumps(out)

    def test_no_ambient_credentials_means_no_pattern(self, monkeypatch):
        # An empty variable must never become a pattern: "" is a substring of every position and
        # would blank the whole document.
        for name in s3_endpoint.AMBIENT_STORE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        assert publication_secrets(self.ENDPOINT) == ()
        payload = {"error": "nothing to scrub here"}
        assert redact_credentials(payload, self.ENDPOINT) == payload

    def test_the_archive_cli_still_sees_them_through_the_shared_definition(self):
        from src.cli.forecast_ensemble_archive import _secret_values

        values = _secret_values(None)
        assert self.KEY in values
        assert self.SECRET in values


class TestEffectiveAwsCredentialsAreRedacted:
    """#155 — values already selected by botocore, without resolving a provider here."""

    @pytest.fixture(autouse=True)
    def _effective_only(self, monkeypatch):
        monkeypatch.setattr(s3_endpoint.settings, "S3_ACCESS_KEY", "")
        monkeypatch.setattr(s3_endpoint.settings, "S3_SECRET_KEY", "")
        for name in s3_endpoint.AMBIENT_STORE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        with s3_endpoint._effective_store_values_lock:
            s3_endpoint._effective_store_values.clear()
        yield
        with s3_endpoint._effective_store_values_lock:
            s3_endpoint._effective_store_values.clear()

    def test_an_observed_generation_reaches_both_shared_secret_sets(self):
        access_key = "AKIAPROVIDER12345678"
        secret_key = "provider/secret+with=specials"
        token = "provider-session-token-abcdefgh"

        remember_effective_store_credentials(access_key, secret_key, token)

        for secrets in (configured_store_secrets(""), publication_secrets("")):
            assert access_key in secrets
            assert secret_key in secrets
            assert quote(secret_key, safe="") in secrets
            assert token in secrets

        embedded = json.dumps(secret_key)[1:-1]
        payload = {"error": f"key={access_key} secret={embedded} token={token}"}
        assert (
            json.dumps(redact_credentials(payload, ""))
            == '{"error": "key=[REDACTED] secret=[REDACTED] token=[REDACTED]"}'
        )

    def test_every_observed_generation_remains_available_for_delayed_errors(self):
        generations = [(f"AKIAGENERATION{i:08d}", f"generation-secret-{i:08d}") for i in range(4)]
        for generation in generations:
            remember_effective_store_credentials(*generation)

        values = set(effective_store_values())
        assert values == {value for pair in generations for value in pair}
        delayed = {"error": f"delayed failure for {generations[0][0]} after rotation 4"}
        assert generations[0][0] not in json.dumps(redact_credentials(delayed, ""))

    def test_reobserving_a_generation_is_deduplicated(self):
        generation = ("key-abcdefgh", "secret-abcdefgh", "token-abcdefgh")
        remember_effective_store_credentials(*generation)
        remember_effective_store_credentials(*generation)

        assert set(effective_store_values()) == set(generation)


class TestForecastApiKeysAreRedacted:
    """#160 — the Open-Meteo commercial ``apikey``, a third credential family.

    It does not travel in a userinfo like the store credentials or ``STAC_URL``; it travels in a
    query string. ``httpx.HTTPStatusError.__str__`` embeds the request URL in full,
    ``src/utils/forecast.py`` records that as ``entry["reason"]``, and the entry rides
    ``provider_chain`` out to the MCP client inside ``metadata`` — a client-visible boundary.

    Nothing about the scoping or the walker was wrong: ``reason`` is already a DIAGNOSTIC_FIELDS
    key and the walk already descends the nested list. The value was simply not in the secret set.
    """

    KEY = "om-commercial-key-abcdef"
    ENDPOINT = "http://rustfs:9000"

    @pytest.fixture(autouse=True)
    def _forecast_key_configured(self, monkeypatch):
        monkeypatch.setattr(s3_endpoint.settings, "FORECAST_API_KEY", self.KEY, raising=False)
        for name in ("FORECAST_ARCHIVE_API_KEY", "FORECAST_ENSEMBLE_API_KEY"):
            monkeypatch.setattr(s3_endpoint.settings, name, "", raising=False)
        for name in s3_endpoint.AMBIENT_STORE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)

    def test_the_key_is_in_the_publication_secret_set(self):
        assert self.KEY in s3_endpoint.publication_secrets(self.ENDPOINT)

    def test_the_provider_chain_returned_to_the_client_is_scrubbed(self):
        # The shape src/tools/forecast.py:172 actually returns on ForecastProviderUnavailable.
        payload = {
            "error": "all forecast providers unavailable",
            "metadata": {
                "provider_chain": [
                    {
                        "provider": "open-meteo",
                        "status": "outage",
                        "reason": (
                            "HTTPStatusError: Server error '500' for url "
                            f"'https://customer-api.open-meteo.com/v1/forecast?lat=43&apikey={self.KEY}'"
                        ),
                    }
                ],
                "requested_days": 7,
            },
        }
        out = s3_endpoint.redact_credentials(payload, self.ENDPOINT)
        assert self.KEY not in json.dumps(out)
        # Scoped, not blanket: the surrounding identifiers stay readable or the operator cannot
        # tell which provider failed.
        assert out["metadata"]["provider_chain"][0]["provider"] == "open-meteo"
        assert out["metadata"]["requested_days"] == 7

    def test_the_log_record_for_the_same_failure_is_scrubbed(self):
        # src/utils/forecast.py:229 logs the same exception through structlog.
        out = s3_endpoint.redact_credentials(
            {"event": "forecast_provider_transport", "error": f"...?apikey={self.KEY}"},
            self.ENDPOINT,
        )
        assert self.KEY not in json.dumps(out)

    def test_a_short_configured_key_is_not_dropped_by_the_length_floor(self, monkeypatch):
        # `configured=`, not `derived=`: an explicitly configured credential is a credential
        # whatever its length. The MIN_SECRET_LENGTH floor exists to stop a *derived* fragment
        # from matching everything, and must not silently exempt a real configured key.
        short = "k" * (s3_endpoint.MIN_SECRET_LENGTH - 1)
        monkeypatch.setattr(s3_endpoint.settings, "FORECAST_API_KEY", short, raising=False)
        assert short in s3_endpoint.publication_secrets(self.ENDPOINT)

    def test_the_store_scoped_accessor_stays_store_scoped(self):
        # configured_store_secrets answers "what does this process use to reach the object store".
        # The archive CLI consumes it under exactly that meaning, so widening it would be wrong —
        # #160 belongs to the publication boundary, not to the store definition.
        assert self.KEY not in s3_endpoint.configured_store_secrets(self.ENDPOINT)

    def test_all_three_entitlements_are_covered(self, monkeypatch):
        # Three separate Open-Meteo entitlements with three separate keys; covering only the one
        # the forecast client uses would leave Previous Runs and Ensemble publishable.
        keys = {
            "FORECAST_API_KEY": "fk-aaaaaaaaaaaa",
            "FORECAST_ARCHIVE_API_KEY": "ak-bbbbbbbbbbbb",
            "FORECAST_ENSEMBLE_API_KEY": "ek-cccccccccccc",
        }
        for name, value in keys.items():
            monkeypatch.setattr(s3_endpoint.settings, name, value, raising=False)
        secrets = s3_endpoint.publication_secrets(self.ENDPOINT)
        for value in keys.values():
            assert value in secrets
