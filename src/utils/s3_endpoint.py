"""One place that knows what an ``S3_ENDPOINT`` may be, and refuses everything else.

Two directions live here and they are NOT symmetrical:

* :func:`sanitize_endpoint` decides what may be *published* and what may be *configured*. It is an
  ACCEPT grammar — an allowlist. A value either matches ``[scheme://]host[:port][/]`` exactly or it
  is refused whole. Nothing is recovered or repaired, because every previous round of this function
  tried to recover and every round shipped a new way for a credential to survive the recovery. The
  lone trailing ``/`` is stripped only AFTER the authority in front of it has been accepted, which
  is what keeps it a normalisation and not a recovery; :func:`sanitize_endpoint` documents why that
  one spelling is admitted, along with the grammar and the table of shapes it now refuses.
* :func:`endpoint_credentials` decides what must be *redacted*. Its job is the opposite one —
  find a secret in a value that is already known to be malformed — so it stays deliberately
  permissive and keeps the old hostile-input parser (:func:`_split_authority`). A false positive
  there costs a redundant redaction; a false negative there costs a leak. It never produces a
  value that is published.

The userinfo is not the only place a credential lives, and in the deployment that actually runs it
is not the place at all: ``docker-compose.yml`` sets ``S3_ENDPOINT=http://rustfs:9000`` and passes
``S3_ACCESS_KEY``/``S3_SECRET_KEY`` separately. ``configured_store_secrets`` is therefore the one
answer to "what credentials does this process reach the OBJECT STORE with", covering both, and the
archive CLI in ``src/cli/forecast_ensemble_archive.py`` is built on it rather than on its own copy.
``publication_secrets`` is the wider answer the two redactors below use, because a publication
boundary is not store-only: a private ``STAC_URL`` carries a SECOND credential, ``pystac_client``
quotes that URL in full in the exceptions a panel copies into ``errors[].error``, and the panels
publish that field into the destination bucket.

:func:`sanitize_stac_url` is the third, separate contract, and it is a POLICY rather than a
second spelling of the first one: ``STAC_URL`` legitimately carries a path (``STAC_URL`` is
``https://earth-search.aws.element84.com/v1``), and a query on it may be a credential or may be
provenance. It reuses the same strict authority grammar, so it inherits every refusal above the
path; below the path it decides for itself, and it NAMES what it removed so a manifest can record
that the address it publishes is not the value that was configured. Every production caller uses it
directly, so there is no second, boolean-flavoured wrapper around it to keep true: a public
contract with no caller is a contract nothing measures.

Parsing is string-based rather than ``urlsplit``-based: ``S3_ENDPOINT`` is not always a full URL —
DuckDB's own ``SET s3_endpoint`` takes a bare ``host:port``, which is why the grammar admits a
scheme-less value — and ``urlsplit`` reads such a value as a scheme with no netloc
(``urlsplit("rustfs:9000")`` is ``scheme='rustfs', netloc='', path='9000'``), silently dropping the
host. boto3 does NOT accept that spelling — measured on the pinned botocore 1.43.57,
``boto3.client("s3", endpoint_url="rustfs:9000")`` raises ``ValueError: Invalid endpoint`` at
construction. The normals serving reader explicitly adapts an already-validated scheme-less
authority to HTTPS, preserving DuckDB's TLS default; other SDK callers still require the scheme.
The deployment example (``S3_ENDPOINT=http://rustfs:9000``) spells it out.

The permissive parser below is retained for :func:`endpoint_credentials` alone. Four shapes used
to smuggle the userinfo past a naive split and it still handles them explicitly, because those are
exactly the shapes whose credentials a redactor must be able to name:

* the authority was delimited on ``/`` alone, so ``//user:pass@host`` and ``http:///user:pass@host``
  parsed as an empty authority plus a *path* that was copied through verbatim;
* a query or a fragment stayed glued to the host (``http://u:p@host?debug=u:p``), carrying a second
  copy of the credentials the authority had just had removed;
* ``@`` was the only recognised userinfo delimiter, so ``http://user:pass%40host`` read as a bare
  hostname while still naming the account in front of the real host;
* an unencoded ``/``, ``?`` or ``#`` INSIDE the password ended the authority early, so
  ``http://user:sup3r/s3cret@host:9000`` published ``http://user:sup3r`` — the account name plus a
  PREFIX of the secret, which every "the whole secret is absent" assertion passes.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from threading import Lock
from typing import Any
from urllib.parse import quote, quote_plus, unquote, unquote_plus

from src.config import settings

# ``@`` separates the userinfo from the host. ``%40`` is that same delimiter percent-encoded, and
# it must delimit too: ``http://user:pass%40host:9000`` is what an operator gets from any tool that
# quotes the endpoint, and reading it as a bare hostname would publish the account name.
_USERINFO_DELIMITERS = ("@", "%40")
# An authority ends at the first of these three; whatever follows is path, query or fragment.
_AUTHORITY_END = "/?#"
# A path ends at the first of these two.
_PATH_END = "?#"


def _host_of(host_port: str) -> str:
    """The host in a ``host[:port]``, or ``""`` when the value names no host at all.

    A scheme is not an address and neither is a bare port: ``http://`` and ``http://:9000`` both
    parse to an empty host here, and both used to survive sanitization as a TRUTHY value
    (``"http://"``) that every ``if address:`` in the codebase read as "an endpoint is
    configured". See :func:`sanitize_endpoint` for what that cost.

    An IPv6 literal is bracketed precisely so its own colons are not port separators, so the
    bracket form is split on ``]`` rather than on ``:``.
    """
    if host_port.startswith("["):
        return host_port[1:].partition("]")[0].strip()
    return host_port.partition(":")[0].strip()


def _first_index(value: str, characters: str) -> int:
    """Index of the first character of ``characters`` in ``value``, or ``len(value)``."""
    found = [index for index in (value.find(char) for char in characters) if index >= 0]
    return min(found, default=len(value))


def _split_userinfo(authority: str) -> tuple[str, str]:
    """``(userinfo, host_port)`` split on the LAST userinfo delimiter, literal or encoded.

    Last, not first: ``http://a:b@c:d@host`` is a single authority whose host is ``host``, and
    splitting on the first ``@`` would leave ``c:d@host`` — a copy of a credential — as the host.
    """
    position, width = max((authority.rfind(d), len(d)) for d in _USERINFO_DELIMITERS)
    if position < 0:
        return "", authority
    return authority[:position], authority[position + width :]


def _split_authority(endpoint: str) -> tuple[str, str, str, str, bool]:
    """``(prefix, userinfo, host_port, path, malformed)``. ``prefix`` is ``scheme://`` or ``//``.

    REDACTION ONLY. Nothing this returns is ever published: :func:`sanitize_endpoint` parses
    against the ACCEPT grammar below instead, and the only caller left here is
    :func:`endpoint_credentials`, whose job is to NAME a secret inside a value the accept grammar
    has already refused. That asymmetry is why this half stays permissive — it errs towards
    finding a credential, and finding one it did not need costs a redundant redaction.

    The query and the fragment are not returned at all: an object-store endpoint is an address,
    it has no query and no fragment, and the only thing either can hold on this value is a stray
    copy of what the authority just had removed.

    ``malformed`` is the case where no address can be identified at all. A userinfo delimiter
    that sits OUTSIDE the parsed authority means the authority boundary MAY have been drawn in the
    wrong place: RFC 3986 forbids ``/``, ``?`` and ``#`` inside userinfo, but ``S3_ENDPOINT``
    carries a raw password out of an env var and nothing re-encodes it, so
    ``http://rustfs-admin:sup3r/s3cret@host:9000`` ends its authority at the ``/`` and what looks
    like a ``host:port`` is really ``account:password-prefix``. Nothing in the string distinguishes
    that from a genuine host followed by an ``@`` in its path, so neither reading may be published.
    The whole value is re-read as userinfo instead, which yields no address (the caller returns
    none) while still handing the redactors the credential components they would otherwise miss.

    That refusal is narrowed by ONE condition: the authority must contain a ``:``. The ambiguity
    it defends against is ``account:password-prefix`` masquerading as ``host:port``, and a
    password only exists on the far side of a colon — with no colon in the authority there is no
    secret for the address to be hiding, whatever else the value is. So
    ``https://stac.example/catalogs/user@example.com/v1`` — a real catalogue path with an ``@``
    in it and no userinfo anywhere — parses as itself instead of being refused outright, which is
    what it did before: applied to ``STAC_URL``, that refusal published ``"endpoint": null`` and
    silently lost the provenance the field exists to record. The colon test is deliberately crude
    rather than "is the part after the colon a valid port": an all-digit password prefix
    (``http://user:1234/5678@host``) would pass a port test and publish ``http://user:1234``, and
    the point of this branch is that a plausible-looking reading is not evidence.

    Two consequences are accepted rather than fixed, both narrower than what they replace:

    * an address WITH a port and an ``@`` later (``https://stac.example:8443/a/user@b``) is still
      refused, because there its ``host:port`` is exactly the shape the leak wears;
    * a userinfo that is a username with NO password and an unencoded ``/`` in it
      (``http://accesskeyonly/x@host``) is read as a host and published. No secret can be spelled
      that way — an S3 access key contains no ``/`` and a password needs the colon this branch
      requires — so what survives is at worst an account name in a value already malformed.

    A path is dropped when it carries a userinfo delimiter AND the authority carried one too:
    there the credential is known to be present, and the path is where the observed shapes keep a
    second copy of it. When the authority is a plain host the ``@`` in the path is path text and
    the path survives whole.
    """
    scheme, separator, remainder = endpoint.partition("://")
    if separator:
        prefix = f"{scheme}://"
    elif endpoint.startswith("//"):
        # Protocol-relative. The marker is part of the address and is kept, but what follows it
        # is the authority, not a path — the old parser read it as one and returned it verbatim.
        prefix, remainder = "//", endpoint[2:]
    else:
        prefix, remainder = "", endpoint
    # An authority never begins with a slash. Any leading slash left here comes from an
    # empty-authority spelling (``http:///user:pass@host``); the authority is the bytes after it.
    remainder = remainder.lstrip("/")

    cut = _first_index(remainder, _AUTHORITY_END)
    authority, tail = remainder[:cut], remainder[cut:]
    path = tail[: _first_index(tail, _PATH_END)]
    userinfo, host_port = _split_userinfo(authority)
    # An authority that already carried a delimiter told us where the host is; a later "@" is
    # then ordinary path text. It is the OTHER case — no delimiter in the authority, one after
    # it — that can mean the authority is not an address; and only if the authority holds a ":",
    # because `account:password-prefix` is the misreading and a password needs that colon.
    malformed = not userinfo and ":" in authority and any(d in tail for d in _USERINFO_DELIMITERS)
    if malformed:
        userinfo, host_port, path = *_split_userinfo(remainder), ""
    elif userinfo and any(delimiter in path for delimiter in _USERINFO_DELIMITERS):
        # The credential is known to be in this value, and the path is where the observed shapes
        # keep a second copy of it. With no userinfo the "@" is just path text; keep the path.
        path = ""
    return prefix, userinfo, host_port, path, malformed


# --------------------------------------------------------------------------- #
# The ACCEPT grammar. Everything above this line finds secrets; everything below decides what may
# be configured and published, and it is an allowlist.
# --------------------------------------------------------------------------- #

# One DNS label. Split-then-match rather than one nested-quantifier regex over the whole hostname:
# `^L(?:\.L)*$` with `L = [A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?` nests `*` inside `*` and
# backtracks super-linearly on a long near-match, and this pattern is applied to an env var.
_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")
# `http` or `https`, case-insensitive, and NOTHING else. `ftp://`, `s3://`, `admin:pass://` and a
# bare `://` are refused rather than stripped: a scheme this code does not understand is a value
# whose shape it cannot vouch for.
_SCHEME = re.compile(r"^(?i:https?)://")
_MAX_HOSTNAME = 253
_MAX_LABEL = 63
_MAX_PORT = 65535


def _is_hostname(host: str) -> bool:
    """A full hostname: dot-separated labels of ``[A-Za-z0-9-]``, none empty, none hyphen-edged.

    Subsumes a dotted IPv4 literal — ``127.0.0.1`` is four all-digit labels — so there is no
    second production for it. An out-of-range dotted quad (``999.1.1.1``) is therefore accepted
    as a hostname, which is what it is: a name that will not resolve, not a way to smuggle
    anything past this function.

    The empty label is what closes ``rustfs..internal`` and the trailing-dot FQDN ``rustfs.``.
    """
    if not host or len(host) > _MAX_HOSTNAME:
        return False
    return all(len(label) <= _MAX_LABEL and _LABEL.fullmatch(label) for label in host.split("."))


def _is_ipv6_literal(bracketed: str) -> bool:
    """``[…]`` around a real IPv6 address, brackets balanced, nothing after the closing one.

    :mod:`ipaddress` is the arbiter rather than a character-class regex, which is what accepted
    ``http://[::1:9000`` (no closing bracket) and ``[::1]secret`` (bytes after it). A scope id
    (``[fe80::1%eth0]``) is refused: httpfs and botocore have no use for one and ``%`` is the
    percent-encoding lead-in this parser must never have to interpret.
    """
    if not bracketed.startswith("[") or not bracketed.endswith("]"):
        return False
    inner = bracketed[1:-1]
    if not inner or "%" in inner:
        return False
    try:
        return ipaddress.ip_address(inner).version == 6
    except ValueError:
        return False


def _parse_authority(authority: str) -> tuple[str, str | None] | None:
    """``(host, port)`` when ``authority`` is exactly ``host[:port]``, else ``None``.

    The port is validated, which is the hole the previous rounds left open on every side of it:
    ``http://admin:secret`` has no ``@``, so ``secret`` read as a port and the PASSWORD was
    configured into httpfs and quoted back in its errors; ``http://host:99999`` and
    ``http://host:abc`` were accepted outright. A port here is ASCII digits — ``str.isdigit`` is
    true for ``"٩٠٠٠"`` and ``int()`` then happily returns 9000 — with no leading zero and in
    ``1..65535``.
    """
    if authority.startswith("["):
        host, closed, remainder = authority.partition("]")
        # Not `host + "]"` unconditionally: `partition` returns the whole string when the
        # separator is absent, so appending one INVENTED the closing bracket and turned the
        # unbalanced `[::1:9000` into the perfectly valid literal `[::1:9000]`.
        if not closed:
            return None
        host += "]"
        if not _is_ipv6_literal(host):
            return None
    else:
        host, separator, port = authority.partition(":")
        remainder = f":{port}" if separator else ""
        if not _is_hostname(host):
            return None
    if not remainder:
        return host, None
    port = remainder[1:]
    if not remainder.startswith(":") or not port.isascii() or not port.isdigit():
        return None
    if port != str(int(port)) or not 1 <= int(port) <= _MAX_PORT:
        return None
    return host, port


def names_an_authority(value: str) -> bool:
    """True when ``value`` is exactly ``host[:port]`` under the grammar above.

    Public because :mod:`src.utils.duckdb_conn` re-derives its own guard from the FINAL string it
    is about to hand httpfs, after the scheme has been stripped. Sharing the grammar rather than
    keeping a second copy of it is deliberate: the two copies drifted, and the drift was the leak.
    """
    return _parse_authority(value) is not None


def _split_scheme(raw: str) -> tuple[str, str] | None:
    """``(prefix, remainder)`` where ``prefix`` is ``''``, ``'//'`` or a lowercased ``scheme://``.

    ``None`` for anything else, which is the whole point: an unknown scheme, an empty authority
    (``http:///x``, ``///x``) and a leading slash that is not exactly ``//`` are refusals, not
    things to strip. ``//`` is normalised to that one spelling — ``///`` is not protocol-relative,
    it is a URL with an empty authority, and the userinfo used to slide past it into the path.
    """
    match = _SCHEME.match(raw)
    if match:
        prefix, remainder = raw[: match.end()].lower(), raw[match.end() :]
    elif raw.startswith("//"):
        prefix, remainder = "//", raw[2:]
    elif raw.startswith("/"):
        return None
    else:
        prefix, remainder = "", raw
    if not remainder or remainder.startswith("/"):
        return None
    return prefix, remainder


def endpoint_credentials(endpoint: object) -> tuple[str, ...]:
    """Credential components embedded in ``endpoint``'s userinfo, raw and URL-unquoted.

    Both forms are returned because the userinfo is percent-encoded in the URL but appears decoded
    in botocore error messages, so a redactor has to match either spelling.
    """
    raw = endpoint if isinstance(endpoint, str) else ""
    _prefix, userinfo, _host_port, _path, _malformed = _split_authority(raw)
    return tuple(
        candidate
        for component in userinfo.split(":", 1)
        for candidate in (component, unquote(component))
        if candidate
    )


# The substrings that make a QUERY OR FRAGMENT PARAMETER NAME a credential, whatever else the
# parameter is called. Matched case-insensitively against the whole name, so `?token=`,
# `?api_key=`, `?X-Amz-Signature=` and `?stac_password=` are all caught while `?collections=`,
# `?limit=` and `?datetime=` are not.
#
# Names, not every parameter. A redactor built on every query VALUE would put `1` (from
# `?limit=1`) into the secret set and blank every `1` in the published document — the destructive
# redactor `redact_credentials` was already scoped down once to stop being. A credential-named
# parameter is the shape that actually carries one.
CREDENTIAL_NAME_MARKERS = (
    "secret",
    "password",
    "passwd",
    "passphrase",
    "token",
    "credential",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "signature",
    "authorization",
)

_PARAMETER_SEPARATORS = re.compile(r"[&;]")


def names_a_credential(name: object) -> bool:
    """Whether ``name`` says, in its own name, that what it labels is a credential."""
    return isinstance(name, str) and any(m in name.lower() for m in CREDENTIAL_NAME_MARKERS)


def _parameter_secrets(region: str) -> list[str]:
    """The values of the credential-named ``k=v`` pairs in one query or fragment region.

    The NAME is decoded before it is read: ``?api%5Fkey=`` is ``?api_key=`` to the server that
    receives it, and matching the raw spelling let that token through the marker check and out of
    the secret set. The VALUE is left raw here; :func:`url_secrets` adds its decoded spellings.
    """
    secrets: list[str] = []
    for pair in _PARAMETER_SEPARATORS.split(region):
        name, separator, value = pair.partition("=")
        if separator and value and names_a_credential(unquote_plus(name)):
            secrets.append(value)
    return secrets


def url_secrets(url: object) -> tuple[str, ...]:
    """Every credential ``url`` carries, from ALL THREE places a URL can hold one, LONGEST FIRST.

    :func:`endpoint_credentials` reads the userinfo and only the userinfo, and that is the RARE
    spelling. A catalogue takes its credential on the query far more often —
    ``https://stac.example/v1?token=abc`` — and the two failure modes are not symmetric:

    * the STRUCTURAL field is fine either way. The URL sanitizers drop the query from the
      PUBLISHED address, so ``environment.stac_url`` never carried the token. That is exactly
      what makes this easy to miss;
    * the PROSE field is not. ``pystac_client`` quotes the configured URL IN FULL in the
      exception a panel copies verbatim into ``errors[].error``, and with the token absent from
      the redaction set that string is republished intact through every panel writer and through
      ``run_manifest_errors[].error``. Measured, with ``STAC_URL`` carrying ``?token=abc123``::

          endpoint_credentials("https://stac.example/v1?token=abc123")  -> ()
          redact_credentials({"errors": [{"error": "APIError: ... ?token=abc123"}]}, "")
              -> {"errors": [{"error": "APIError: ... ?token=abc123"}]}

    So a credential can live in a userinfo, a query or a fragment, and the redaction set has to
    know all three. The fragment is included because it is the same ``k=v`` region reached by a
    different delimiter, and a value put there is never sent to the server but IS quoted back by
    anything that echoes the configured string.

    Longest first, and the ordering is load-bearing rather than tidy for the same reason it is in
    :func:`configured_store_secrets`: an account name routinely prefixes its own password, and
    replacing the short one first publishes the tail of the long one wearing a redaction.

    Raw and decoded, because the value is percent-encoded in the URL and appears decoded in the
    library's message. ``unquote_plus`` as well as ``unquote``: a query encodes a space as ``+``,
    which ``unquote`` leaves alone.
    """
    raw = url if isinstance(url, str) else ""
    head, _hash, fragment = raw.partition("#")
    query = head.partition("?")[2]
    candidates = [
        *endpoint_credentials(raw),
        *_parameter_secrets(query),
        *_parameter_secrets(fragment),
    ]
    spellings = {
        spelling
        for candidate in candidates
        for spelling in (candidate, unquote(candidate), unquote_plus(candidate))
        if spelling
    }
    return tuple(sorted(spellings, key=len, reverse=True))


class S3EndpointConfigurationError(RuntimeError):
    """``S3_ENDPOINT`` holds a value the accept grammar refuses.

    Its message never quotes the configured value: the commonest reason a value is refused is
    that it carries a credential where an address belongs, so echoing it to explain the problem
    would leak exactly what the refusal exists to withhold.

    Lives here rather than in :mod:`src.utils.duckdb_conn` so that the boto3 construction site in
    ``src/cli/forecast_ensemble_archive.py`` raises the same type without importing duckdb.
    """


ENDPOINT_REFUSAL = (
    "S3_ENDPOINT does not match the accepted grammar for an object-store address: an optional "
    "'http://' or 'https://' scheme (or a protocol-relative '//', or no scheme at all), then a "
    "host - a hostname, a dotted IPv4, or a bracketed IPv6 literal - then an optional ':port' in "
    "1..65535, and at most one trailing '/' after that host, which is dropped. Nothing else is "
    "accepted: no path, no query, no fragment, no 'user:password@' userinfo, no other scheme. "
    "Refusing to open the connection: configuring "
    "nothing - or configuring the empty string - would leave DuckDB httpfs on its built-in AWS "
    "endpoint and send S3_ACCESS_KEY/S3_SECRET_KEY to Amazon. Supply the credentials through "
    "S3_ACCESS_KEY/S3_SECRET_KEY instead of spelling them into the URL. The configured value is "
    "not quoted here on purpose - it may contain the password."
)


def sanitize_endpoint(endpoint: object) -> tuple[str | None, bool]:
    """``(the address, whether the configured value differed)`` under a strict ACCEPT grammar.

    The grammar, and nothing outside it::

        endpoint  := [ scheme "://" | "//" ] authority [ "/" ]
        scheme    := "http" | "https"                       (case-insensitive, normalised down)
        authority := host [ ":" port ]
        host      := hostname | dotted-IPv4 | "[" IPv6 "]"
        port      := 1..65535, ASCII digits, no leading zero

    No path. No query. No fragment. No userinfo. No other scheme. A value that does not match is
    REFUSED WHOLE — ``(None, True)`` — rather than trimmed down to the part of it that looked
    address-shaped.

    THE ONE EXCEPTION, and it is deliberate — do not "fix" it back. A SINGLE ``/`` immediately
    after an otherwise-valid authority is accepted and stripped: ``http://rustfs:9000/`` returns
    ``http://rustfs:9000``. Everything else that carries a path stays refused, including
    ``http://rustfs:9000/bucket`` and the doubled ``http://rustfs:9000//``. Four reasons, and the
    exception is bounded by all four rather than by any one of them:

    * it is not ambiguous. The slash sits AFTER a value ``_parse_authority`` has already accepted
      as ``host[:port]``, so there is no second reading of those bytes to choose between;
    * it cannot hide a credential. A credential needs a ``:`` or an ``@`` — a password lives past
      a colon and a userinfo past an ``@`` — and both are already validated inside the authority
      before this strip is considered. A trailing slash adds neither;
    * it is a very common operator and tool spelling for exactly the same address;
    * it WORKED before. ``main``'s ``duckdb_conn`` did ``endpoint.rstrip("/")``, so
      ``S3_ENDPOINT=http://rustfs:9000/`` is a deployment that is running today. Refusing it whole
      would convert a live deployment into a hard failure at startup, for a spelling that carries
      no risk — a security posture that breaks working, safe configurations gets turned off, and
      an accept grammar nobody runs protects nothing.

    Note the asymmetry with the strip that was removed: ``rstrip("/")`` on the RAW value was a
    recovery step — it ran before anything was validated, so it turned ``http://user:pass@host/``
    and ``http://host/bucket/`` into new values to keep parsing. This strip runs AFTER the accept
    decision, on a value already known to be an address, and removes exactly one character.

    That posture is the fix, and it is a posture rather than a patch because four rounds of
    recovery-style parsing each shipped a fresh way to survive it. What recovery accepted, and
    what the grammar now refuses:

    ==============================  ==================================================
    configured value                what it used to do
    ==============================  ==================================================
    ``http://admin:secret``         configured ``admin:secret`` — no ``@``, so the
                                    PASSWORD parsed as the port and httpfs quoted it
    ``ftp://rustfs:9000``           configured verbatim, TLS on, scheme never checked
    ``admin:pass://host:9000``      passed through whole
    ``/secret``                     configured ``secret`` as a hostname
    ``[::1]secret``                 passed through; brackets never balanced
    ``http://[::1:9000``            configured ``[::1:9000``
    ``http://host:99999``           accepted; the port was never validated at all
    ``http://host:abc``             accepted
    ==============================  ==================================================

    Every one of those is ``(None, True)`` now, before DuckDB or boto3 sees it.

    The two ``None`` answers stay distinguishable, because callers depend on the difference:
    ``(None, False)`` is "nothing configured" — the legitimate AWS-default deployment — and
    ``(None, True)`` is "configured, and unusable". :mod:`src.utils.duckdb_conn` raises on the
    second and takes the AWS default only on the first. ``""`` is never returned, for either:
    it reads as a configured address at every ``if address:`` and it is DuckDB's spelling of
    "use s3.us-east-1.amazonaws.com" — with ``S3_ACCESS_KEY``/``S3_SECRET_KEY`` signed into the
    next request, observed as ``HTTP 403 ... Invalid Access Key: "deposium-dev"``.

    The flag is ``True`` whenever the returned address differs from the configured value, which
    on this grammar can only mean the scheme's case was normalised, a single trailing slash was
    dropped — or that the value was refused. There is no longer any such thing as a partial
    recovery to report.

    For a value that legitimately carries a path, see :func:`sanitize_stac_url`. This function is
    for object-store endpoints, where a path is a configuration error.
    """
    raw = endpoint.strip() if isinstance(endpoint, str) else ""
    if not raw:
        return None, False
    split = _split_scheme(raw)
    if split is None:
        return None, True
    prefix, authority = split
    # The bounded trailing-slash exception documented above: exactly one `/`, and only when what
    # precedes it is already a valid authority. `authority[:-1]` still ending in `/` is the
    # doubled `host:9000//`, which stays refused; anything else after the slash is a path and
    # never reaches this strip, because `_parse_authority` refuses it below.
    if authority.endswith("/") and not authority[:-1].endswith("/"):
        authority = authority[:-1]
    parsed = _parse_authority(authority)
    if parsed is None:
        return None, True
    host, port = parsed
    address = f"{prefix}{host}:{port}" if port else f"{prefix}{host}"
    return address, address != raw


# The names :func:`sanitize_stac_url` reports for the components it removed, in the order they
# appear in a URL. Published beside the address so a reader can see that the recorded value is
# not the configured one, and exactly which part is missing.
STAC_REDUCTIONS = ("userinfo", "path", "query", "fragment")
# The one value that accompanies a ``None`` address: the URL matched no production at all. It is
# not a component name, and it never appears alongside one.
STAC_REFUSED = "refused"


def sanitize_stac_url(url: object) -> tuple[str | None, tuple[str, ...]]:
    """``(the catalogue address, what was removed to get there)`` — the STAC policy.

    A STAC catalogue root is NOT an object-store endpoint and must not be parsed as one.
    :func:`sanitize_endpoint` implements "an address, and nothing below it": no path, no query, no
    fragment, refuse the whole value otherwise. That is right for ``S3_ENDPOINT``, where a path is
    a configuration error, and wrong here twice over —

    * ``STAC_URL`` is ``https://earth-search.aws.element84.com/v1``. The path IS the catalogue
      root. Under the endpoint grammar the DEFAULT deployment publishes ``null``, losing the
      provenance the field exists to record;
    * before that grammar existed, the same helper answered
      ``sanitize_endpoint("https://stac.example/v1?token=abc")`` with
      ``("https://stac.example/v1", True)`` — the query silently gone, the flag raised, and
      nothing anywhere saying which of the two had happened.

    So the policy below is deliberate about each component rather than inheriting a decision made
    for a different kind of value:

    ==========  ========  ==============================================================
    component   verdict   why
    ==========  ========  ==============================================================
    scheme      KEPT      ``http``/``https`` only, lowercased. Anything else is refused.
    host, port  KEPT      Parsed by the same :func:`_parse_authority` the endpoint
                          grammar uses, so every unbalanced bracket, non-numeric port
                          and out-of-range port is refused here too.
    path        KEPT      It is the catalogue root. Dropped ONLY when it contains ``@``
                          or ``%40``, where this declines to decide whether that is an
                          email address inside a catalogue name or a credential spelled
                          after the authority.
    userinfo    DROPPED   A private catalogue is reached as
                          ``https://user:token@stac.internal/v1`` and this value is
                          published into a handoff manifest.
    query       DROPPED   and REPORTED. The judgement call, and it is made on what a
                          query most often carries: ``?token=``/``?api_key=`` is how a
                          catalogue takes a credential on the URL, and preserving it
                          would publish that credential into the manifest this whole
                          contract exists to make safe to hand over. A ``?collections=``
                          filter is real provenance and losing it is a real cost — which
                          is why it is not lost SILENTLY. ``"query"`` comes back in the
                          reductions, the manifests publish it beside the address, and a
                          reader can see that what is recorded is a prefix of what was
                          configured.
    fragment    DROPPED   and reported, for the same reason and with less to lose.
    ==========  ========  ==============================================================

    The second element is a tuple of :data:`STAC_REDUCTIONS` names, in URL order — ``()`` when the
    recorded address is the configured value verbatim. A refused URL comes back as
    ``(None, ("refused",))`` and an unset one as ``(None, ())``; those two stay distinguishable
    for the same reason :func:`sanitize_endpoint`'s two ``None`` answers do.

    A scheme is mandatory here, unlike on an endpoint: a catalogue URL is always a full URL.
    ``https://user:sup3r/s3cret@host:9000`` is still refused whole rather than reduced — its
    authority reads ``user:sup3r``, whose "port" is not numeric, so no reading of it is offered.
    """
    raw = url.strip() if isinstance(url, str) else ""
    if not raw:
        return None, ()
    match = _SCHEME.match(raw)
    if not match:
        return None, (STAC_REFUSED,)
    prefix, remainder = raw[: match.end()].lower(), raw[match.end() :]
    if not remainder or remainder.startswith("/"):
        return None, (STAC_REFUSED,)
    cut = _first_index(remainder, _AUTHORITY_END)
    userinfo, host_port = _split_userinfo(remainder[:cut])
    parsed = _parse_authority(host_port)
    if parsed is None:
        return None, (STAC_REFUSED,)
    host, port = parsed
    tail = remainder[cut:]
    path = tail[: _first_index(tail, _PATH_END)]
    query, has_fragment, _fragment = tail[len(path) :].partition("#")

    reductions: list[str] = []
    if userinfo:
        reductions.append("userinfo")
    if any(delimiter in path for delimiter in _USERINFO_DELIMITERS):
        # A PORT plus an `@` below the authority is refused whole rather than reduced, because
        # `host:port` and `user:password` are the same two-token shape and the `@` says which one
        # the operator was writing. Measured before this guard:
        #
        #     sanitize_stac_url("https://user:1234/5678@stac.example/v1")
        #         -> ("https://user:1234", ("path",))
        #
        # The authority ends at the first `/`, so the intended userinfo `user:1234/5678` was cut
        # in two, `user:1234` parsed as host `user` port `1234`, and the published address was
        # the username plus the head of an all-numeric password -- reported as a `path`
        # reduction, which never says the word userinfo. The docstring above already claimed
        # `https://user:sup3r/s3cret@host:9000` was "refused whole rather than reduced"; that was
        # true only while the password had a non-numeric character in its first segment.
        #
        # Without a port the authority is a bare hostname, which cannot spell `user:password`, so
        # `https://stac.example/catalogs/user@example.com/v1` keeps its host -- the email-in-a-
        # catalogue-name case this drop was written for is untouched.
        #
        # `(None, (STAC_REFUSED,))`, not the `(None, True)` this arrived from #143 as: the two
        # functions have different return types, and a bare `True` where a reductions tuple
        # belongs is a refusal no caller can read.
        if port:
            return None, (STAC_REFUSED,)
        path = ""
        reductions.append("path")
    if query:
        reductions.append("query")
    if has_fragment:
        reductions.append("fragment")

    address = f"{prefix}{host}:{port}{path}" if port else f"{prefix}{host}{path}"
    return address, tuple(reductions)


REDACTED = "[REDACTED]"


# A URL-DERIVED candidate shorter than this is not admitted as a redaction pattern. The filter used
# to be "non-empty", and a short value is a substring of ordinary prose everywhere it occurs:
# measured with `?token=1`, `"zone 1 of 15 failed after 1 retry"` came out as
# `"zone [REDACTED] of [REDACTED]5 failed after [REDACTED] retry"`, and with a two-character token
# `"IOException: about to abort"` became `"IOException: [REDACTED]out to [REDACTED]ort"`.
#
# Both failures are certain; the one this trades against is theoretical. A credential of one or two
# characters protects nothing, while a redactor that mangles every diagnostic destroys the field it
# was added to keep readable — and a downstream reader cannot tell a redaction from a corruption.
# Eight is below every real credential this deployment can hold (`deposium-dev` is twelve, an AWS
# access key id is twenty) and above the lengths that collide with prose.
#
# It applies to the candidates READ OUT OF A URL — a userinfo, a `?token=` query, a fragment — and
# to those only. The explicitly configured `S3_ACCESS_KEY`/`S3_SECRET_KEY` are exempt: a value
# someone put in `S3_SECRET_KEY` IS the credential whatever its length, and applying the floor to
# it silently republished a seven-character store key that `main` had been redacting.
MIN_SECRET_LENGTH = 8

# The ambient AWS environment DuckDB httpfs falls back to when the ``S3_*`` settings are unset:
# credentials exported into the process, as a CI secret or an ``aws sso login`` shell does. httpfs
# quotes the access key straight back in its errors (``Invalid Access Key: "AKIA…"``), which then
# travel into published run manifests. Kept beside MIN_SECRET_LENGTH because these are CONFIGURED
# credentials, not URL-derived candidates: the length floor must never apply to them.
#
# SCOPE, and it is narrower than "the AWS default": an EC2 instance profile, an ECS task role and
# ``AWS_PROFILE`` all deliver credentials WITHOUT populating these variables — through IMDS, the
# container credential endpoint, or ``~/.aws/credentials``. The boto3 signing bridge below this
# block covers values actually selected from those providers without resolving them here.
AMBIENT_STORE_ENV_VARS: tuple[str, ...] = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
)

# Credentials acquired through botocore's non-environment providers do not exist until a request
# is signed.  The signing hook in ``forecast_ensemble_archive`` records the exact immutable
# credential generation used for that request here; every redaction boundary can then consume it
# without resolving the provider chain itself. Values are retained, deduplicated, for this short-
# lived CLI process rather than evicted after an arbitrary number of rotations: an evicted access
# key can still occur in a delayed error and would then be republished. Nothing is persisted, no
# provider object is retained, and the set grows only when this process actually signs with a new
# access/secret/token value. A future long-lived boto3 service should replace this deliberately
# conservative process-lifetime policy with request-scoped error handling, not a time/count guess.
_effective_store_values: set[str] = set()
_effective_store_values_lock = Lock()


def remember_effective_store_credentials(*values: object) -> None:
    """Remember one credential generation already selected for an object-store request.

    This function never resolves credentials and never performs I/O.  Values arrive only from the
    boto3 signing boundary, after the provider chain has selected them. Repeated values are
    deduplicated and values are never written to disk.
    """
    generation = tuple(
        dict.fromkeys(value for value in values if isinstance(value, str) and value.strip())
    )
    if not generation:
        return
    with _effective_store_values_lock:
        _effective_store_values.update(generation)


def effective_store_values() -> tuple[str, ...]:
    """A snapshot of provider-resolved credentials already observed while signing."""
    with _effective_store_values_lock:
        return tuple(_effective_store_values)


def ambient_store_values() -> tuple[str, ...]:
    """The ambient ``AWS_*`` credentials, read at CALL time.

    Shared by :func:`configured_store_secrets` and :func:`publication_secrets` so the two cannot
    drift on WHICH credentials exist, the way they silently did on the ambient environment (#147):
    ``_spellings`` already stopped them drifting on how a credential is spelled, but the source
    list was still duplicated. Read per call, never cached: a process may set these after import,
    and a cached empty tuple would republish the very key it exists to scrub.
    """
    return tuple(os.environ.get(name, "") for name in AMBIENT_STORE_ENV_VARS)


def configured_store_secrets(endpoint: object) -> tuple[str, ...]:
    """Every spelling of every configured object-store credential, LONGEST FIRST.

    FOUR sources, because this deployment can hold a store credential in four different ways:
    the userinfo of ``endpoint``, the separately supplied ``S3_ACCESS_KEY``/``S3_SECRET_KEY``,
    and — when the ``S3_*`` settings are unset — the ambient ``AWS_*`` environment that DuckDB
    httpfs falls back to (:data:`AMBIENT_STORE_ENV_VARS`), plus credentials actually observed at
    the boto3 signing boundary. Missing the third source made this function return an empty tuple
    in the AWS-default environment shape (#147); missing the fourth left shared profiles,
    ECS/IMDS roles and IRSA invisible even after boto3 had selected them (#155).

    The userinfo is only one of the four, and it is the RARE one. ``docker-compose.yml`` — and every deployment modelled on it — sets
    ``S3_ENDPOINT=http://rustfs:9000`` with the credentials supplied SEPARATELY through
    ``S3_ACCESS_KEY``/``S3_SECRET_KEY``. A redactor built on the userinfo alone therefore scrubs
    nothing at all in the normal shape, which is the shape that actually runs, while DuckDB
    httpfs quotes the configured ``s3_access_key_id`` in its error messages:

        Invalid Error: ... http_error: HTTP GET error on '...' (HTTP 403) ... AccessDenied ...
        <AWSAccessKeyId>rustfs-admin</AWSAccessKeyId>

    and those messages travel into published run manifests verbatim.

    Ordering is load-bearing, not tidy: this repo's own dev store uses account ``deposium-dev``
    with password ``deposium-dev-secret``, so the account name is a PREFIX of the password.
    Replacing the short one first rewrites the password to ``[REDACTED]-secret`` and publishes
    the tail — which every "the whole secret is absent" assertion passes. Longest first.

    Empty and whitespace-only values are dropped before they can become a pattern: ``""`` is a
    substring of every position in every string, so admitting one would blank the document. That
    is the ONLY filter on the explicitly configured keys. The ``_MIN_SECRET_LENGTH`` floor is for
    the URL-DERIVED candidates alone — userinfo, query, fragment — where a short token is prose
    noise (``?token=1``) far more often than a credential; a short ``S3_SECRET_KEY`` is still the
    credential, and must still be scrubbed.

    Three spellings, because a credential is quoted decoded by botocore, percent-encoded by a
    URL, and JSON-escaped by anything that embeds a serialized document inside an error string —
    a 403 body quoted back into an exception message arrives as
    ``{\"AWSAccessKeyId\":\"depos\\u00efum\"}``, where neither the raw nor the URL spelling
    matches. ``json.dumps(value)[1:-1]`` is that third one: the value as it appears BETWEEN the
    quotes of a JSON string, with ``"``, ``\\`` and every non-ASCII character escaped.
    """
    return _spellings(
        configured=(
            getattr(settings, "S3_ACCESS_KEY", ""),
            getattr(settings, "S3_SECRET_KEY", ""),
            *ambient_store_values(),
            *effective_store_values(),
        ),
        derived=tuple(url_secrets(endpoint)),
    )


def _spellings(*, configured: tuple[object, ...], derived: tuple[object, ...]) -> tuple[str, ...]:
    """Every spelling of every value in ``values``, LONGEST FIRST, empties dropped.

    Factored out so :func:`configured_store_secrets` and :func:`publication_secrets` cannot
    drift: the ordering rule and the spelling table are the load-bearing parts and a second
    copy of either is a second place for a leak to appear.


    Two inputs on purpose, filtered differently: a ``configured`` value (``S3_ACCESS_KEY``,
    ``S3_SECRET_KEY``) is the credential whatever its length and is only dropped when empty;
    a ``derived`` value, read out of a URL, must reach ``MIN_SECRET_LENGTH`` before it is
    admitted as a pattern, because a short URL token is prose noise far more often than a
    credential. Applying the floor to both republished a seven-character store key.
    """
    values = (
        *(value for value in configured if isinstance(value, str) and value.strip()),
        *(
            value
            for value in derived
            if isinstance(value, str) and len(value.strip()) >= MIN_SECRET_LENGTH
        ),
    )
    spellings = {
        spelling
        for value in values
        for encoded in (
            value.strip(),
            quote(value.strip(), safe=""),
            quote_plus(value.strip(), safe=""),
            unquote(value.strip()),
            json.dumps(value.strip())[1:-1],
        )
        # Percent-encoding is case-insensitive in its hex digits: ``%2f`` and ``%2F`` both decode
        # to ``/``. ``quote`` only ever emits UPPERCASE, and the replacement downstream is a
        # case-sensitive ``str.replace``, so a producer that emits lowercase escapes — some SDKs
        # and hand-built URLs do — would sail straight through an otherwise-covered secret.
        for spelling in _percent_case_variants(encoded)
        if spelling
    }
    return tuple(sorted(spellings, key=len, reverse=True))


def _percent_case_variants(spelling: str) -> tuple[str, ...]:
    """``spelling`` with its percent-escape hex digits folded BOTH ways.

    Percent-encoding is case-insensitive in its hex digits — ``%2f`` and ``%2F`` both decode to
    ``/`` — so a pattern in one case does not match a producer emitting the other. Both directions
    are needed, not just one: ``quote`` emits UPPERCASE, so the lowercase twin covers a producer
    that emits ``%2f``; but a credential whose own characters already spell ``%2f`` is echoed as
    ``%2F`` by any producer that normalises escapes, which RFC 3986 §6.2.2.1 recommends. Folding
    only one way left that second case publishable.

    Only the two hex digits after a ``%`` are rewritten — folding the whole string would produce a
    pattern that no longer matches the credential itself, which is case-SENSITIVE everywhere else.
    """
    variants = {
        spelling,
        _PERCENT_ESCAPE.sub(lambda m: m.group(0).lower(), spelling),
        _PERCENT_ESCAPE.sub(lambda m: m.group(0).upper(), spelling),
    }
    return tuple(variants)


_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")


def publication_secrets(endpoint: object) -> tuple[str, ...]:
    """Every credential a PUBLISHED document may have to have punched out of it.

    :func:`configured_store_secrets` answers one question — "what does this process use to reach
    the object store" — and that is the right question for the archive CLI, which only ever talks
    to the store. It is not the whole question at a publication boundary. A private STAC
    catalogue is reached with a SECOND, unrelated credential spelled into ``STAC_URL``
    (``https://user:token@stac.internal/v1``), and ``pystac_client`` quotes that URL IN FULL in
    the exception a panel copies verbatim into ``errors[].error``. Measured here, on
    pystac_client 0.9.x / requests 2.x, with ``STAC_URL`` carrying a userinfo::

        APIError: Failed to parse: https://stac-reader:stac-s3cret@stac.internal:notaport/v1
        APIError: Invalid URL 'https://stac-reader:stac-s3cret@/v1': No host supplied
        FileNotFoundError: [Errno 2] No such file or directory:
            'stac-reader:stac-s3cret@stac.origin/v1'

    (A connection-level failure does NOT leak — urllib3 rebuilds the message from host and port
    alone — so the exposure is the URL-parse family, which is exactly what a misconfigured
    private catalogue produces.)

    Sanitizing ``inputs[].endpoint`` does not help: that is a different field, and the address
    published there was never where the credential survived. Measured before this function
    existed, ``climate_baseline_panel._write_s3_json`` — the writer ``ndvi_pilot_panel`` also
    uses — published that whole line, token included, into the destination bucket.

    A THIRD family joined in #160: the Open-Meteo commercial tier's ``apikey``, which travels in a
    query string rather than in a userinfo, and reaches the MCP client through
    ``metadata.provider_chain[].reason`` rather than through a manifest. The lesson each addition
    repeats is that "credential" here is not a synonym for "store credential" — this function is
    defined by the BOUNDARY (anything a published document or a returned payload must have punched
    out of it), which is why :func:`configured_store_secrets` stays deliberately narrower and is
    the wrong thing to widen.
    """
    return _spellings(
        configured=(
            getattr(settings, "S3_ACCESS_KEY", ""),
            getattr(settings, "S3_SECRET_KEY", ""),
            # Same ambient AWS_* set configured_store_secrets uses — a published manifest is
            # exactly where the httpfs-quoted ambient key was landing verbatim (#147).
            *ambient_store_values(),
            # Every generation botocore has signed with during this CLI process. Reading this
            # in-memory set cannot contact IMDS/ECS/STS or a local profile.
            *effective_store_values(),
            # A THIRD credential family, and the one this function was still blind to (#160). The
            # Open-Meteo commercial tier authenticates with `?apikey=...` in the query string, and
            # `httpx.HTTPStatusError.__str__` embeds the request URL IN FULL. `src/utils/forecast.py`
            # records that as `entry["reason"]`, which rides `provider_chain` out to the MCP client
            # inside `metadata` — a client-visible boundary, not just a log line.
            #
            # Nothing else about the redaction was wrong: `reason` is already a DIAGNOSTIC_FIELDS
            # key and the walker already descends the nested list. The value was simply not a
            # secret as far as this function was concerned, so there was nothing to match.
            #
            # `configured=` and not `derived=`: an explicitly configured credential is a credential
            # whatever its length, so it must not be subject to the MIN_SECRET_LENGTH floor.
            getattr(settings, "FORECAST_API_KEY", ""),
            getattr(settings, "FORECAST_ARCHIVE_API_KEY", ""),
            getattr(settings, "FORECAST_ENSEMBLE_API_KEY", ""),
        ),
        derived=(
            *url_secrets(endpoint),
            *url_secrets(getattr(settings, "STAC_URL", "")),
        ),
    )


# The fields a credential can actually land in: prose. Every one of them holds text a human
# reads, produced by `f"{type(exc).__name__}: {exc}"` or by a hand-written note — never an
# identifier, a URI or a value another program parses. Derived from the payloads this repo
# publishes, not invented: grep the panel scripts for the keys they build manifests out of.
DIAGNOSTIC_FIELDS = frozenset(
    {
        "detail",
        "details",
        "error",
        "errors",
        "exception",
        "fallback_reason",
        "message",
        "messages",
        "ndvi_truncation_warning",
        "note",
        "notes",
        "reason",
        "run_manifest_errors",
        "source_fallback_reason",
        "stderr",
        "stdout",
        "stop_reason",
        "traceback",
        "truncation_warnings",
        "warning",
        "warnings",
    }
)


def redact_text(text: str, endpoint: object) -> str:
    """Every configured credential punched out of one string that is KNOWN to be prose.

    :func:`redact_credentials` decides whether a string is prose from the key it hangs under, and
    a bare string has no key. Handing it one therefore returns it UNTOUCHED — correct for the
    structural values that scoping exists to protect, and silently wrong for the one caller that
    has a diagnostic in its hand and no dict around it: ``climate_manifest_consumer.main`` prints
    ``error: {exception}`` to stderr, and every exception it catches comes back from DuckDB
    httpfs, which quotes the endpoint in full.

    So the two entry points are split by what the CALLER knows. ``redact_credentials`` walks a
    document and infers; this one is told. It is the leaf both share.
    """
    for secret in publication_secrets(endpoint):
        text = text.replace(secret, REDACTED)
    return text


def redact_credentials(payload: Any, endpoint: object) -> Any:
    """``payload`` with every configured credential punched out of its DIAGNOSTIC fields.

    ``sanitize_endpoint`` cleans the ONE field that is meant to hold the address. It cannot help
    with the fields that hold *prose*, and a run manifest is full of those: every ``errors[].error``
    and ``run_manifest_errors[].error`` entry is ``f"{type(exc).__name__}: {exc}"`` taken verbatim
    from whatever failed. Both object-store clients this repo uses name the endpoint IN FULL in
    those messages — measured, not assumed, on duckdb 1.5.4 and botocore 1.43.57:

    * ``IOException: IO Error: Could not connect to server error for HTTP HEAD to
      'http://user:pass@host:9000/bucket/key'`` — httpfs quotes whatever address it was
      configured with;
    * ``EndpointConnectionError: Could not connect to the endpoint URL:
      "http://user:pass@host:9000/bucket/key"`` — botocore.

    So a single failed zone puts the store's password into the manifest published beside the
    sanitized ``environment.s3_endpoint`` — and ``import-zip`` then republishes it into the
    DESTINATION bucket, the one being handed to another party.

    SCOPE, and it is the part that changed: only values under a key in :data:`DIAGNOSTIC_FIELDS`
    are rewritten. This used to rewrite every string anywhere in the payload, which made it a
    destructive boundary rather than a redacting one. Measured on the dev store's own
    configuration — ``S3_ACCESS_KEY=deposium-dev``, ``S3_BUCKET=deposium``::

        run_id            "deposium-dev-2026-08-21"
                       -> "[REDACTED]-2026-08-21"
        run_manifest_uri  "s3://deposium/climate/runs/deposium-dev-2026-08-21/manifest.json"
                       -> "s3://deposium/climate/runs/[REDACTED]-2026-08-21/manifest.json"
        outputs[].uri     "s3://deposium/climate/unified/deposium-dev/z.parquet"
                       -> "s3://deposium/climate/unified/[REDACTED]/z.parquet"

    The run id no longer matches the run, and every URI in the manifest points at an object that
    does not exist — a redactor that corrupts the artifact it is protecting will be turned off,
    which protects nothing. A credential cannot reach those fields in the first place: they are
    built from a run id, a bucket name and a zone id, all code- or operator-controlled, and none
    of them is ever an exception message.

    The scoping resets at every dict rather than inheriting: under ``errors: [{...}]`` the
    ``error`` value is prose and the sibling ``zone_id`` is an identifier, so ``errors`` opening
    the scope must not put the whole entry inside it. A list DOES inherit, so a bare
    ``warnings: ["…"]`` is covered.

    Applied to the whole payload at the publication boundary rather than at each ``except`` site:
    there are a dozen of those across the panels and one more gets added with every new stage,
    whereas there is exactly one place where a manifest becomes an object in a bucket. Recursion
    keeps non-string leaves (dates, numbers) untouched, so the DuckDB ``COPY`` that publishes the
    payload still binds the types it was given.

    Dict KEYS are never rewritten either, at any depth. A manifest key is code-controlled and
    becomes a COLUMN NAME in that ``COPY``: with ``S3_ACCESS_KEY='s3'``, ``{'s3_endpoint': …}``
    came back as ``{'[REDACTED]_endpoint': …}``, which fails the ``^[A-Za-z_][A-Za-z0-9_]*$``
    identifier check in the panel writers outright.

    The secrets come from :func:`publication_secrets`, NOT from the endpoint's userinfo alone.
    Keying it on the userinfo made it a complete no-op in the normal deployment shape —
    ``S3_ENDPOINT=http://rustfs:9000`` with ``S3_ACCESS_KEY``/``S3_SECRET_KEY`` supplied
    separately — so it protected against the rare misconfiguration and not against the standard
    configuration, while httpfs kept quoting ``s3_access_key_id`` into the manifests.

    :func:`publication_secrets`, not :func:`configured_store_secrets`, and this docstring named
    the wrong one after 683a4ef moved the call: the two differ by the STAC credential spelled
    into ``STAC_URL``, which ``pystac_client`` quotes in full into the exception text a panel
    copies verbatim into ``errors[].error``. A reader checking here for what a published manifest
    is scrubbed against would have concluded the catalogue token was not covered. It is.
    Ordering (longest first), the empty-value guard and the three spellings live in
    :func:`_spellings`, which both share; see those docstrings.
    """
    secrets = publication_secrets(endpoint)
    if not secrets:
        return payload

    def scrub(node: Any, diagnostic: bool) -> Any:
        if isinstance(node, str):
            if not diagnostic:
                return node
            for secret in secrets:
                node = node.replace(secret, REDACTED)
            return node
        if isinstance(node, dict):
            # Keys untouched, and the scope RESETS here: whether a value is prose is decided by
            # its own key, not by an ancestor's.
            return {key: scrub(value, key in DIAGNOSTIC_FIELDS) for key, value in node.items()}
        if isinstance(node, list):
            return [scrub(value, diagnostic) for value in node]
        if isinstance(node, tuple):
            return tuple(scrub(value, diagnostic) for value in node)
        return node

    return scrub(payload, False)
