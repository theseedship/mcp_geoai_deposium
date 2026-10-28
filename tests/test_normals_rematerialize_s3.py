"""The normals operator's object-store mode (#223 A), against a fake store.

Hermetic: the store is a dict behind the four S3 calls the operator makes, with md5 etags, a 412
on a failed ``If-None-Match``/``If-Match`` and a call log, as in
`tests/test_forecast_ensemble_archive.py`. No boto3 client is constructed and nothing reaches the
network. The ERA5 inputs come from the local-mode fixture, and the "current canonical" is a real
v1 normals parquet: the v2 computed from the same days with the v2 columns dropped.
"""

from __future__ import annotations

import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import normals_rematerialize as op
from src.config import settings
from tests.test_normals_rematerialize import YEAR_END, YEAR_START, _write_era5_year

BUCKET = "deposium"
HOST = "rustfs"
ENDPOINT = f"{HOST}:9000"
COMMIT = "9a2232e" + "0" * 33
AUTHORIZATION = "https://github.com/theseedship/deposium_geoai/issues/172#issuecomment-1"
ZONES = ("niamey", "fos_sur_mer")

#: The MCPs science-manifest pattern (deposium_MCPs `src/services/geoai-object-store.ts`).
MCPS_MANIFEST = re.compile(r"^climate\/(?!runs\/)(?:[^/]+\/)*manifest\.json$")


class _StoreError(RuntimeError):
    def __init__(self, status: int, code: str) -> None:
        super().__init__(f"{status} {code}")
        self.response = {
            "ResponseMetadata": {"HTTPStatusCode": status},
            "Error": {"Code": code},
        }


def _etag(body: bytes) -> str:
    return f'"{hashlib.md5(body).hexdigest()}"'  # noqa: S324 - an S3 etag, not a digest claim


class FakeObjectStore:
    """HEAD, GET and PUT over a dict. It has no delete and no copy, so neither can be called."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []
        self.put_requests: list[dict] = []
        self.on_put = None  # (key, body) -> body actually stored, or raise
        # What a failed precondition answers. RustFS or S3 may say 409 as well as 412.
        self.conflict = (412, "PreconditionFailed")

    def _key(self, request: dict) -> str:
        assert request["Bucket"] == BUCKET
        return request["Key"]

    def head_object(self, **request):
        key = self._key(request)
        self.calls.append(("head_object", key))
        if key not in self.objects:
            raise _StoreError(404, "404")
        body = self.objects[key]
        return {"ETag": _etag(body), "ContentLength": len(body)}

    def get_object(self, **request):
        key = self._key(request)
        self.calls.append(("get_object", key))
        if key not in self.objects:
            raise _StoreError(404, "NoSuchKey")
        body = self.objects[key]
        return {"Body": BytesIO(body), "ETag": _etag(body), "ContentLength": len(body)}

    def put_object(self, **request):
        key = self._key(request)
        self.calls.append(("put_object", key))
        self.put_requests.append(request)
        if request.get("IfNoneMatch") == "*" and key in self.objects:
            raise _StoreError(*self.conflict)
        if_match = request.get("IfMatch")
        if if_match is not None and (
            key not in self.objects or _etag(self.objects[key]) != if_match
        ):
            raise _StoreError(*self.conflict)
        body = request["Body"]
        if self.on_put is not None:
            body = self.on_put(key, body)
        self.objects[key] = body

    def puts(self) -> list[str]:
        return [key for verb, key in self.calls if verb == "put_object"]

    def put_requests_for(self, key: str) -> list[dict]:
        return [request for request in self.put_requests if request["Key"] == key]


#: Both answers a store gives to a failed precondition.
CONFLICTS = pytest.mark.parametrize(
    "conflict", [(412, "PreconditionFailed"), (409, "ConditionalRequestConflict")]
)
FOREIGN = b"a concurrent writer's object"


@pytest.fixture(scope="module")
def seed(tmp_path_factory):
    """ERA5 bytes and a real v1 canonical per zone, computed once for the module."""
    import duckdb

    root = tmp_path_factory.mktemp("seed")
    out: dict[str, dict[str, bytes]] = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
        mp.setattr(settings, "LOCAL_PARQUET_SOURCE_ROOT", root.as_posix())
        for zone_id in ZONES:
            era5 = _write_era5_year(root, zone_id)
        argv = ["materialize", "--root", root.as_posix(), "--era5-root", root.as_posix()]
        argv += ["--run-id", "seed", "--zone-id", *ZONES]
        assert op.main([*argv, "--start-date", YEAR_START, "--end-date", YEAR_END]) == 0
    for zone_id in ZONES:
        era5 = root / op._ERA5.format(zone_id=zone_id)
        v2 = root / op._STAGING.format(run_id="seed", zone_id=zone_id)
        v1 = root / f"{zone_id}.v1.parquet"
        con = duckdb.connect(":memory:")
        con.execute(
            "COPY (SELECT * EXCLUDE (p10_wet, p50_wet, p90_wet, n_wet, wet_day_threshold_mm, "
            f"schema_version) FROM read_parquet('{v2}')) TO '{v1}' (FORMAT PARQUET)"
        )
        con.close()
        out[zone_id] = {"era5": era5.read_bytes(), "v1": v1.read_bytes()}
    return out


@pytest.fixture
def env(tmp_path, monkeypatch, seed):
    """A configured S3 mode over a fake store holding ERA5 and a v1 canonical for every zone."""
    monkeypatch.setattr(settings, "ALLOW_LOCAL_PARQUET_SOURCE", True)
    monkeypatch.setattr(settings, "LOCAL_PARQUET_SOURCE_ROOT", tmp_path.as_posix())
    monkeypatch.setattr(settings, "S3_BUCKET", BUCKET)
    monkeypatch.setattr(settings, "S3_ENDPOINT", f"http://{ENDPOINT}")
    # The container shape: a commit injected through GIT_COMMIT, and no git to observe the tree.
    monkeypatch.setattr(
        op, "code_ref", lambda: {"repo": "deposium_geoai", "commit": COMMIT, "dirty": None}
    )
    # Canonical is the 1991-2020 baseline, and this ERA5 covers one year: the guard is narrowed to
    # that year here, and TestOnlyTheBaselineWindowReachesCanonical pins it at its real value.
    monkeypatch.setattr(op, "_CANONICAL_WINDOW", {"start": YEAR_START, "end": YEAR_END})
    fake = FakeObjectStore()
    for zone_id in ZONES:
        fake.objects[op._ERA5.format(zone_id=zone_id)] = seed[zone_id]["era5"]
        fake.objects[op._CANONICAL.format(zone_id=zone_id)] = seed[zone_id]["v1"]
    state = SimpleNamespace(fake=fake, root=tmp_path / "work", clients=0, seed=seed)

    def _client():
        state.clients += 1
        return fake

    monkeypatch.setattr(op, "_object_store_client", _client)

    # `verify` reads through `climate_normals`' own function and URI. DuckDB httpfs cannot reach
    # a dict, so the URI is checked and its bytes are served from a local copy of the fake key.
    real = op._read_normals_rows

    def _served(uri, *, zone_id, variables):
        assert uri == f"s3://{BUCKET}/climate/normals/{zone_id}.parquet"
        local = tmp_path / "served" / f"{zone_id}.parquet"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(fake.objects[uri.removeprefix(f"s3://{BUCKET}/")])
        return real(local.as_posix(), zone_id=zone_id, variables=variables)

    monkeypatch.setattr(op, "_read_normals_rows", _served)
    return state


def _argv(env, step: str, *zones: str, run_id: str = "r1", omit: str | None = None) -> list[str]:
    argv = [step, "--root", env.root.as_posix(), "--run-id", run_id, "--zone-id", *zones]
    flags = {
        "--store": "s3",
        "--confirm-bucket": BUCKET,
        "--confirm-endpoint": ENDPOINT,
        "--authorization": AUTHORIZATION,
    }
    for flag, value in flags.items():
        if flag != omit:
            argv += [flag, value]
    if step == "materialize":
        argv += ["--start-date", YEAR_START, "--end-date", YEAR_END]
    return argv


def _run(env, step: str, *zones: str, run_id: str = "r1", extra: tuple = ()) -> int:
    return op.main([*_argv(env, step, *zones, run_id=run_id), *extra])


def _with(argv: list[str], flag: str, value: str) -> list[str]:
    at = argv.index(flag)
    return [*argv[:at], flag, value, *argv[at + 2 :]]


def _run_events(env, run_id: str = "r1") -> list[dict]:
    prefix = f"climate/normals/events/runs/{run_id}/"
    return [
        json.loads(body) for key, body in sorted(env.fake.objects.items()) if key.startswith(prefix)
    ]


def _prepare(env, *zones: str, run_id: str = "r1") -> None:
    for step in ("fetch", "materialize", "publish", "plan"):
        assert _run(env, step, *zones, run_id=run_id) == 0, step


def _canonical(env, zone_id: str) -> bytes | None:
    return env.fake.objects.get(op._CANONICAL.format(zone_id=zone_id))


def _staged(env, zone_id: str, run_id: str = "r1") -> bytes:
    return env.fake.objects[op._STAGING.format(run_id=run_id, zone_id=zone_id)]


def _journal(env, zone_id: str) -> list[dict]:
    prefix = f"climate/normals/events/zones/{zone_id}/"
    return [
        json.loads(body) for key, body in sorted(env.fake.objects.items()) if key.startswith(prefix)
    ]


def _killed_on_the_zone_event(env, zone_id: str, *zones: str, run_id: str = "r1") -> None:
    """Promote ``zones``, killed on ``zone_id``'s event PUT: its canonical PUT landed, and no event
    of this attempt names it. A KeyboardInterrupt is not an Exception, so nothing unwinds."""
    journal = f"climate/normals/events/zones/{zone_id}/"

    def _die(key, body):
        if key.startswith(journal):
            raise KeyboardInterrupt
        return body

    env.fake.on_put = _die
    with pytest.raises(KeyboardInterrupt):
        _run(env, "promote", *zones, run_id=run_id)
    env.fake.on_put = None


def _fail_once_on_read(env, key: str) -> None:
    """The first GET of ``key`` once it exists fails, as a connection reset; the object is intact."""
    real_get = env.fake.get_object
    failed: list[str] = []

    def _get(**request):
        if request["Key"] == key and key in env.fake.objects and not failed:
            failed.append(key)
            raise ConnectionError("connection reset")
        return real_get(**request)

    env.fake.get_object = _get


def _retried_after_it_landed(env, key: str) -> None:
    """The first PUT of ``key`` lands and its response is lost, so botocore sends it again with the
    same precondition, and the store refuses that retry: the first attempt changed the key."""
    real_put = env.fake.put_object
    retried: list[str] = []

    def _put(**request):
        real_put(**request)
        if request["Key"] == key and not retried:
            retried.append(key)
            real_put(**request)  # the retry: a 412 or 409 from the fake's own precondition check

    env.fake.put_object = _put


class TestTheObjectStoreModeIsRefusedWithoutEveryConfirmation:
    @pytest.mark.parametrize("flag", ["--confirm-bucket", "--confirm-endpoint", "--authorization"])
    @pytest.mark.parametrize("step", ["fetch", "plan", "promote", "rollback"])
    def test_a_missing_confirmation_is_refused_before_any_client_exists(
        self, env, flag, step, capsys
    ):
        code = op.main(_argv(env, step, "niamey", omit=flag))

        assert code == 2
        assert flag in capsys.readouterr().err
        assert env.clients == 0
        assert env.fake.calls == []

    @pytest.mark.parametrize("step", ["fetch", "publish", "plan", "verify"])
    def test_an_object_store_step_without_store_s3_is_refused(self, env, step, capsys):
        code = op.main(_argv(env, step, "niamey", omit="--store"))

        assert code == 2
        assert "runs only with --store s3" in capsys.readouterr().err
        assert env.clients == 0

    def test_confirmations_without_store_s3_are_refused_rather_than_run_locally(self, env, capsys):
        """A promote carrying store confirmations but no `--store s3` would otherwise run the
        LOCAL promotion against the work root — the one outcome nobody typing it meant."""
        code = op.main(_argv(env, "promote", "niamey", omit="--store"))

        assert code == 2
        assert "apply only with --store s3" in capsys.readouterr().err

    def test_a_bucket_mismatch_is_refused_without_echoing_the_configured_bucket(
        self, env, monkeypatch, capsys
    ):
        argv = _argv(env, "fetch", "niamey", omit="--confirm-bucket") + ["--confirm-bucket", "x"]

        assert op.main(argv) == 2
        err = capsys.readouterr().err
        assert "does not match the configured S3_BUCKET" in err
        assert BUCKET not in err
        assert env.clients == 0

    def test_a_bucket_mismatch_does_not_echo_a_credential_typed_by_mistake(self, env, capsys):
        """The prior test's typed value ("x") is too short to prove anything: this one carries a
        scheme and inline credentials, independent of whatever is configured as a secret."""
        typed = "s3://AKIAABCDEFGHIJKLMNOP:S3cr3tKeyValue@rustfs:9000/deposium"
        argv = _with(_argv(env, "fetch", "niamey"), "--confirm-bucket", typed)

        assert op.main(argv) == 2
        err = capsys.readouterr().err
        assert "does not match the configured S3_BUCKET" in err
        assert typed not in err
        assert "AKIAABCDEFGHIJKLMNOP" not in err
        assert "S3cr3tKeyValue" not in err

    @pytest.mark.parametrize(
        ("endpoint", "confirmed"),
        [
            # The mesh, confirmed with the harness's value: a substring check would pass it.
            ("http://storage-rustfs:9000", "rustfs:9000"),
            ("http://rustfs:9000", "storage-rustfs:9000"),
            ("http://rustfs:9000", "rust:9000"),  # a prefix check would pass it
            # On the maintainer's host the harness is 127.0.0.1:9200 and the mesh localhost:9100.
            ("http://localhost:9100", "localhost:9200"),
            ("http://localhost:9100", "localhost"),  # the port is required when there is one
            ("http://rustfs:9000", "rustfs"),
        ],
    )
    def test_an_endpoint_mismatch_is_refused_even_when_the_bucket_matches(
        self, env, monkeypatch, capsys, endpoint, confirmed
    ):
        """The bucket cannot be relied on to tell a rehearsal from a real write: until 2026-09-27
        the harness and the mesh both used `deposium`, and `GEOAI_DEV_BUCKET=deposium` still makes
        them share it. Neither can a host without its port."""
        monkeypatch.setattr(settings, "S3_ENDPOINT", endpoint)
        argv = _with(_argv(env, "fetch", "niamey"), "--confirm-endpoint", confirmed)

        assert op.main(argv) == 2
        err = capsys.readouterr().err
        assert "does not match the host and port of the configured S3_ENDPOINT" in err
        assert endpoint.removeprefix("http://") not in err
        assert env.clients == 0

    @pytest.mark.parametrize(
        ("endpoint", "confirmed"),
        [
            ("http://localhost:9100", "localhost:9100"),
            ("http://RustFS:9000/", "rustfs:9000"),
            ("http://rustfs", "rustfs"),
        ],
    )
    def test_the_endpoint_is_confirmed_by_its_host_and_port(
        self, env, monkeypatch, endpoint, confirmed
    ):
        monkeypatch.setattr(settings, "S3_ENDPOINT", endpoint)
        argv = _with(_argv(env, "fetch", "niamey"), "--confirm-endpoint", confirmed)

        assert op.main(argv) == 0
        assert env.clients == 1

    @pytest.mark.parametrize(
        "confirmed",
        ["http://admin:s3cret@rustfs:9000", "admin:s3cret@rustfs:9000", "rustfs:9000/x"],
    )
    def test_an_endpoint_confirmation_that_is_not_a_bare_authority_is_refused_unechoed(
        self, env, capsys, confirmed
    ):
        argv = _with(_argv(env, "fetch", "niamey"), "--confirm-endpoint", confirmed)

        assert op.main(argv) == 2
        err = capsys.readouterr().err
        assert "takes a bare host[:port]" in err
        assert "s3cret" not in err
        assert confirmed not in err
        assert env.clients == 0

    def test_a_refusal_is_redacted_like_any_other_diagnostic(self, env, monkeypatch, capsys):
        """The bucket refusal repeats what was typed; a configured secret typed there by mistake
        must not reach stderr."""
        monkeypatch.setattr(settings, "S3_SECRET_KEY", "sup3r-s3cret-value")
        argv = _with(_argv(env, "fetch", "niamey"), "--confirm-bucket", "sup3r-s3cret-value")

        assert op.main(argv) == 2
        err = capsys.readouterr().err
        assert "does not match the configured S3_BUCKET" in err
        assert "sup3r-s3cret-value" not in err

    def test_an_endpoint_carrying_credentials_is_refused_as_unparseable(
        self, env, monkeypatch, capsys
    ):
        monkeypatch.setattr(settings, "S3_ENDPOINT", "http://admin:s3cret@rustfs:9000")

        assert op.main(_argv(env, "fetch", "niamey")) == 2
        err = capsys.readouterr().err
        assert "not a plain address" in err
        assert "s3cret" not in err

    @pytest.mark.parametrize(
        "authorization",
        [
            "http://example.org/ok",
            "https:///theseedship/deposium_geoai/issues/172",
            "https://nico:ghp_s3cretTOKEN@github.com/theseedship/deposium_geoai/issues/172",
            "https://ghp_s3cretTOKEN@github.com/theseedship/deposium_geoai/issues/172",
            f"{AUTHORIZATION}?access_token=ghp_s3cretTOKEN",
            "https://github.com/theseedship/deposium_geoai/issues/172?ref=ghp_s3cretTOKEN",
            "https://github.com/theseedship/deposium_geoai/issues/172#access_token=ghp_s3cretTOKEN",
            f"{AUTHORIZATION} ghp_s3cretTOKEN",
            "https://github.com/" + "x" * 512,
            # A clone URL with its "@github.com" cut off: no "@", so the token reads as a port...
            "https://x-access-token:ghp_s3cretTOKEN/theseedship/deposium_geoai/issues/172",
            "https://github.com:ghp_s3cretTOKEN/theseedship/deposium_geoai/issues/172",
            # ...or as the host.
            "https://ghp_s3cretTOKEN/theseedship/deposium_geoai/issues/172",
            # ...even when the token has no "_" and so is a valid hostname label (a legacy 40-hex
            # classic PAT): only github.com hosts the authorization comment.
            "https://s3cretTOKEN" + "0123456789abcdef" * 2 + "/theseedship/deposium_geoai.git",
            # A host that only ends like github.com is another host.
            "https://s3cretTOKEN" + "0123456789abcdef" * 2 + ".github.com/theseedship/x/issues/1",
            "https://nots3cretTOKENgithub.com/theseedship/deposium_geoai/issues/172",
            # urlsplit raises on these, and its NFKC message quotes the netloc, userinfo included.
            "https://nico:ghp_s3cretTOKEN\uff20github.com/theseedship/deposium_geoai/issues/172",
            "https://[ghp_s3cretTOKEN/theseedship/deposium_geoai/issues/172",
            "https://[ghp_s3cretTOKEN]/theseedship/deposium_geoai/issues/172",
        ],
    )
    def test_an_authorization_that_is_not_a_plain_https_url_is_refused_unechoed(
        self, env, capsys, authorization
    ):
        """A token in the authorization would otherwise be printed on stderr, and when accepted
        it would be written into run.json, every event and the plan."""
        argv = _with(_argv(env, "fetch", "niamey"), "--authorization", authorization)

        assert op.main(argv) == 2
        err = capsys.readouterr().err
        assert "not an https URL with a host, no credentials and no query" in err
        assert "s3cretTOKEN" not in err
        assert env.clients == 0
        assert not env.root.exists()

    def test_an_unknown_commit_is_refused(self, env, monkeypatch, capsys):
        monkeypatch.setattr(op, "code_ref", lambda: {"commit": None, "dirty": None})

        assert op.main(_argv(env, "fetch", "niamey")) == 2
        assert "GIT_COMMIT=<merge sha>" in capsys.readouterr().err
        assert env.clients == 0

    @pytest.mark.parametrize("commit", ["whatever", "9a2232e", COMMIT.upper(), f"{COMMIT}0"])
    def test_a_commit_git_did_not_observe_must_be_a_full_sha(
        self, env, monkeypatch, capsys, commit
    ):
        """The real `code_ref` in the container shape: no git, a GIT_COMMIT that nothing checks.
        Any string used to pass, recorded as the commit with `dirty: null`."""
        import src.utils.code_ref as code_ref_module

        monkeypatch.setattr(op, "code_ref", code_ref_module.code_ref)
        monkeypatch.setattr(code_ref_module, "_git", lambda *a, **k: None)
        for name in ("RAILWAY_GIT_COMMIT_SHA", "SOURCE_VERSION"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("GIT_COMMIT", commit)

        assert op.main(_argv(env, "fetch", "niamey")) == 2
        assert "not a full 40-character lowercase sha" in capsys.readouterr().err
        assert env.clients == 0

    def test_the_record_says_the_tree_was_not_observed_and_digests_the_source(self, env):
        assert op.main(_argv(env, "fetch", "niamey")) == 0

        fetched = json.loads((env.root / op._FETCH_RECORD.format(run_id="r1")).read_text())
        assert fetched["code_ref"]["commit"] == COMMIT
        assert fetched["code_ref"]["tree_observed"] is False
        # Comparable afterwards with the same digest over a clean checkout of the merge sha.
        assert fetched["code_ref"]["source_sha256"] == op._source_digest()

    def test_a_dirty_tree_is_refused(self, env, monkeypatch, capsys):
        monkeypatch.setattr(op, "code_ref", lambda: {"commit": "9a2232e", "dirty": True})

        assert op.main(_argv(env, "fetch", "niamey")) == 2
        assert "uncommitted changes" in capsys.readouterr().err
        assert env.clients == 0


class TestTheLocalAndStoreArgumentsDoNotMix:
    def test_an_object_store_uri_as_the_work_root_is_refused(self, env, capsys):
        argv = _with(_argv(env, "fetch", "niamey"), "--root", "s3://deposium/work")

        assert op.main(argv) == 2
        assert "is not a local path" in capsys.readouterr().err
        assert env.clients == 0

    def test_an_object_store_root_does_not_echo_a_credential_it_carries(self, env, capsys):
        """A copy-pasted s3:// URL given as --root can carry a key and a secret in its userinfo;
        the refusal names why the value is rejected and must not repeat any of it."""
        typed = "s3://AKIAABCDEFGHIJKLMNOP:S3cr3tKeyValue@host/bucket"
        argv = _with(_argv(env, "fetch", "niamey"), "--root", typed)

        assert op.main(argv) == 2
        err = capsys.readouterr().err
        assert "is not a local path" in err
        assert typed not in err
        assert "AKIAABCDEFGHIJKLMNOP" not in err
        assert "S3cr3tKeyValue" not in err

    @pytest.mark.parametrize(("flag", "value"), [("--run-id", "../x"), ("--zone-id", "../x")])
    def test_an_unsafe_identifier_is_refused(self, env, capsys, flag, value):
        argv = _with(_argv(env, "fetch", "niamey"), flag, value)

        assert op.main(argv) == 2
        assert "is not a safe identifier" in capsys.readouterr().err
        assert env.clients == 0

    def test_an_era5_root_is_refused_in_the_object_store_mode(self, env, capsys, tmp_path):
        argv = [*_argv(env, "materialize", "niamey"), "--era5-root", tmp_path.as_posix()]

        assert op.main(argv) == 2
        assert "--era5-root is local-mode only" in capsys.readouterr().err

    def test_a_local_materialize_without_an_era5_root_exits_2(self, tmp_path, capsys):
        argv = ["materialize", "--root", tmp_path.as_posix(), "--run-id", "r1"]

        with pytest.raises(SystemExit) as exited:
            op.main([*argv, "--zone-id", "niamey"])
        assert exited.value.code == 2
        assert "required: --era5-root" in capsys.readouterr().err

    def test_the_if_match_flag_is_refused_without_store_s3(self, tmp_path, capsys):
        """--if-match exists only for the object-store conditional PUTs; local mode has none to
        apply it to, and dropping --store s3 by mistake must refuse rather than run the local
        promotion silently instead."""
        argv = [
            "promote",
            "--root",
            tmp_path.as_posix(),
            "--run-id",
            "r1",
            "--zone-id",
            "niamey",
            "--if-match",
        ]

        assert op.main(argv) == 2
        err = capsys.readouterr().err
        assert "--if-match" in err
        assert "apply only with --store s3" in err


class TestOnlyScientificZonesArePromoted:
    def test_a_client_zone_is_refused(self, env, monkeypatch, capsys):
        zones = op._load_zones()
        zones["acme_site"] = {**zones["niamey"], "role": "client"}
        monkeypatch.setattr(op, "_load_zones", lambda: zones)

        assert op.main(_argv(env, "fetch", "niamey", "acme_site")) == 2
        assert "not scientific" in capsys.readouterr().err
        assert env.clients == 0

    def test_a_zone_given_twice_is_refused(self, env, capsys):
        assert op.main(_argv(env, "fetch", "niamey", "niamey")) == 2
        assert "more than once" in capsys.readouterr().err


class TestTheStoreKeepsItsOwnAllowlist:
    """Boto3 writes do not pass through `_validate_dst`, so the store confines itself."""

    @pytest.fixture
    def store(self, env):
        return op._ObjectStore(env.fake, bucket=BUCKET, run_id="r1", zone_ids=["niamey"])

    @pytest.mark.parametrize(
        "key",
        [
            "climate/era5/marseille.parquet",  # another zone's input
            "climate/normals/marseille.parquet",  # another zone's canonical
            "climate/normals/staging/r2/niamey.parquet",  # another run's staging
            "climate/unified/niamey.parquet",
            "climate/normals/staging/r1/../../era5/x.parquet",
        ],
    )
    def test_a_read_outside_the_allowlist_is_refused_before_the_client(self, store, env, key):
        with pytest.raises(op.OperatorRefusal, match="outside this run's key allowlist"):
            store.get(key)
        assert env.fake.calls == []

    @pytest.mark.parametrize(
        "key",
        [
            "climate/normals/niamey.parquet",  # canonical, outside promote/rollback
            "climate/era5/niamey.parquet",  # the input is never written
            "climate/normals/rollback/r2/niamey.parquet",  # another run's rollback
            "climate/normals/events/zones/marseille/000001.json",  # another zone's journal
            "climate/normals/events/runs/r2/000001.json",  # another run's journal
            "climate/runs/r1/manifest.json",
        ],
    )
    def test_a_write_outside_the_allowlist_is_refused_before_the_client(self, store, env, key):
        with pytest.raises(op.OperatorRefusal, match="outside this run's key allowlist"):
            store.put_if_absent(key, b"x", content_type="text/plain")
        assert env.fake.calls == []

    def test_canonical_is_writable_only_inside_promote_and_rollback(self, store, env):
        key = op._CANONICAL.format(zone_id="niamey")
        with pytest.raises(op.OperatorRefusal):
            store.put_canonical(key, b"x", if_match=None)

        with store.canonical_writes():
            store.put_canonical(key, b"x", if_match=None)
            with pytest.raises(op.OperatorRefusal):
                store.put_canonical(op._CANONICAL.format(zone_id="marseille"), b"x", if_match=None)
            # Inside the window put_canonical is still an overwrite of canonical ONLY: it never
            # replaces a create-only key of the run's own.
            for own in (
                op._ROLLBACK.format(run_id="r1", zone_id="niamey"),
                op._STAGING.format(run_id="r1", zone_id="niamey"),
                op._ZONE_EVENT.format(zone_id="niamey", seq=1),
            ):
                with pytest.raises(op.OperatorRefusal, match="not a canonical key"):
                    store.put_canonical(own, b"x", if_match=None)
        # And the window closes.
        with pytest.raises(op.OperatorRefusal, match="outside this run's key allowlist"):
            store.put_canonical(key, b"y", if_match=None)

        assert env.fake.objects[key] == b"x"
        assert env.fake.puts() == [key]

    def test_there_is_no_delete(self):
        assert not [name for name in dir(op._ObjectStore) if "delete" in name or "remove" in name]


class TestPlanWritesNothing:
    def test_plan_makes_zero_puts_and_reports_what_promote_would_write(self, env, capsys):
        for step in ("fetch", "materialize", "publish"):
            assert _run(env, step, *ZONES) == 0
        before = dict(env.fake.objects)
        env.fake.calls.clear()

        assert _run(env, "plan", *ZONES) == 0

        assert env.fake.puts() == []
        assert {verb for verb, _ in env.fake.calls} <= {"get_object", "head_object"}
        assert env.fake.objects == before
        plan = json.loads((env.root / op._PLAN.format(run_id="r1")).read_text())
        niamey = plan["zones"]["niamey"]
        assert niamey["canonical"]["exists"] is True
        assert niamey["canonical"]["schema_version"] == 1
        assert niamey["canonical"]["sha256"] == hashlib.sha256(env.seed["niamey"]["v1"]).hexdigest()
        assert niamey["staged"]["schema_version"] == op.NORMALS_SCHEMA_VERSION
        assert niamey["staged"]["matches_record"] is True
        assert niamey["rollback"]["state"] == "not written yet"
        # v1 and v2 from the same days: the all-days percentiles are identical, pair for pair.
        assert niamey["percentiles_all_days"] == {
            "pairs": 72,
            "identical": 72,
            "max_abs_diff": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
            "unpaired": [],
        }
        assert [w["key"] for w in niamey["writes"]] == [
            "climate/normals/rollback/r1/niamey.parquet",
            "climate/normals/niamey.parquet",
            "climate/normals/events/zones/niamey/000001.json",
        ]

    def test_plan_reports_a_percentile_difference_from_the_current_canonical(self, env, tmp_path):
        """The fixture's v1 is the v2 minus its columns, so every pair is identical and a
        comparison that never detects a difference would pass. One p50 moved shows it does."""
        import duckdb

        v1, moved = tmp_path / "v1.parquet", tmp_path / "moved.parquet"
        v1.write_bytes(env.seed["niamey"]["v1"])
        con = duckdb.connect(":memory:")
        month, variable = con.execute(
            f"SELECT month, variable FROM read_parquet('{v1}') ORDER BY 1, 2 LIMIT 1"
        ).fetchone()
        con.execute(
            "COPY (SELECT * REPLACE (CASE WHEN month = ? AND variable = ? THEN p50 + 1.5 "
            f"ELSE p50 END AS p50) FROM read_parquet('{v1}')) TO '{moved}' (FORMAT PARQUET)",
            [month, variable],
        )
        con.close()
        env.fake.objects[op._CANONICAL.format(zone_id="niamey")] = moved.read_bytes()

        _prepare(env, "niamey")

        plan = json.loads((env.root / op._PLAN.format(run_id="r1")).read_text())
        compared = plan["zones"]["niamey"]["percentiles_all_days"]
        assert compared["pairs"] == 72
        assert compared["identical"] == 71
        assert compared["max_abs_diff"]["p50"] == pytest.approx(1.5)
        assert compared["max_abs_diff"]["p10"] == 0.0

    def test_plan_reports_a_staged_object_that_is_not_the_recorded_one(self, env):
        for step in ("fetch", "materialize", "publish"):
            assert _run(env, step, "niamey") == 0
        env.fake.objects[op._STAGING.format(run_id="r1", zone_id="niamey")] = env.seed["niamey"][
            "v1"
        ]

        assert _run(env, "plan", "niamey") == 0

        staged = json.loads((env.root / op._PLAN.format(run_id="r1")).read_text())["zones"][
            "niamey"
        ]["staged"]
        assert staged["sha256"] == hashlib.sha256(env.seed["niamey"]["v1"]).hexdigest()
        assert staged["sha256"] != staged["recorded_sha256"]
        assert staged["matches_record"] is False
        assert staged["schema_version"] == 1

    def test_a_re_plan_after_a_rollback_reports_the_kept_copy(self, env):
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        assert _run(env, "rollback", "niamey") == 0

        assert _run(env, "plan", "niamey") == 0

        niamey = json.loads((env.root / op._PLAN.format(run_id="r1")).read_text())["zones"][
            "niamey"
        ]
        assert niamey["rollback"]["state"] == "kept copy exists"
        assert niamey["journal"]["entries"] == 2
        # The kept copy is not rewritten, so the only writes are canonical and the next event.
        assert [w["key"] for w in niamey["writes"]] == [
            "climate/normals/niamey.parquet",
            "climate/normals/events/zones/niamey/000003.json",
        ]

    def test_a_re_plan_after_the_promotion_keeps_the_pre_run_canonical_readback(self, env):
        """The runbook copies readback/{run}/canonical/ off the store as the only backup of the v1.
        The re-plan that follows a promotion (or resumes one that stopped between its PUT and its
        event) reads this run's v2 in canonical: it adds that file, named by its sha256, and never
        replaces the v1 one, whose name and bytes still match the first plan."""
        _prepare(env, "fos_sur_mer")
        first = json.loads((env.root / op._PLAN.format(run_id="r1")).read_text())
        v1 = env.seed["fos_sur_mer"]["v1"]
        v1_sha = first["zones"]["fos_sur_mer"]["canonical"]["sha256"]
        readback = env.root / "readback" / "r1" / "canonical"
        assert v1_sha == hashlib.sha256(v1).hexdigest()
        assert (readback / f"fos_sur_mer.{v1_sha}.parquet").read_bytes() == v1
        assert _run(env, "promote", "fos_sur_mer") == 0

        assert _run(env, "plan", "fos_sur_mer") == 0

        replanned = json.loads((env.root / op._PLAN.format(run_id="r1")).read_text())
        v2_sha = replanned["zones"]["fos_sur_mer"]["canonical"]["sha256"]
        assert v2_sha == hashlib.sha256(_staged(env, "fos_sur_mer")).hexdigest()
        assert replanned["zones"]["fos_sur_mer"]["canonical"]["schema_version"] == 2
        files = {p.name: p.read_bytes() for p in readback.iterdir() if not p.name.startswith(".")}
        assert files == {
            f"fos_sur_mer.{v1_sha}.parquet": v1,
            f"fos_sur_mer.{v2_sha}.parquet": _staged(env, "fos_sur_mer"),
        }

    def test_plan_refuses_a_published_run_record_made_under_another_authorization(
        self, env, capsys
    ):
        """Promote re-checks the published run record's authority; the plan-side call is its own
        guard and is reached before promote's, so it needs its own test."""
        for step in ("fetch", "materialize", "publish"):
            assert _run(env, step, "niamey") == 0
        argv = _with(_argv(env, "plan", "niamey"), "--authorization", f"{AUTHORIZATION}2")

        assert op.main(argv) == 2
        assert "published run record for run 'r1' was made under another authorization" in (
            capsys.readouterr().err
        )
        assert not (env.root / op._PLAN.format(run_id="r1")).exists()

    def test_the_journal_ceiling_is_refused_rather_than_silently_truncated(
        self, env, monkeypatch, capsys
    ):
        """`_journal` walks the store by key; with the ceiling lowered, a journal that fills every
        slot it looked at must be refused as not one this operator wrote, not read back as if it
        simply stopped there. Every field a consumer of the entry reads is filled in, so a passing
        plan under the mutant is a real "silently truncated", not an accidental crash."""
        monkeypatch.setattr(op, "_JOURNAL_CEILING", 2)
        for step in ("fetch", "materialize", "publish"):
            assert _run(env, step, "niamey") == 0
        for seq in (1, 2):
            key = op._ZONE_EVENT.format(zone_id="niamey", seq=seq)
            entry = {
                "kind": "promote",
                "run_id": "someone-else",
                "seq": seq,
                "at": "2020-01-01T00:00:00+00:00",
            }
            env.fake.objects[key] = json.dumps(entry).encode()

        assert _run(env, "plan", "niamey") == 2
        assert "exceeds its ceiling" in capsys.readouterr().err

    def test_plan_fails_when_the_percentile_comparisons_own_validate_dst_refuses(
        self, env, monkeypatch, capsys
    ):
        """`_percentile_comparison` validates both readback paths (`_validate_dst`, the same guard
        `era5_source` gets) before DuckDB ever opens them. Neither readback path can be made to
        fail that guard through this fake store — both sit under LOCAL_PARQUET_SOURCE_ROOT by
        construction — so the guard itself is replaced with one that always refuses. Its module
        binding here is used ONLY by `_percentile_comparison`: `_normals_coverage`, called just
        before it in the same loop, goes through its own copy imported in `src/tools/climate`, so
        this does not also mask that earlier call."""
        for step in ("fetch", "materialize", "publish"):
            assert _run(env, step, "niamey") == 0

        def _refuse(dst):
            raise ValueError(f"unsafe parquet destination: {dst!r}")

        monkeypatch.setattr(op, "_validate_dst", _refuse)

        assert _run(env, "plan", "niamey") == 1
        assert "unsafe parquet destination" in capsys.readouterr().err


class TestTheWholeChainOnAFakeStore:
    def test_fetch_materialize_publish_plan_promote_verify_rollback_verify(self, env, capsys):
        assert _run(env, "fetch", *ZONES) == 0
        calls = len(env.fake.calls)
        assert _run(env, "materialize", *ZONES) == 0
        assert len(env.fake.calls) == calls, "materialize touched the store"
        assert _run(env, "publish", *ZONES) == 0
        assert _run(env, "plan", *ZONES) == 0
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 0

        for zone_id in ZONES:
            assert _canonical(env, zone_id) == _staged(env, zone_id)
            kept = env.fake.objects[op._ROLLBACK.format(run_id="r1", zone_id=zone_id)]
            assert kept == env.seed[zone_id]["v1"]
            assert [(e["kind"], e["run_id"]) for e in _journal(env, zone_id)] == [("promote", "r1")]
        run_events = [k for k in env.fake.objects if k.startswith("climate/normals/events/runs/")]
        assert run_events == ["climate/normals/events/runs/r1/000001.json"]
        capsys.readouterr()

        assert _run(env, "verify", *ZONES) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["endpoint"] == ENDPOINT
        for zone_id in ZONES:
            assert report["zones"][zone_id]["schema_versions"] == [2]
            assert report["zones"][zone_id]["rows"] == 72
            assert report["zones"][zone_id]["wet_day_columns"] is True
            assert report["zones"][zone_id]["checks"] == {
                "sha256": True,
                "readable": True,
                "schema_version": True,
                "rows": True,
                "wet_day_columns": True,
            }

        assert _run(env, "rollback", *ZONES) == 0

        for zone_id in ZONES:
            assert _canonical(env, zone_id) == env.seed[zone_id]["v1"]
            # The staged v2 is untouched, so the rollback can itself be undone.
            assert _staged(env, zone_id)
        capsys.readouterr()
        assert _run(env, "verify", *ZONES) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["zones"]["niamey"]["state"] == "rollback"
        assert report["zones"]["niamey"]["schema_versions"] == [1]
        assert report["zones"]["niamey"]["checks"] == {"sha256": True, "readable": True}
        assert [(e["kind"], sorted(e["zones"])) for e in _run_events(env)] == [
            ("promoted", sorted(ZONES)),
            ("rolled_back", sorted(ZONES)),
        ]
        rolled_back = _run_events(env)[-1]["zones"]["niamey"]
        assert rolled_back == {
            "seq": 2,
            "kind": "rollback",
            "canonical_sha256_after": hashlib.sha256(env.seed["niamey"]["v1"]).hexdigest(),
        }

        # Every write stayed under the run's own prefixes or canonical; nothing is deletable.
        allowed = (
            "climate/normals/staging/r1/",
            "climate/normals/rollback/r1/",
            "climate/normals/events/",
            *(op._CANONICAL.format(zone_id=z) for z in ZONES),
        )
        assert all(key.startswith(allowed) for key in env.fake.puts())
        # A catalog rebuild over climate/ must not mistake the run record for a dataset manifest.
        assert not [key for key in env.fake.objects if MCPS_MANIFEST.match(key)]
        record = json.loads(env.fake.objects[op._RUN_RECORD.format(run_id="r1")])
        assert record["authorization"] == AUTHORIZATION
        assert record["bucket"] == BUCKET
        assert record["endpoint"] == ENDPOINT
        assert record["code_ref"]["commit"] == COMMIT
        niamey = next(z for z in record["zones"] if z["zone_id"] == "niamey")
        assert niamey["era5"]["sha256"] == hashlib.sha256(env.seed["niamey"]["era5"]).hexdigest()
        assert niamey["staged_sha256"] == hashlib.sha256(_staged(env, "niamey")).hexdigest()

    def test_verify_fails_on_a_run_that_does_not_own_canonical(self, env, capsys):
        _prepare(env, "niamey")
        capsys.readouterr()

        assert _run(env, "verify", "niamey") == 1
        assert json.loads(capsys.readouterr().out)["ok"] is False

    def test_verify_refuses_a_subset_of_the_runs_zones(self, env, capsys):
        """A verify over part of the run's zones must not pass as a whole-run verify: the zones
        it was not given are simply never checked, and their state could be anything."""
        _prepare(env, *ZONES)
        assert _run(env, "promote", *ZONES) == 0

        assert _run(env, "verify", "niamey") == 2

    def test_verify_does_not_take_another_runs_latest_rollback_as_this_runs_state(self, env):
        """r1 promotes and rolls back fos_sur_mer; canonical then changes out of band to bytes r1
        staged but never wrote through its own promotion, and r2 promotes and rolls that back too.
        r1's verify must not read r2's rollback event as r1's own state just because it is the
        journal's latest entry."""
        zone = "fos_sur_mer"
        _prepare(env, zone)
        assert _run(env, "promote", zone) == 0
        assert _run(env, "rollback", zone) == 0
        env.fake.objects[op._CANONICAL.format(zone_id=zone)] = env.fake.objects[
            op._STAGING.format(run_id="r1", zone_id=zone)
        ]
        _prepare(env, zone, run_id="r2")
        assert _run(env, "promote", zone, run_id="r2") == 0
        assert _run(env, "rollback", zone, run_id="r2") == 0

        assert _run(env, "verify", zone) == 1

    def test_verify_fails_on_v1_bytes_where_the_promotion_should_be(self, env, capsys):
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        env.fake.objects[op._CANONICAL.format(zone_id="niamey")] = env.seed["niamey"]["v1"]
        capsys.readouterr()

        assert _run(env, "verify", "niamey") == 1

        report = json.loads(capsys.readouterr().out)
        assert report["ok"] is False
        assert report["zones"]["niamey"]["checks"] == {
            "sha256": False,
            "readable": True,
            "schema_version": False,
            "rows": True,
            "wet_day_columns": False,
        }

    def test_verify_fails_on_a_v2_with_missing_rows(self, env, capsys, tmp_path):
        import duckdb

        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        full, short = tmp_path / "full.parquet", tmp_path / "short.parquet"
        full.write_bytes(_staged(env, "niamey"))
        con = duckdb.connect(":memory:")
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{full}') WHERE month <> 12) TO '{short}' "
            "(FORMAT PARQUET)"
        )
        con.close()
        env.fake.objects[op._CANONICAL.format(zone_id="niamey")] = short.read_bytes()
        capsys.readouterr()

        assert _run(env, "verify", "niamey") == 1

        assert json.loads(capsys.readouterr().out)["zones"]["niamey"]["checks"] == {
            "sha256": False,
            "readable": True,
            "schema_version": True,
            "rows": False,
            "wet_day_columns": True,
        }

    @pytest.mark.parametrize("after", ["promote", "rollback"])
    def test_verify_fails_when_climate_normals_cannot_read_canonical(
        self, env, monkeypatch, capsys, after
    ):
        """In every state, not only after a promote: the sha256 alone is a boto3 read, and the
        serving path is DuckDB httpfs. Its error is redacted."""
        monkeypatch.setattr(settings, "S3_SECRET_KEY", "sup3r-s3cret-value")
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        if after == "rollback":
            assert _run(env, "rollback", "niamey") == 0

        def _forbidden(*_a, **_k):
            raise RuntimeError("IO Error: HTTP 403 for sup3r-s3cret-value")

        monkeypatch.setattr(op, "_read_normals_rows", _forbidden)
        capsys.readouterr()

        assert _run(env, "verify", "niamey") == 1

        out = capsys.readouterr().out
        zone = json.loads(out)["zones"]["niamey"]
        assert zone["checks"]["sha256"] is True
        assert zone["checks"]["readable"] is False
        assert "IO Error: HTTP 403" in zone["read_error"]
        assert "sup3r-s3cret-value" not in out

    def test_if_match_is_sent_only_when_asked_for(self, env):
        _prepare(env, "niamey")
        planned = json.loads((env.root / op._PLAN.format(run_id="r1")).read_text())
        key = op._CANONICAL.format(zone_id="niamey")

        assert _run(env, "promote", "niamey") == 0
        assert _run(env, "rollback", "niamey") == 0
        assert _run(env, "plan", "niamey") == 0
        replanned = json.loads((env.root / op._PLAN.format(run_id="r1")).read_text())
        assert all("IfMatch" not in r for r in env.fake.put_requests_for(key))
        env.fake.put_requests.clear()

        assert _run(env, "promote", "niamey", extra=("--if-match",)) == 0
        assert _run(env, "rollback", "niamey", extra=("--if-match",)) == 0

        promote_put, rollback_put = env.fake.put_requests_for(key)
        assert promote_put["IfMatch"] == replanned["zones"]["niamey"]["canonical"]["etag"]
        assert promote_put["IfMatch"] == planned["zones"]["niamey"]["canonical"]["etag"]
        # The rollback's precondition is the etag of the promoted bytes it verified.
        assert rollback_put["IfMatch"] == _etag(_staged(env, "niamey"))


class TestPublishIsCreateOnly:
    @CONFLICTS
    def test_a_republish_of_identical_bytes_is_accepted(self, env, capsys, conflict):
        env.fake.conflict = conflict
        for step in ("fetch", "materialize", "publish"):
            assert _run(env, step, "niamey") == 0

        assert _run(env, "publish", "niamey") == 0
        assert "identical" in capsys.readouterr().out

    @CONFLICTS
    def test_other_bytes_at_a_published_key_are_refused_and_kept(self, env, capsys, conflict):
        env.fake.conflict = conflict
        for step in ("fetch", "materialize"):
            assert _run(env, step, "niamey") == 0
        key = op._STAGING.format(run_id="r1", zone_id="niamey")
        env.fake.objects[key] = b"somebody else's upload"

        assert _run(env, "publish", "niamey") == 2
        assert "already exists with different bytes" in capsys.readouterr().err
        assert env.fake.objects[key] == b"somebody else's upload"

    @pytest.mark.parametrize("which", ["staged", "record"])
    def test_a_fresh_upload_that_reads_back_with_other_bytes_is_refused(self, env, capsys, which):
        for step in ("fetch", "materialize"):
            assert _run(env, step, "niamey") == 0
        key = (
            op._STAGING.format(run_id="r1", zone_id="niamey")
            if which == "staged"
            else op._RUN_RECORD.format(run_id="r1")
        )
        env.fake.on_put = lambda k, body: body[:-1] if k == key else body

        assert _run(env, "publish", "niamey") == 2
        assert "reads back with a different sha256 than was uploaded" in capsys.readouterr().err

    def test_an_upload_that_is_not_readable_back_at_all_is_refused(self, env, capsys):
        """Only a readback with OTHER bytes is exercised above; a readback that finds no object
        at all must be refused the same way, not treated as if the upload never happened."""
        for step in ("fetch", "materialize"):
            assert _run(env, step, "niamey") == 0
        key = op._STAGING.format(run_id="r1", zone_id="niamey")
        real_get = env.fake.get_object

        def _get(**request):
            if request["Key"] == key:
                raise _StoreError(404, "NoSuchKey")
            return real_get(**request)

        env.fake.get_object = _get

        assert _run(env, "publish", "niamey") == 2
        assert "not readable back" in capsys.readouterr().err

    def test_a_staged_file_changed_since_materialize_is_not_published(self, env, capsys):
        for step in ("fetch", "materialize"):
            assert _run(env, step, "niamey") == 0
        local = env.root / op._STAGING.format(run_id="r1", zone_id="niamey")
        local.write_bytes(local.read_bytes() + b"\0")

        assert _run(env, "publish", "niamey") == 2
        assert "differs from the sha256 the run record holds" in capsys.readouterr().err
        assert env.fake.puts() == []

    def test_publish_names_exactly_the_zones_materialized(self, env, capsys):
        for step in ("fetch", "materialize"):
            assert _run(env, step, *ZONES) == 0

        assert _run(env, "publish", "niamey") == 2
        assert "Name exactly the run's zones" in capsys.readouterr().err
        assert env.fake.puts() == []

    def test_a_run_record_made_under_another_endpoint_and_authorization_is_not_published(
        self, env, monkeypatch, capsys
    ):
        """A work root reused from the rehearsal would put a run record naming the harness and
        its authorization into the mesh bucket."""
        for step in ("fetch", "materialize"):
            assert _run(env, step, "niamey") == 0
        monkeypatch.setattr(settings, "S3_ENDPOINT", "http://storage-rustfs:9000")
        argv = _with(_argv(env, "publish", "niamey"), "--confirm-endpoint", "storage-rustfs:9000")
        argv = _with(argv, "--authorization", f"{AUTHORIZATION}2")

        assert op.main(argv) == 2
        assert "made under another endpoint, authorization" in capsys.readouterr().err
        assert env.fake.puts() == []

    def test_a_run_record_made_under_another_bucket_is_not_published(
        self, env, monkeypatch, capsys
    ):
        """The bucket is part of the authority a record is bound to, same as endpoint and
        authorization above — isolated here: only the bucket differs."""
        for step in ("fetch", "materialize"):
            assert _run(env, step, "niamey") == 0
        monkeypatch.setattr(settings, "S3_BUCKET", "deposium-b")
        argv = _with(_argv(env, "publish", "niamey"), "--confirm-bucket", "deposium-b")
        calls = len(env.fake.calls)

        assert op.main(argv) == 2
        assert "another bucket" in capsys.readouterr().err
        assert len(env.fake.calls) == calls


class TestFetchAndMaterializeWorkOnlyFromWhatWasFetched:
    def test_fetch_refuses_a_zone_with_no_stored_era5(self, env, capsys):
        del env.fake.objects[op._ERA5.format(zone_id="niamey")]

        assert _run(env, "fetch", "niamey") == 2
        assert "no stored ERA5" in capsys.readouterr().err

    def test_materialize_refuses_an_era5_changed_since_fetch(self, env, capsys):
        assert _run(env, "fetch", "niamey") == 0
        local = env.root / op._ERA5.format(zone_id="niamey")
        local.write_bytes(local.read_bytes() + b"\0")

        assert _run(env, "materialize", "niamey") == 2
        assert "not the object fetch digested" in capsys.readouterr().err
        assert not (env.root / op._RUN_RECORD.format(run_id="r1")).exists()

    def test_materialize_names_exactly_the_zones_fetched(self, env, capsys):
        assert _run(env, "fetch", "niamey") == 0

        assert _run(env, "materialize", *ZONES) == 2
        assert "Name exactly the run's zones" in capsys.readouterr().err

    def test_materialize_refuses_a_fetch_made_under_another_endpoint(
        self, env, monkeypatch, capsys
    ):
        assert _run(env, "fetch", "niamey") == 0
        monkeypatch.setattr(settings, "S3_ENDPOINT", "http://storage-rustfs:9000")
        argv = _with(
            _argv(env, "materialize", "niamey"), "--confirm-endpoint", "storage-rustfs:9000"
        )

        assert op.main(argv) == 2
        assert "made under another endpoint" in capsys.readouterr().err

    def test_a_store_error_is_reported_redacted(self, env, monkeypatch, capsys):
        monkeypatch.setattr(settings, "S3_SECRET_KEY", "sup3r-s3cret-value")

        def _refused(**_request):
            raise RuntimeError("SignatureDoesNotMatch for sup3r-s3cret-value")

        env.fake.get_object = _refused

        assert _run(env, "fetch", "niamey") == 1
        err = capsys.readouterr().err
        assert "FAILED: RuntimeError: SignatureDoesNotMatch" in err
        assert "sup3r-s3cret-value" not in err


class TestPromoteRefusesWhatThePlanDidNotSee:
    def test_canonical_changed_between_plan_and_promote(self, env, capsys):
        _prepare(env, *ZONES)
        key = op._CANONICAL.format(zone_id="fos_sur_mer")
        env.fake.objects[key] = b"written by someone else after the plan"
        env.fake.calls.clear()

        assert _run(env, "promote", *ZONES) == 2

        assert "changed since plan" in capsys.readouterr().err
        # Refused while validating, before ANY zone was written.
        assert env.fake.puts() == []
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        assert env.fake.objects[key] == b"written by someone else after the plan"

    def test_a_partial_promotion_is_refused_before_any_write(self, env, capsys):
        _prepare(env, *ZONES)
        env.fake.calls.clear()

        assert _run(env, "promote", "niamey") == 2
        assert "Promote the run the manifest describes" in capsys.readouterr().err
        assert env.fake.puts() == []

    def test_canonical_changed_after_its_rollback_copy_was_written_is_refused(self, env, capsys):
        """The TOCTOU window step 2 exists for: canonical moves while promote is writing, after
        the pre-validation HEAD. The other writer's object is kept, and niamey, already promoted
        in this invocation, is restored."""
        _prepare(env, *ZONES)
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")
        fos_kept = op._ROLLBACK.format(run_id="r1", zone_id="fos_sur_mer")

        def _race(key, body):
            if key == fos_kept:
                env.fake.objects[fos] = FOREIGN
            return body

        env.fake.on_put = _race

        assert _run(env, "promote", *ZONES) == 2

        err = capsys.readouterr().err
        assert "changed since plan" in err
        assert "after this run's rollback copy for it was written" in err
        assert env.fake.objects[fos] == FOREIGN
        assert fos not in env.fake.puts()
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    def test_canonical_changed_while_an_earlier_zone_was_promoted_refuses_before_any_write(
        self, env, capsys
    ):
        _prepare(env, *ZONES)
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")
        niamey = op._CANONICAL.format(zone_id="niamey")

        def _race(key, body):
            if key == niamey:
                env.fake.objects[fos] = FOREIGN
            return body

        env.fake.on_put = _race

        assert _run(env, "promote", *ZONES) == 2

        assert "before anything was written for it" in capsys.readouterr().err
        assert env.fake.objects[fos] == FOREIGN
        # Refused before fos's rollback copy, so a re-plan can still promote the run.
        assert op._ROLLBACK.format(run_id="r1", zone_id="fos_sur_mer") not in env.fake.objects
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    def test_a_change_caught_after_the_rollback_copy_needs_a_new_run_id(self, env, capsys):
        """Fail-closed, and what the runbook says: the kept copy describes the canonical the plan
        saw, so a re-plan of the same run id is refused rather than promoted over the change."""
        _prepare(env, "niamey")
        key = op._CANONICAL.format(zone_id="niamey")
        kept = op._ROLLBACK.format(run_id="r1", zone_id="niamey")

        def _race(k, body):
            if k == kept:
                env.fake.objects[key] = FOREIGN
            return body

        env.fake.on_put = _race
        assert _run(env, "promote", "niamey") == 2
        assert "materialize under a new run id" in capsys.readouterr().err
        env.fake.on_put = None

        assert _run(env, "plan", "niamey") == 0
        assert _run(env, "promote", "niamey") == 2
        assert "describes neither" in capsys.readouterr().err
        assert env.fake.objects[key] == FOREIGN

    @pytest.mark.parametrize("canonical", ["exists", "absent"])
    def test_a_rollback_copy_that_reads_back_wrong_stops_before_canonical(
        self, env, capsys, canonical
    ):
        """Create-only: a truncated kept copy can never be rewritten, and would only be found at
        rollback time, after v1 was overwritten."""
        key = op._CANONICAL.format(zone_id="niamey")
        if canonical == "absent":
            del env.fake.objects[key]
            kept = op._ABSENT.format(run_id="r1", zone_id="niamey")
        else:
            kept = op._ROLLBACK.format(run_id="r1", zone_id="niamey")
        _prepare(env, "niamey")
        env.fake.on_put = lambda k, body: body[:-1] if k == kept else body

        assert _run(env, "promote", "niamey") == 2

        assert "reads back with a different sha256" in capsys.readouterr().err
        assert key not in env.fake.puts()
        assert _canonical(env, "niamey") == (
            env.seed["niamey"]["v1"] if canonical == "exists" else None
        )

    def test_a_plan_for_another_run_record_is_refused(self, env, capsys):
        _prepare(env, "niamey")
        path = env.root / op._PLAN.format(run_id="r1")
        plan = json.loads(path.read_text())
        plan["run_record_sha256"] = "0" * 64
        path.write_text(json.dumps(plan))

        env.fake.calls.clear()

        assert _run(env, "promote", "niamey") == 2
        assert "made for another run record" in capsys.readouterr().err
        assert env.fake.puts() == []

    def test_a_plan_made_under_another_endpoint_does_not_drive_the_promote(
        self, env, monkeypatch, capsys
    ):
        _prepare(env, "niamey")
        path = env.root / op._PLAN.format(run_id="r1")
        plan = json.loads(path.read_text())
        plan["endpoint"] = "storage-rustfs:9000"
        path.write_text(json.dumps(plan))
        env.fake.calls.clear()

        assert _run(env, "promote", "niamey") == 2
        assert "the plan for run 'r1' was made under another endpoint" in capsys.readouterr().err
        assert env.fake.puts() == []

    def test_a_published_run_record_made_under_another_authorization_is_refused(self, env, capsys):
        _prepare(env, "niamey")
        env.fake.calls.clear()
        argv = _with(_argv(env, "promote", "niamey"), "--authorization", f"{AUTHORIZATION}2")

        assert op.main(argv) == 2
        assert "published run record for run 'r1' was made under another authorization" in (
            capsys.readouterr().err
        )
        assert env.fake.puts() == []

    def test_a_plan_whose_journal_moved_is_stale_and_refused(self, env, capsys):
        """After a promote and a rollback the original plan lists a rollback-copy PUT and event
        000001; the promote would write neither. The reviewed writes must be the ones made."""
        _prepare(env, "niamey")
        stale = (env.root / op._PLAN.format(run_id="r1")).read_text()
        assert _run(env, "promote", "niamey") == 0
        assert _run(env, "rollback", "niamey") == 0
        (env.root / op._PLAN.format(run_id="r1")).write_text(stale)
        env.fake.calls.clear()

        assert _run(env, "promote", "niamey") == 2

        assert "has 2 entries and the plan saw 0" in capsys.readouterr().err
        assert env.fake.puts() == []

    @pytest.mark.parametrize("record", ["missing", "another kind"])
    def test_promote_requires_the_published_run_record(self, env, capsys, record):
        _prepare(env, "niamey")
        key = op._RUN_RECORD.format(run_id="r1")
        if record == "missing":
            del env.fake.objects[key]
        else:
            env.fake.objects[key] = json.dumps({"kind": "dataset", "run_id": "r1"}).encode()

        assert _run(env, "promote", "niamey") == 2
        err = capsys.readouterr().err
        assert ("in the store" if record == "missing" else "does not describe that run") in err
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    def test_a_run_record_describing_v1_bytes_is_refused_by_the_v2_contract(self, env, capsys):
        """The record's own sha256 check passes when the record was built to match: the v2
        contract (`_validate_staged`) is the pre-check that remains."""
        for step in ("fetch", "materialize"):
            assert _run(env, step, "niamey") == 0
        v1 = env.seed["niamey"]["v1"]
        (env.root / op._STAGING.format(run_id="r1", zone_id="niamey")).write_bytes(v1)
        path = env.root / op._RUN_RECORD.format(run_id="r1")
        record = json.loads(path.read_text())
        record["zones"][0]["staged_sha256"] = hashlib.sha256(v1).hexdigest()
        path.write_text(json.dumps(record))
        for step in ("publish", "plan"):
            assert _run(env, step, "niamey") == 0
        capsys.readouterr()

        assert _run(env, "promote", "niamey") == 2

        assert "is schema v1" in capsys.readouterr().err
        assert op._CANONICAL.format(zone_id="niamey") not in env.fake.puts()

    def test_promote_without_a_plan_is_refused(self, env, capsys):
        for step in ("fetch", "materialize", "publish"):
            assert _run(env, step, "niamey") == 0

        assert _run(env, "promote", "niamey") == 2
        assert "run plan first" in capsys.readouterr().err

    def test_a_staged_object_that_is_not_the_recorded_one_is_refused(self, env, capsys):
        _prepare(env, "niamey")
        env.fake.objects[op._STAGING.format(run_id="r1", zone_id="niamey")] = b"swapped"

        assert _run(env, "promote", "niamey") == 2
        assert "the run record says" in capsys.readouterr().err
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    def test_an_existing_rollback_copy_that_describes_something_else_is_kept_and_refused(
        self, env, capsys
    ):
        _prepare(env, "niamey")
        kept = op._ROLLBACK.format(run_id="r1", zone_id="niamey")
        env.fake.objects[kept] = b"a copy of some other canonical"

        assert _run(env, "promote", "niamey") == 2

        assert "describes neither" in capsys.readouterr().err
        assert env.fake.objects[kept] == b"a copy of some other canonical"
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    def test_a_rollback_copy_left_by_an_earlier_promotion_is_kept_on_re_promotion(self, env):
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        assert _run(env, "rollback", "niamey") == 0
        assert _run(env, "plan", "niamey") == 0
        env.fake.calls.clear()

        assert _run(env, "promote", "niamey") == 0

        kept = op._ROLLBACK.format(run_id="r1", zone_id="niamey")
        assert kept not in env.fake.puts()
        assert env.fake.objects[kept] == env.seed["niamey"]["v1"]
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote", "rollback", "promote"]

    def test_a_run_rolled_back_off_the_top_of_a_stack_can_be_promoted_again(self, env, capsys):
        """r1's kept copy was taken while r0 owned canonical, so it IS r0's bytes: restoring it
        cannot undo r0's promotion, and the N4 refusal does not apply."""
        _prepare(env, "niamey", run_id="r0")
        assert _run(env, "promote", "niamey", run_id="r0") == 0
        _prepare(env, "niamey", run_id="r1")
        assert _run(env, "promote", "niamey", run_id="r1") == 0
        assert _run(env, "rollback", "niamey", run_id="r1") == 0
        assert _run(env, "plan", "niamey", run_id="r1") == 0
        capsys.readouterr()

        assert _run(env, "promote", "niamey", run_id="r1") == 0, capsys.readouterr().err

        assert [(e["kind"], e["run_id"]) for e in _journal(env, "niamey")] == [
            ("promote", "r0"),
            ("promote", "r1"),
            ("rollback", "r1"),
            ("promote", "r1"),
        ]
        assert _run(env, "rollback", "niamey", run_id="r1") == 0
        assert _run(env, "rollback", "niamey", run_id="r0") == 0
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    def test_a_run_cannot_re_promote_over_another_runs_promotion(self, env, capsys):
        """The promote half of N4: r1's kept copy is the v1 from before r2 promoted, so r1
        promoting on top and later rolling back would restore v1 and erase r2's promotion."""
        _prepare(env, "niamey", run_id="r1")
        assert _run(env, "promote", "niamey", run_id="r1") == 0
        assert _run(env, "rollback", "niamey", run_id="r1") == 0
        _prepare(env, "niamey", run_id="r2")
        assert _run(env, "promote", "niamey", run_id="r2") == 0
        assert _run(env, "plan", "niamey", run_id="r1") == 0
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", "niamey", run_id="r1") == 2

        assert "Roll the owning run back first" in capsys.readouterr().err
        assert env.fake.puts() == []


class TestAFailedPromotionUnwindsWithoutDeleting:
    def test_a_readback_mismatch_restores_every_touched_zone(self, env, capsys):
        _prepare(env, *ZONES)
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")

        def _corrupt(key, body):
            return (
                b"a truncated write"
                if key == fos and body != env.seed["fos_sur_mer"]["v1"]
                else body
            )

        env.fake.on_put = _corrupt

        assert _run(env, "promote", *ZONES) == 2

        err = capsys.readouterr().err
        assert "reads back with sha256" in err
        assert "No canonical object holds bytes this invocation wrote" in err
        for zone_id in ZONES:
            assert _canonical(env, zone_id) == env.seed[zone_id]["v1"]
        # niamey had been promoted and journaled; the unwind says so, so it owns nothing now.
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote", "unwind"]
        assert _journal(env, "fos_sur_mer") == []

        env.fake.on_put = None
        assert _run(env, "plan", *ZONES) == 0
        assert _run(env, "promote", *ZONES) == 0
        assert _canonical(env, "fos_sur_mer") == _staged(env, "fos_sur_mer")

    def test_a_put_that_landed_and_then_raised_is_unwound(self, env, capsys):
        """The zone is marked touched BEFORE the PUT: a write that lands and then times out must
        be restored like one that returned."""
        _prepare(env, *ZONES)
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")

        def _landed_then_timed_out(key, body):
            if key == fos and body != env.seed["fos_sur_mer"]["v1"]:
                env.fake.objects[key] = body
                raise ConnectionError("read timeout after the body was sent")
            return body

        env.fake.on_put = _landed_then_timed_out

        assert _run(env, "promote", *ZONES) == 1

        assert "restored to their bytes before this invocation: ['fos_sur_mer', 'niamey']" in (
            capsys.readouterr().err
        )
        for zone_id in ZONES:
            assert _canonical(env, zone_id) == env.seed[zone_id]["v1"]
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote", "unwind"]
        assert _journal(env, "fos_sur_mer") == []

    @CONFLICTS
    def test_a_refused_if_match_put_is_not_unwound_over_the_concurrent_write(
        self, env, capsys, conflict
    ):
        """--if-match: another writer lands between step 2's HEAD and the PUT. The refused PUT
        proves nothing of ours landed; restoring the pre-run bytes would erase that writer."""
        env.fake.conflict = conflict
        _prepare(env, "niamey")
        key = op._CANONICAL.format(zone_id="niamey")
        real_put = env.fake.put_object

        def _racing_put(**request):
            if request["Key"] == key and request.get("IfMatch"):
                env.fake.objects[key] = FOREIGN
            return real_put(**request)

        env.fake.put_object = _racing_put

        assert _run(env, "promote", "niamey", extra=("--if-match",)) == 2

        err = capsys.readouterr().err
        assert "refused the conditional PUT (If-Match)" in err
        assert "No canonical object holds bytes this invocation wrote" in err
        assert _canonical(env, "niamey") == FOREIGN
        assert [bool(r.get("IfMatch")) for r in env.fake.put_requests_for(key)] == [True]
        assert _journal(env, "niamey") == []

    @CONFLICTS
    def test_a_conflict_on_an_unconditional_put_proves_nothing_and_is_unwound(
        self, env, capsys, conflict
    ):
        """Without If-Match no precondition was sent, so a 409 or 412 cannot mean "nothing of
        ours landed": the zone stays touched and the unwind reads canonical to decide."""
        _prepare(env, "niamey")
        key = op._CANONICAL.format(zone_id="niamey")
        real_put = env.fake.put_object

        def _landed_then_conflict(**request):
            real_put(**request)
            if request["Key"] == key and request["Body"] != env.seed["niamey"]["v1"]:
                raise _StoreError(*conflict)

        env.fake.put_object = _landed_then_conflict

        assert _run(env, "promote", "niamey") == 1

        assert "restored to their bytes before this invocation: ['niamey']" in (
            capsys.readouterr().err
        )
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    @CONFLICTS
    def test_a_canonical_that_appeared_under_a_create_only_put_is_left_alone(
        self, env, capsys, conflict
    ):
        env.fake.conflict = conflict
        key = op._CANONICAL.format(zone_id="niamey")
        del env.fake.objects[key]
        _prepare(env, "niamey")
        real_put = env.fake.put_object

        def _appears(**request):
            if request["Key"] == key:
                env.fake.objects[key] = FOREIGN  # after step 2's HEAD, before the PUT
            return real_put(**request)

        env.fake.put_object = _appears

        assert _run(env, "promote", "niamey") == 2

        err = capsys.readouterr().err
        assert "refused the conditional PUT (If-None-Match)" in err
        assert "No canonical object holds bytes this invocation wrote" in err
        assert _canonical(env, "niamey") == FOREIGN
        assert _journal(env, "niamey") == []

    def test_a_write_that_landed_after_ours_is_named_and_not_restored(self, env, capsys):
        """Without --if-match the unwind used to restore every touched zone blind."""
        _prepare(env, *ZONES)
        niamey = op._CANONICAL.format(zone_id="niamey")
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")

        def _overwritten_then_fos_fails(key, body):
            if key == fos and body != env.seed["fos_sur_mer"]["v1"]:
                env.fake.objects[niamey] = FOREIGN
                raise ConnectionError("connection reset")
            return body

        env.fake.on_put = _overwritten_then_fos_fails

        assert _run(env, "promote", *ZONES) == 1

        err = capsys.readouterr().err
        assert "CHANGED BY ANOTHER WRITER after this run's PUT, so NOT restored" in err
        assert f"s3://{BUCKET}/{niamey}" in err
        assert "No canonical object holds bytes" not in err
        assert _canonical(env, "niamey") == FOREIGN
        assert [r.get("Body") for r in env.fake.put_requests_for(niamey)] == [
            _staged(env, "niamey")
        ]
        assert _canonical(env, "fos_sur_mer") == env.seed["fos_sur_mer"]["v1"]

    def test_the_unwind_restore_carries_if_match_on_the_etag_it_read(self, env):
        _prepare(env, *ZONES)
        niamey = op._CANONICAL.format(zone_id="niamey")
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")
        env.fake.on_put = lambda k, body: (
            b"a truncated write" if k == fos and body != env.seed["fos_sur_mer"]["v1"] else body
        )

        assert _run(env, "promote", *ZONES, extra=("--if-match",)) == 2

        restore = env.fake.put_requests_for(niamey)[-1]
        assert restore["Body"] == env.seed["niamey"]["v1"]
        assert restore["IfMatch"] == _etag(_staged(env, "niamey"))
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    def test_the_unwind_restore_refuses_a_write_between_its_read_and_its_put(self, env, capsys):
        """The restore's If-Match must be the etag of the GET that proved canonical still held
        this run's bytes — not a fresh HEAD taken right before the PUT, which would carry whatever
        a writer landed in between and let the restore silently overwrite it."""
        zone = "fos_sur_mer"
        _prepare(env, zone)
        key = op._CANONICAL.format(zone_id=zone)
        staged = env.fake.objects[op._STAGING.format(run_id="r1", zone_id=zone)]
        env.fake.on_put = lambda k, body: body[:-1] if k == key and body == staged else body
        real_get, seen = env.fake.get_object, {"after_put": 0}

        def _get(**request):
            response = real_get(**request)
            if request["Key"] == key and key in env.fake.puts():
                seen["after_put"] += 1
                if seen["after_put"] == 2:  # the unwind's read: a writer lands right after it
                    env.fake.objects[key] = FOREIGN
            return response

        env.fake.get_object = _get

        assert _run(env, "promote", zone, extra=("--if-match",)) == 2
        assert _canonical(env, zone) == FOREIGN
        # The reason names another writer's object sitting there, not a restore that landed
        # with wrong bytes.
        err = capsys.readouterr().err
        assert (
            "fos_sur_mer (the store refused the restore's If-Match and canonical does not hold "
            "the bytes it restores, so the restore did not land)"
        ) in err
        assert "reads back with other bytes" not in err

    def test_an_event_that_landed_but_did_not_read_back_is_unwound_in_the_journal(
        self, env, capsys
    ):
        """The unwind re-reads the journal: an in-memory "was the event written" said no, and the
        journal kept naming r1 as the owner of bytes canonical no longer held — a wedge neither
        rollback nor a re-promote could move."""
        _prepare(env, "niamey")
        event = op._ZONE_EVENT.format(zone_id="niamey", seq=1)
        real_get = env.fake.get_object
        failed = []

        def _get(**request):
            if request["Key"] == event and event in env.fake.objects and not failed:
                failed.append(event)
                raise ConnectionError("connection reset")
            return real_get(**request)

        env.fake.get_object = _get

        assert _run(env, "promote", "niamey") == 1

        env.fake.get_object = real_get
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote", "unwind"]
        assert _run(env, "plan", "niamey") == 0
        assert _run(env, "promote", "niamey") == 0
        assert _canonical(env, "niamey") == _staged(env, "niamey")

    def test_an_event_that_reads_back_with_other_bytes_is_refused(self, env, capsys):
        _prepare(env, "niamey")
        event = op._ZONE_EVENT.format(zone_id="niamey", seq=1)
        env.fake.on_put = lambda k, body: b"{}" if k == event else body

        assert _run(env, "promote", "niamey") == 2

        assert "does not read back as the event just written" in capsys.readouterr().err
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    def test_a_lost_journal_race_is_refused_and_unwound(self, env, capsys):
        _prepare(env, "niamey")
        event = op._ZONE_EVENT.format(zone_id="niamey", seq=1)
        canonical = op._CANONICAL.format(zone_id="niamey")
        other = json.dumps({"kind": "promote", "run_id": "other", "seq": 1}).encode()

        def _race(key, body):
            if key == canonical and body != env.seed["niamey"]["v1"]:
                env.fake.objects[event] = other
            return body

        env.fake.on_put = _race

        assert _run(env, "promote", "niamey") == 2

        assert "another writer appended to this journal" in capsys.readouterr().err
        assert env.fake.objects[event] == other
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    def test_an_unwind_that_fails_is_reported_zone_by_zone_and_redacted(
        self, env, monkeypatch, capsys
    ):
        """fos's restore reads back wrong; niamey is restored but its unwind event cannot be
        written, and that error quotes a secret."""
        monkeypatch.setattr(settings, "S3_SECRET_KEY", "sup3r-s3cret-value")
        _prepare(env, *ZONES)
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")

        def _corrupt(key, body):
            if key != fos:
                return body
            return b"a truncated restore" if body == env.seed["fos_sur_mer"]["v1"] else b"cut"

        env.fake.on_put = _corrupt
        real_append = op._append_event

        def _leaky_append(store, template, entries, event, **fields):
            if event["kind"] == "unwind":
                raise RuntimeError("journal write refused for sup3r-s3cret-value")
            return real_append(store, template, entries, event, **fields)

        monkeypatch.setattr(op, "_append_event", _leaky_append)

        assert _run(env, "promote", *ZONES) == 2

        err = capsys.readouterr().err
        assert "THE UNWIND FAILED for these" in err
        assert "fos_sur_mer (the restored canonical reads back with other bytes)" in err
        assert "niamey (journal write refused for" in err
        assert "sup3r-s3cret-value" not in err
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        assert _canonical(env, "fos_sur_mer") == b"a truncated restore"

    def test_a_resumed_promotion_skips_the_zones_already_done(self, env, monkeypatch, capsys):
        """The process died after niamey: no unwind ran, and the same plan is resumed."""
        _prepare(env, *ZONES)
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")

        def _die_on_fos(key, body):
            if key == fos:
                raise ConnectionError("connection reset")
            return body

        env.fake.on_put = _die_on_fos
        monkeypatch.setattr(op, "_unwind", lambda *a, **k: None)
        assert _run(env, "promote", *ZONES) == 1
        assert _canonical(env, "niamey") == _staged(env, "niamey")
        assert _canonical(env, "fos_sur_mer") == env.seed["fos_sur_mer"]["v1"]
        env.fake.on_put = None
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 0

        assert "1 already promoted by this run" in capsys.readouterr().out
        assert op._CANONICAL.format(zone_id="niamey") not in env.fake.puts()
        assert _canonical(env, "fos_sur_mer") == _staged(env, "fos_sur_mer")
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote"]

    def test_a_promotion_that_stopped_between_the_put_and_its_event_resumes_after_a_re_plan(
        self, env, capsys
    ):
        """Canonical already holds the staged v2 and no event says so. The old plan's etag has
        moved, so promote refuses; a re-plan sees the kept copy, and the rollback that follows
        still restores the v1 — not the staged bytes canonical held when the resume began."""
        _prepare(env, "niamey")
        journal = "climate/normals/events/zones/niamey/"

        def _die_on_the_event(key, body):
            if key.startswith(journal):
                raise ConnectionError("connection reset")
            return body

        env.fake.on_put = _die_on_the_event
        with pytest.MonkeyPatch.context() as died:
            died.setattr(op, "_unwind", lambda *a, **k: None)
            assert _run(env, "promote", "niamey") == 1
        assert _canonical(env, "niamey") == _staged(env, "niamey")
        assert _journal(env, "niamey") == []
        env.fake.on_put = None
        capsys.readouterr()

        assert _run(env, "promote", "niamey") == 2
        assert "changed since plan" in capsys.readouterr().err

        assert _run(env, "plan", "niamey") == 0
        assert _run(env, "promote", "niamey") == 0
        assert _run(env, "rollback", "niamey") == 0
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    def test_a_resumed_zone_that_fails_is_unwound_to_its_kept_copy(self, env, capsys):
        """The promotion stopped between fos_sur_mer's PUT and its event, so the resume found the
        staged v2 already in canonical. When the resume then failed after fos's event, the unwind
        "restored" that v2 and released the journal: canonical served this run's v2 with no run
        owning it, rollback restored only niamey and exited 0, and verify passed. A resumed zone
        is unwound to the pre-run bytes its kept copy holds."""
        _prepare(env, *ZONES)
        _killed_on_the_zone_event(env, "fos_sur_mer", *ZONES)
        assert _canonical(env, "fos_sur_mer") == _staged(env, "fos_sur_mer")
        assert _journal(env, "fos_sur_mer") == []
        assert _run(env, "plan", *ZONES) == 0
        _fail_once_on_read(env, op._ZONE_EVENT.format(zone_id="fos_sur_mer", seq=1))
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 1

        v1 = env.seed["fos_sur_mer"]["v1"]
        assert _canonical(env, "fos_sur_mer") == v1
        unwound = _journal(env, "fos_sur_mer")
        assert [e["kind"] for e in unwound] == ["promote", "unwind"]
        assert unwound[-1]["canonical_sha256_after"] == hashlib.sha256(v1).hexdigest()
        err = capsys.readouterr().err
        assert "restored to their bytes before this invocation: none" in err
        assert "Restored to this run's kept copy" in err
        assert "between its PUT and its event): ['fos_sur_mer']" in err
        assert _run(env, "rollback", *ZONES) == 0
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        assert _canonical(env, "fos_sur_mer") == v1
        assert _run(env, "verify", *ZONES) == 0

    def test_a_resumed_zone_that_fails_hands_canonical_back_to_the_run_below(
        self, env, tmp_path, monkeypatch
    ):
        """Stacked over r0, whose staged bytes differ: the unwind of r1's resume handed the stack
        back to r0 while canonical held r1's v2, and r0's rollback was then refused as written
        outside the protocol. The unwind returns r0's bytes, so r0 can still roll back.

        r0's values differ from r1's, which the EXACT parity rule refuses before r1 writes
        anything (TestTheParityRuleIsExact). The unwind this pins is the one for canonical bytes
        that are not r1's, whatever put them there, so the rule is lifted here and only here."""
        monkeypatch.setattr(op, "_require_parity", lambda *_a, **_k: None)
        zone = "fos_sur_mer"
        era5 = op._ERA5.format(zone_id=zone)
        wet = env.fake.objects[era5]
        env.fake.objects[era5] = _write_era5_year(tmp_path / "dry", zone, wet=False).read_bytes()
        _prepare(env, zone, run_id="r0")
        assert _run(env, "promote", zone, run_id="r0") == 0
        promoted_by_r0 = _canonical(env, zone)
        env.fake.objects[era5] = wet
        _prepare(env, zone)
        assert _staged(env, zone) != promoted_by_r0
        _killed_on_the_zone_event(env, zone, zone)
        assert _run(env, "plan", zone) == 0
        _fail_once_on_read(env, op._ZONE_EVENT.format(zone_id=zone, seq=2))

        assert _run(env, "promote", zone) == 1

        assert _canonical(env, zone) == promoted_by_r0
        assert [(e["kind"], e["run_id"]) for e in _journal(env, zone)] == [
            ("promote", "r0"),
            ("promote", "r1"),
            ("unwind", "r1"),
        ]
        assert _run(env, "rollback", zone, run_id="r0") == 0
        assert _canonical(env, zone) == env.seed[zone]["v1"]

    def test_a_resumed_zone_that_had_no_canonical_is_left_in_place_and_keeps_its_owner(
        self, env, capsys
    ):
        """Only the .absent marker describes the pre-run state, and this operator does not delete:
        the unwind names the zone for removal by hand, and its journal keeps naming this run as
        the owner of the bytes it serves, so rollback gives its own "does not delete" refusal."""
        zone = "fos_sur_mer"
        del env.fake.objects[op._CANONICAL.format(zone_id=zone)]
        _prepare(env, zone)
        _killed_on_the_zone_event(env, zone, zone)
        assert _run(env, "plan", zone) == 0
        _fail_once_on_read(env, op._ZONE_EVENT.format(zone_id=zone, seq=1))
        capsys.readouterr()

        assert _run(env, "promote", zone) == 1

        assert [(e["kind"], e["run_id"]) for e in _journal(env, zone)] == [("promote", "r1")]
        assert _canonical(env, zone) == _staged(env, zone)
        assert "remove them yourself" in capsys.readouterr().err
        assert _run(env, "rollback", zone) == 2
        assert "does not delete" in capsys.readouterr().err

    def test_a_resumed_zone_overwritten_since_is_refused(self, env, monkeypatch, capsys):
        _prepare(env, *ZONES)
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")

        def _die_on_fos(key, body):
            if key == fos:
                raise ConnectionError("connection reset")
            return body

        env.fake.on_put = _die_on_fos
        monkeypatch.setattr(op, "_unwind", lambda *a, **k: None)
        assert _run(env, "promote", *ZONES) == 1
        env.fake.on_put = None
        env.fake.objects[op._CANONICAL.format(zone_id="niamey")] = FOREIGN
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 2

        assert "no longer holds its staged bytes" in capsys.readouterr().err
        assert env.fake.puts() == []

    def test_a_resumed_zone_whose_canonical_was_removed_since_is_refused(self, env, capsys):
        """niamey was promoted by r1 with a kept copy (its v1), and its canonical was then removed
        out of band. The check for this run's kept copy must not index a missing object: promote
        refuses and does not crash."""
        _prepare(env, *ZONES)
        _killed_on_the_zone_event(env, "fos_sur_mer", *ZONES)
        assert _journal(env, "niamey")[-1]["rollback_sha256"]  # a kept copy exists
        del env.fake.objects[op._CANONICAL.format(zone_id="niamey")]
        assert _run(env, "plan", *ZONES) == 0
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 2

        err = capsys.readouterr().err
        assert "no longer holds its staged bytes" in err
        assert "TypeError" not in err
        assert env.fake.puts() == []
        assert _canonical(env, "niamey") is None

    def test_an_absent_marker_with_a_canonical_that_appeared_since_is_refused(self, env, capsys):
        key = op._CANONICAL.format(zone_id="niamey")
        del env.fake.objects[key]
        _prepare(env, "niamey")

        def _die(k, body):
            if k == key:
                raise ConnectionError("connection reset")
            return body

        env.fake.on_put = _die
        assert _run(env, "promote", "niamey") == 1
        assert op._ABSENT.format(run_id="r1", zone_id="niamey") in env.fake.objects
        env.fake.on_put = None
        env.fake.objects[key] = env.seed["niamey"]["v1"]
        assert _run(env, "plan", "niamey") == 0
        capsys.readouterr()

        assert _run(env, "promote", "niamey") == 2

        assert "absent marker says there was no canonical" in capsys.readouterr().err
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    def test_promoting_a_promoted_run_again_is_refused(self, env, capsys):
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0

        assert _run(env, "promote", "niamey") == 2
        assert "already promoted" in capsys.readouterr().err

    def test_a_zone_another_writer_changed_is_recovered_through_its_kept_copy(self, env, capsys):
        """The runbook's recovery for a zone the unwind named: the journal still names this run
        as owner, rollback refuses the foreign bytes, and once canonical is put back to the kept
        copy by hand, rollback writes only the zone's event (Greptile P1, PR #224)."""
        _prepare(env, *ZONES)
        first, second = ZONES
        first_key = op._CANONICAL.format(zone_id=first)

        def _other_writer_then_fail(key, body):
            if key == op._ROLLBACK.format(run_id="r1", zone_id=second):
                env.fake.objects[first_key] = FOREIGN
                raise ConnectionError("connection reset")
            return body

        env.fake.on_put = _other_writer_then_fail
        assert _run(env, "promote", *ZONES) == 1
        env.fake.on_put = None
        assert "CHANGED BY ANOTHER WRITER" in capsys.readouterr().err
        assert _canonical(env, first) == FOREIGN
        assert [e["kind"] for e in _journal(env, first)] == ["promote"]

        assert _run(env, "rollback", *ZONES) == 2
        assert "outside this protocol" in capsys.readouterr().err

        env.fake.objects[first_key] = env.fake.objects[
            op._ROLLBACK.format(run_id="r1", zone_id=first)
        ]
        assert _run(env, "rollback", *ZONES) == 0

        assert _canonical(env, first) == env.seed[first]["v1"]
        assert _journal(env, first)[-1]["canonical_written"] is False

    def test_a_stack_of_identical_bytes_is_unwound_not_left_for_removal(self, env, capsys):
        """r0 promoted the bytes r1 stages. r1 has no kept copy and no marker, so canonical already
        holding its staged bytes is not a resume: the unwind restores r0's bytes and releases r1,
        and does not call niamey a zone that had no canonical."""
        _prepare(env, "niamey", run_id="r0")
        assert _run(env, "promote", "niamey", run_id="r0") == 0
        _prepare(env, "niamey", run_id="r1")
        assert _staged(env, "niamey", "r1") == _canonical(env, "niamey")
        _fail_once_on_read(env, op._ZONE_EVENT.format(zone_id="niamey", seq=2))
        capsys.readouterr()

        assert _run(env, "promote", "niamey", run_id="r1") == 1

        err = capsys.readouterr().err
        assert "remove them yourself" not in err
        assert "restored to their bytes before this invocation: ['niamey']" in err
        assert [(e["kind"], e["run_id"]) for e in _journal(env, "niamey")] == [
            ("promote", "r0"),
            ("promote", "r1"),
            ("unwind", "r1"),
        ]
        assert _run(env, "rollback", "niamey", run_id="r0") == 0
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    def test_a_second_cycle_failure_is_not_reported_as_a_resume(self, env, capsys):
        """After a promote and a rollback, the kept copy exists and canonical holds it again. A
        second-cycle promote that fails reading back its own event is unwound to the bytes before
        this invocation, and is not reported as a resume of a promotion that stopped between its
        PUT and its event."""
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        assert _run(env, "rollback", "niamey") == 0
        assert _run(env, "plan", "niamey") == 0
        _fail_once_on_read(env, op._ZONE_EVENT.format(zone_id="niamey", seq=3))
        capsys.readouterr()

        assert _run(env, "promote", "niamey") == 1

        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        assert [e["kind"] for e in _journal(env, "niamey")] == [
            "promote",
            "rollback",
            "promote",
            "unwind",
        ]
        err = capsys.readouterr().err
        assert "restored to their bytes before this invocation: ['niamey']" in err
        assert "Restored to this run's kept copy" not in err

    def test_a_zone_the_unwind_restored_before_its_event_is_named_for_rollback(self, env, capsys):
        """Killed right after the unwind's restore of niamey landed and before its unwind event:
        canonical holds the kept copy and the journal still names r1. Promote used to call that
        written outside this protocol, with no other writer; it names the rollback that writes
        the missing event."""
        _prepare(env, *ZONES)
        niamey = op._CANONICAL.format(zone_id="niamey")
        _fail_once_on_read(env, op._ZONE_EVENT.format(zone_id="fos_sur_mer", seq=1))
        real_put = env.fake.put_object

        def _put(**request):
            real_put(**request)
            if request["Key"] == niamey and request["Body"] == env.seed["niamey"]["v1"]:
                raise KeyboardInterrupt  # killed once the unwind's restore landed

        env.fake.put_object = _put
        with pytest.raises(KeyboardInterrupt):
            _run(env, "promote", *ZONES)
        del env.fake.put_object
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote"]
        assert [e["kind"] for e in _journal(env, "fos_sur_mer")] == ["promote", "unwind"]
        assert _run(env, "plan", *ZONES) == 0
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 2

        err = capsys.readouterr().err
        assert "outside this protocol" not in err
        assert "canonical holds this run's kept copy again" in err
        assert "Run rollback" in err
        assert env.fake.puts() == []
        # The named remedy: rollback writes only niamey's event, and the run promotes again.
        assert _run(env, "rollback", *ZONES) == 0
        assert "by a rollback or an unwind that stopped" in capsys.readouterr().out
        assert _journal(env, "niamey")[-1]["canonical_written"] is False
        assert _run(env, "plan", *ZONES) == 0
        assert _run(env, "promote", *ZONES) == 0
        assert all(_canonical(env, z) == _staged(env, z) for z in ZONES)

    def test_a_resumed_zone_the_unwind_restored_before_its_event_is_named_for_rollback(
        self, env, capsys
    ):
        """The same, on a resume: fos_sur_mer's first promotion was killed between its PUT and its
        event, so the resumed event records canonical_sha256_before as the staged v2 and only
        rollback_sha256 names the kept v1. The resume's unwind restored that kept copy and was
        killed before its unwind event. Promote must recognise the kept copy by rollback_sha256,
        not call it another writer's."""
        _prepare(env, *ZONES)
        _killed_on_the_zone_event(env, "fos_sur_mer", *ZONES)
        assert _run(env, "plan", *ZONES) == 0
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")
        v1 = env.seed["fos_sur_mer"]["v1"]
        _fail_once_on_read(env, op._ZONE_EVENT.format(zone_id="fos_sur_mer", seq=1))
        real_put = env.fake.put_object

        def _put(**request):
            real_put(**request)
            if request["Key"] == fos and request["Body"] == v1:
                raise KeyboardInterrupt  # killed once the unwind's restore landed

        env.fake.put_object = _put
        with pytest.raises(KeyboardInterrupt):
            _run(env, "promote", *ZONES)
        del env.fake.put_object
        (event,) = _journal(env, "fos_sur_mer")
        assert event["canonical_sha256_before"] == event["canonical_sha256_after"]  # a resume
        assert event["rollback_sha256"] != event["canonical_sha256_before"]
        assert _canonical(env, "fos_sur_mer") == v1
        assert _run(env, "plan", *ZONES) == 0
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 2

        err = capsys.readouterr().err
        assert "outside this protocol" not in err
        assert "canonical holds this run's kept copy again" in err
        assert env.fake.puts() == []
        assert _run(env, "rollback", *ZONES) == 0
        assert all(_canonical(env, z) == env.seed[z]["v1"] for z in ZONES)


class TestRollbackOwnershipComesFromTheJournal:
    def test_a_run_cannot_roll_back_a_canonical_another_run_now_owns(self, env, capsys):
        """N4 (#199 review). r1 and r2 compute from the same ERA5, so they stage IDENTICAL bytes
        and a sha256 check alone would let r1's rollback undo r2's promotion."""
        _prepare(env, "niamey", run_id="r1")
        assert _run(env, "promote", "niamey", run_id="r1") == 0
        _prepare(env, "niamey", run_id="r2")
        assert _staged(env, "niamey", "r2") == _staged(env, "niamey", "r1")
        assert _run(env, "promote", "niamey", run_id="r2") == 0

        assert _run(env, "rollback", "niamey", run_id="r1") == 2

        assert "does not own canonical" in capsys.readouterr().err
        assert _canonical(env, "niamey") == _staged(env, "niamey", "r2")
        # In order, both unwind: r2 back to r1's v2, then r1 back to the v1.
        assert _run(env, "rollback", "niamey", run_id="r2") == 0
        assert _run(env, "rollback", "niamey", run_id="r1") == 0
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

    def test_a_run_that_was_never_promoted_cannot_be_rolled_back(self, env, capsys):
        _prepare(env, "niamey")

        assert _run(env, "rollback", "niamey") == 2
        assert "not marked promoted" in capsys.readouterr().err

    def test_rollback_refuses_a_canonical_written_outside_the_protocol(self, env, capsys):
        """`climate_percentiles` with overwrite=True is a second path to the same key."""
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        key = op._CANONICAL.format(zone_id="niamey")
        env.fake.objects[key] = FOREIGN
        env.fake.calls.clear()

        assert _run(env, "rollback", "niamey") == 2

        assert "outside this protocol" in capsys.readouterr().err
        assert env.fake.objects[key] == FOREIGN
        assert env.fake.puts() == []

    def test_rollback_refuses_a_kept_copy_that_is_not_the_one_recorded(self, env, capsys):
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        env.fake.objects[op._ROLLBACK.format(run_id="r1", zone_id="niamey")] = b"tampered"
        env.fake.calls.clear()

        assert _run(env, "rollback", "niamey") == 2

        assert "not the canonical this run replaced" in capsys.readouterr().err
        assert _canonical(env, "niamey") == _staged(env, "niamey")
        assert env.fake.puts() == []

    def test_a_restore_that_reads_back_wrong_is_refused_and_not_journaled(self, env, capsys):
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        key = op._CANONICAL.format(zone_id="niamey")
        env.fake.on_put = lambda k, body: body[:-1] if k == key else body

        assert _run(env, "rollback", "niamey") == 2

        assert "reads back with other bytes than the rollback copy" in capsys.readouterr().err
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote"]
        assert _run_events(env)[-1]["kind"] == "promoted"

    def test_a_rollback_that_stopped_between_its_put_and_its_event_resumes(self, env, capsys):
        """The restore landed and the zone event did not. Canonical already holds the kept
        copy's bytes, so the resumed rollback writes only the missing event."""
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        journal = "climate/normals/events/zones/niamey/"

        def _die_on_the_event(key, body):
            if key.startswith(journal):
                raise ConnectionError("connection reset")
            return body

        env.fake.on_put = _die_on_the_event
        assert _run(env, "rollback", "niamey") == 1
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote"]
        env.fake.on_put = None
        env.fake.calls.clear()

        assert _run(env, "rollback", "niamey") == 0

        assert op._CANONICAL.format(zone_id="niamey") not in env.fake.puts()
        rolled_back = _journal(env, "niamey")[-1]
        assert rolled_back["kind"] == "rollback"
        assert rolled_back["canonical_written"] is False
        assert rolled_back["canonical_sha256_after"] == (
            hashlib.sha256(env.seed["niamey"]["v1"]).hexdigest()
        )
        capsys.readouterr()
        assert _run(env, "verify", "niamey") == 0
        assert _run(env, "plan", "niamey") == 0
        assert _run(env, "promote", "niamey") == 0

    def test_rollback_if_match_protects_the_state_it_verified(self, env, capsys):
        """A write landing after rollback verified canonical by GET: a fresh HEAD's etag would
        match it and the restore would overwrite it."""
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        key = op._CANONICAL.format(zone_id="niamey")
        real_get = env.fake.get_object
        armed = [True]

        def _get(**request):
            out = real_get(**request)
            if request["Key"] == key and armed[0]:
                armed.pop()
                armed.append(False)
                env.fake.objects[key] = FOREIGN  # right after the verifying read
            return out

        env.fake.get_object = _get

        assert _run(env, "rollback", "niamey", extra=("--if-match",)) == 2

        assert "changed after this rollback verified it" in capsys.readouterr().err
        assert env.fake.objects[key] == FOREIGN
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote"]

    def test_rolling_back_part_of_the_run_is_refused(self, env, capsys):
        _prepare(env, *ZONES)
        assert _run(env, "promote", *ZONES) == 0

        assert _run(env, "rollback", "niamey") == 2
        assert "Roll back the whole run" in capsys.readouterr().err
        assert _canonical(env, "niamey") == _staged(env, "niamey")


class TestRollbackAfterAPromoteThatDiedBeforeItsEvent:
    """Killed between a zone's canonical PUT and its event: canonical holds the staged v2, the
    kept copy the v1, and no standing promotion of this run accounts for the v2. Rollback used to
    restore only the journaled zones and report success, or claim the run replaced nothing."""

    @pytest.mark.parametrize("zones", [ZONES, ("fos_sur_mer",)], ids=["run", "canary"])
    def test_a_zone_whose_put_landed_without_its_event_is_refused_not_skipped(
        self, env, capsys, zones
    ):
        _prepare(env, *zones)
        _killed_on_the_zone_event(env, "fos_sur_mer", *zones)
        assert _canonical(env, "fos_sur_mer") == _staged(env, "fos_sur_mer")
        assert _journal(env, "fos_sur_mer") == []
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "rollback", *zones) == 2

        assert "stopped between its PUT and its event" in capsys.readouterr().err
        assert env.fake.puts() == []
        assert _run_events(env) == []
        # The way out the refusal names: resume, then roll back.
        assert _run(env, "plan", *zones) == 0
        assert _run(env, "promote", *zones) == 0
        assert _run(env, "rollback", *zones) == 0
        for zone_id in zones:
            assert _canonical(env, zone_id) == env.seed[zone_id]["v1"]

    def test_a_re_promotion_after_a_rollback_that_died_before_its_event_is_refused(
        self, env, capsys
    ):
        """This run's latest entry in fos_sur_mer is its rollback of the first cycle."""
        _prepare(env, *ZONES)
        assert _run(env, "promote", *ZONES) == 0
        assert _run(env, "rollback", *ZONES) == 0
        assert _run(env, "plan", *ZONES) == 0
        _killed_on_the_zone_event(env, "fos_sur_mer", *ZONES)
        assert _canonical(env, "fos_sur_mer") == _staged(env, "fos_sur_mer")
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "rollback", *ZONES) == 2

        assert "stopped between its PUT and its event" in capsys.readouterr().err
        assert env.fake.puts() == []

    def test_a_re_promotion_after_an_unwind_that_died_before_its_event_is_refused(
        self, env, capsys
    ):
        """This run's latest entry in fos_sur_mer is the unwind of a failed first attempt. An
        unwind returns canonical off this run's bytes, so staged bytes after it were written by a
        later attempt, whose event is missing."""
        _prepare(env, *ZONES)
        _fail_once_on_read(env, op._ZONE_EVENT.format(zone_id="fos_sur_mer", seq=1))
        assert _run(env, "promote", *ZONES) == 1
        for zone_id in ZONES:
            assert [e["kind"] for e in _journal(env, zone_id)] == ["promote", "unwind"]
            assert _canonical(env, zone_id) == env.seed[zone_id]["v1"]
        assert _run(env, "plan", *ZONES) == 0
        _killed_on_the_zone_event(env, "fos_sur_mer", *ZONES)
        assert _canonical(env, "fos_sur_mer") == _staged(env, "fos_sur_mer")
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "rollback", *ZONES) == 2

        assert "stopped between its PUT and its event" in capsys.readouterr().err
        assert env.fake.puts() == []
        assert _run(env, "plan", *ZONES) == 0
        assert _run(env, "promote", *ZONES) == 0
        assert _run(env, "rollback", *ZONES) == 0
        for zone_id in ZONES:
            assert _canonical(env, zone_id) == env.seed[zone_id]["v1"]

    def test_a_zone_with_no_prior_canonical_is_refused_not_skipped(self, env, capsys):
        """Only the .absent marker describes fos_sur_mer's pre-run state: no kept copy exists, and
        the refusal must not depend on one. niamey is journaled, so a skip would exit 0."""
        del env.fake.objects[op._CANONICAL.format(zone_id="fos_sur_mer")]
        _prepare(env, *ZONES)
        _killed_on_the_zone_event(env, "fos_sur_mer", *ZONES)
        assert op._ABSENT.format(run_id="r1", zone_id="fos_sur_mer") in env.fake.objects
        assert op._ROLLBACK.format(run_id="r1", zone_id="fos_sur_mer") not in env.fake.objects
        assert _canonical(env, "fos_sur_mer") == _staged(env, "fos_sur_mer")
        assert _journal(env, "fos_sur_mer") == []
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "rollback", *ZONES) == 2

        assert "stopped between its PUT and its event" in capsys.readouterr().err
        assert env.fake.puts() == []
        assert _run_events(env) == []

    def test_identical_bytes_another_run_promoted_are_not_stranded(self, env, capsys):
        """r2's promotion accounts for the bytes: r1's kept copy is there, but r1 has nothing
        stranded."""
        _prepare(env, "niamey", run_id="r1")
        assert _run(env, "promote", "niamey", run_id="r1") == 0
        assert _run(env, "rollback", "niamey", run_id="r1") == 0
        _prepare(env, "niamey", run_id="r2")
        assert _run(env, "promote", "niamey", run_id="r2") == 0
        assert _canonical(env, "niamey") == _staged(env, "niamey", "r1")
        assert op._ROLLBACK.format(run_id="r1", zone_id="niamey") in env.fake.objects
        capsys.readouterr()

        assert _run(env, "rollback", "niamey", run_id="r1") == 2

        err = capsys.readouterr().err
        assert "not marked promoted" in err
        assert "no zone event says so" not in err

    def test_a_run_with_no_copy_is_not_stranded(self, env, capsys):
        """r0's promotion of the same bytes died before its event. r1 never promoted and has no
        kept copy or marker: it is not told to resume a promotion it never made."""
        _prepare(env, "niamey", run_id="r0")
        _killed_on_the_zone_event(env, "niamey", "niamey", run_id="r0")
        _prepare(env, "niamey", run_id="r1")
        assert _canonical(env, "niamey") == _staged(env, "niamey", "r1")
        assert op._ROLLBACK.format(run_id="r1", zone_id="niamey") not in env.fake.objects
        assert op._ABSENT.format(run_id="r1", zone_id="niamey") not in env.fake.objects
        capsys.readouterr()

        assert _run(env, "rollback", "niamey", run_id="r1") == 2

        err = capsys.readouterr().err
        assert "not marked promoted" in err
        assert "no zone event says so" not in err

    @staticmethod
    def _unwound_through_a_flapping_network(env) -> None:
        """Promote both zones. fos_sur_mer's zone-event PUT fails after its canonical PUT landed,
        and the unwind's GET of fos_sur_mer's canonical fails in the same outage, so the unwind
        leaves it on the staged v2 with no event. The network is back for niamey's restore, and
        drops again for niamey's unwind event."""
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")
        real_put, real_get = env.fake.put_object, env.fake.get_object
        outage: list[str] = []

        def _put(**request):
            key = request["Key"]
            if key.startswith("climate/normals/events/zones/fos_sur_mer/") and not outage:
                outage.append("fos event")
                raise ConnectionError("connection reset")
            if key.startswith("climate/normals/events/zones/niamey/") and outage:
                raise ConnectionError("connection reset")
            return real_put(**request)

        def _get(**request):
            if request["Key"] == fos and outage == ["fos event"]:
                outage.append("unwind GET")
                raise ConnectionError("connection reset")
            return real_get(**request)

        env.fake.put_object, env.fake.get_object = _put, _get
        assert _run(env, "promote", *ZONES) == 1
        del env.fake.put_object, env.fake.get_object
        assert outage == ["fos event", "unwind GET"]
        assert _canonical(env, "fos_sur_mer") == _staged(env, "fos_sur_mer")
        assert _journal(env, "fos_sur_mer") == []
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote"]

    @pytest.mark.parametrize("if_match", [(), ("--if-match",)], ids=["plain", "if-match"])
    def test_a_stranded_zone_is_restored_when_promote_cannot_resume(self, env, capsys, if_match):
        """niamey holds its kept copy again with no event, so promote refuses to resume and names
        rollback; fos_sur_mer is stranded, which rollback used to refuse by naming the resume.
        Each named the other and neither wrote anything. Rollback restores the stranded zone from
        its kept copy instead, and journals it."""
        _prepare(env, *ZONES)
        self._unwound_through_a_flapping_network(env)
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        assert _run(env, "plan", *ZONES) == 0
        capsys.readouterr()
        assert _run(env, "promote", *ZONES) == 2
        assert "canonical holds this run's kept copy again" in capsys.readouterr().err
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")
        kept = op._ROLLBACK.format(run_id="r1", zone_id="fos_sur_mer")

        assert _run(env, "rollback", *ZONES, extra=if_match) == 0, capsys.readouterr().err

        out = capsys.readouterr().out
        assert "restored fos_sur_mer" in out
        assert "restored 2 zone(s)" in out
        for zone_id in ZONES:
            assert _canonical(env, zone_id) == env.seed[zone_id]["v1"]
        restore = env.fake.put_requests_for(fos)[-1]
        assert restore["Body"] == env.seed["fos_sur_mer"]["v1"]
        assert restore.get("IfMatch") == (_etag(_staged(env, "fos_sur_mer")) if if_match else None)
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote", "rollback"]
        (event,) = _journal(env, "fos_sur_mer")
        assert event["kind"] == "rollback"
        assert event["without_promote_event"] is True
        assert event["canonical_written"] is True
        assert event["restored_from"] == kept
        assert (
            event["canonical_sha256_before"]
            == hashlib.sha256(_staged(env, "fos_sur_mer")).hexdigest()
        )
        assert (
            event["canonical_sha256_after"]
            == hashlib.sha256(env.seed["fos_sur_mer"]["v1"]).hexdigest()
        )
        run_event = _run_events(env)[-1]
        assert run_event["kind"] == "rolled_back"
        assert run_event["restored_without_promote_event"] == ["fos_sur_mer"]
        assert _run(env, "verify", *ZONES) == 0
        # The run promotes again from a fresh plan, and rolls back again.
        assert _run(env, "plan", *ZONES) == 0
        assert _run(env, "promote", *ZONES) == 0
        assert all(_canonical(env, z) == _staged(env, z) for z in ZONES)
        assert _run(env, "rollback", *ZONES) == 0
        assert all(_canonical(env, z) == env.seed[z]["v1"] for z in ZONES)

    def test_a_stranded_zone_is_restored_once_a_released_zone_was_removed_by_hand(
        self, env, capsys
    ):
        """niamey had no canonical before, so the unwind left this run's v2 there for removal by
        hand. Once it is removed, promote refuses niamey as written outside this protocol, and
        the resume the stranded refusal names is closed as well."""
        niamey = op._CANONICAL.format(zone_id="niamey")
        del env.fake.objects[niamey]
        _prepare(env, *ZONES)
        self._unwound_through_a_flapping_network(env)
        assert _canonical(env, "niamey") == _staged(env, "niamey")
        # While niamey still holds the staged v2, the resume is open and is what rollback names.
        capsys.readouterr()
        assert _run(env, "rollback", *ZONES) == 2
        assert "Re-run plan and promote to resume" in capsys.readouterr().err
        del env.fake.objects[niamey]  # as the unwind asked
        env.fake.calls.clear()

        assert _run(env, "rollback", *ZONES) == 0, capsys.readouterr().err

        assert _canonical(env, "niamey") is None
        assert _canonical(env, "fos_sur_mer") == env.seed["fos_sur_mer"]["v1"]
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote", "rollback"]
        assert [e["kind"] for e in _journal(env, "fos_sur_mer")] == ["rollback"]
        assert _run_events(env)[-1]["kind"] == "rolled_back"
        assert niamey not in env.fake.puts()

    def test_a_stranded_zone_with_no_canonical_before_is_refused_as_a_delete(self, env, capsys):
        """fos_sur_mer had no canonical before: only its .absent marker describes that state, and
        restoring it means removing an object. Rollback names that removal, writes nothing, and
        finishes once it is done."""
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")
        del env.fake.objects[fos]
        _prepare(env, *ZONES)
        self._unwound_through_a_flapping_network(env)
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "rollback", *ZONES) == 2

        err = capsys.readouterr().err
        assert "no zone event says so" in err
        assert "promote cannot resume" in err
        assert f"remove s3://{BUCKET}/{fos} yourself" in err
        assert "Re-run plan and promote to resume" not in err
        assert env.fake.puts() == []
        del env.fake.objects[fos]  # the named way out

        assert _run(env, "rollback", *ZONES) == 0

        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        assert _canonical(env, "fos_sur_mer") is None
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote", "rollback"]
        assert _run_events(env)[-1]["kind"] == "rolled_back"


class TestPromoteNamesTheRunOnTop:
    def test_a_promoted_run_under_another_is_not_told_to_finish_a_rollback(self, env, capsys):
        """r1 was never rolled back: r2 promoted on top of it. Promote refuses and names r2, not a
        rollback of r1 whose run event is missing: `rollback r1` is itself refused (N4)."""
        _prepare(env, "niamey", run_id="r1")
        assert _run(env, "promote", "niamey", run_id="r1") == 0
        _prepare(env, "niamey", run_id="r2")
        assert _run(env, "promote", "niamey", run_id="r2") == 0
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", "niamey", run_id="r1") == 2

        err = capsys.readouterr().err
        assert "rolled back without its run event" not in err
        assert "already promoted, and another run's promotion sits on top" in err
        assert "niamey: run 'r2' (promote, zone event 2)" in err
        assert env.fake.puts() == []

    def test_an_unfinished_rollback_is_still_named(self, env, capsys):
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=2))
        assert _run(env, "rollback", "niamey") == 1
        capsys.readouterr()

        assert _run(env, "promote", "niamey") == 2

        assert "rolled back without its run event" in capsys.readouterr().err

    def test_a_partial_rollback_is_still_named(self, env, capsys):
        """Rollback restored niamey and wrote its event, then fos_sur_mer's restore PUT failed:
        fos_sur_mer is still this run's, and that is not another run's promotion on top of it."""
        _prepare(env, *ZONES)
        assert _run(env, "promote", *ZONES) == 0
        _die_once_on(env, op._CANONICAL.format(zone_id="fos_sur_mer"))
        assert _run(env, "rollback", *ZONES) == 1
        env.fake.on_put = None
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        assert _canonical(env, "fos_sur_mer") == _staged(env, "fos_sur_mer")
        capsys.readouterr()
        env.fake.calls.clear()

        assert _run(env, "promote", *ZONES) == 2

        err = capsys.readouterr().err
        assert "rolled back without its run event" in err
        assert "sits on top" not in err
        assert env.fake.puts() == []
        # The named remedy finishes it.
        assert _run(env, "rollback", *ZONES) == 0
        assert _canonical(env, "fos_sur_mer") == env.seed["fos_sur_mer"]["v1"]

    def test_a_half_rolled_back_run_under_another_is_told_to_finish_its_rollback(self, env, capsys):
        """r1's rollback restored niamey, then fos_sur_mer's restore PUT failed. r2 then promoted
        niamey over nobody's promotion: r1's is not in niamey's stack. Promote names r1's
        unfinished rollback, not r2, whose rollback would undo a legitimate promotion."""
        _prepare(env, *ZONES)
        assert _run(env, "promote", *ZONES) == 0
        _die_once_on(env, op._CANONICAL.format(zone_id="fos_sur_mer"))
        assert _run(env, "rollback", *ZONES) == 1
        env.fake.on_put = None
        _prepare(env, "niamey", run_id="r2")
        assert _run(env, "promote", "niamey", run_id="r2") == 0
        r2_bytes = _canonical(env, "niamey")
        capsys.readouterr()
        env.fake.calls.clear()

        assert _run(env, "promote", *ZONES) == 2

        err = capsys.readouterr().err
        assert "sits on top" not in err
        assert "rolled back without its run event" in err
        assert env.fake.puts() == []
        # The named remedy finishes r1's rollback and leaves r2's promotion alone.
        assert _run(env, "rollback", *ZONES) == 0
        assert _canonical(env, "fos_sur_mer") == env.seed["fos_sur_mer"]["v1"]
        assert _canonical(env, "niamey") == r2_bytes


class TestAConditionalWriteThatLandedIsNotReadAsRefused:
    """A 412 or 409 proves that nothing landed only for a first attempt. botocore retries a PUT
    whose response was lost (a timeout, a reset, a 5xx after the commit) with the same
    precondition, and the store refuses the retry because the first attempt landed. Every
    conditional write reads the key before concluding (PR #224 review)."""

    @CONFLICTS
    def test_an_if_match_canonical_put_that_landed_is_promoted(self, env, capsys, conflict):
        """It used to be refused as another writer's, with this run's v2 left in canonical and
        no journal entry: rollback then said the run replaced nothing."""
        env.fake.conflict = conflict
        _prepare(env, *ZONES)
        _retried_after_it_landed(env, op._CANONICAL.format(zone_id="fos_sur_mer"))

        assert _run(env, "promote", *ZONES, extra=("--if-match",)) == 0

        assert _canonical(env, "fos_sur_mer") == _staged(env, "fos_sur_mer")
        (event,) = _journal(env, "fos_sur_mer")
        assert (event["kind"], event["run_id"], event["if_match"]) == ("promote", "r1", True)
        assert _run(env, "rollback", *ZONES) == 0
        assert _canonical(env, "fos_sur_mer") == env.seed["fos_sur_mer"]["v1"]

    @CONFLICTS
    def test_a_create_only_canonical_put_that_landed_is_promoted(self, env, conflict):
        env.fake.conflict = conflict
        key = op._CANONICAL.format(zone_id="niamey")
        del env.fake.objects[key]
        _prepare(env, "niamey")
        _retried_after_it_landed(env, key)

        assert _run(env, "promote", "niamey") == 0

        assert _canonical(env, "niamey") == _staged(env, "niamey")
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote"]
        assert op._ABSENT.format(run_id="r1", zone_id="niamey") in env.fake.objects

    @CONFLICTS
    @pytest.mark.parametrize("journal", ["zone", "run"])
    def test_an_event_that_landed_is_this_event(self, env, conflict, journal):
        """It used to read as another writer appending concurrently."""
        env.fake.conflict = conflict
        _prepare(env, "niamey")
        key = (
            op._ZONE_EVENT.format(zone_id="niamey", seq=1)
            if journal == "zone"
            else op._RUN_EVENT.format(run_id="r1", seq=1)
        )
        _retried_after_it_landed(env, key)

        assert _run(env, "promote", "niamey") == 0

        assert _canonical(env, "niamey") == _staged(env, "niamey")
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote"]
        assert [e["kind"] for e in _run_events(env)] == ["promoted"]

    @CONFLICTS
    def test_an_if_match_restore_that_landed_is_rolled_back(self, env, conflict):
        """It used to say that nothing was restored, with canonical already back on the v1."""
        env.fake.conflict = conflict
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        _retried_after_it_landed(env, op._CANONICAL.format(zone_id="niamey"))

        assert _run(env, "rollback", "niamey", extra=("--if-match",)) == 0

        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        rolled_back = _journal(env, "niamey")[-1]
        assert (rolled_back["kind"], rolled_back["canonical_written"]) == ("rollback", True)
        assert [e["kind"] for e in _run_events(env)] == ["promoted", "rolled_back"]

    @CONFLICTS
    def test_an_if_match_unwind_restore_that_landed_is_unwound(self, env, capsys, conflict):
        """The unwind's own If-Match restore landed and lost its response; botocore's retry got
        the conflict. It used to be reported as THE UNWIND FAILED with canonical already back on
        the v1 and no unwind event, and a re-promote was refused as written outside this
        protocol."""
        env.fake.conflict = conflict
        _prepare(env, *ZONES)
        niamey = op._CANONICAL.format(zone_id="niamey")
        _fail_once_on_read(env, op._ZONE_EVENT.format(zone_id="fos_sur_mer", seq=1))
        real_put, count = env.fake.put_object, {"n": 0}

        def _put(**request):
            real_put(**request)
            if request["Key"] == niamey:
                count["n"] += 1
                if count["n"] == 2:  # the unwind's restore landed; botocore sends it again
                    real_put(**request)

        env.fake.put_object = _put

        assert _run(env, "promote", *ZONES, extra=("--if-match",)) == 1

        err = capsys.readouterr().err
        assert "THE UNWIND FAILED" not in err
        assert "restored to their bytes before this invocation: ['fos_sur_mer', 'niamey']" in err
        assert [r["IfMatch"] for r in env.fake.put_requests_for(niamey)] == [
            _etag(env.seed["niamey"]["v1"]),
            _etag(_staged(env, "niamey")),
            _etag(_staged(env, "niamey")),
        ]
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        assert [(e["kind"], e["run_id"]) for e in _journal(env, "niamey")] == [
            ("promote", "r1"),
            ("unwind", "r1"),
        ]
        assert _run(env, "plan", *ZONES) == 0
        assert _run(env, "promote", *ZONES, extra=("--if-match",)) == 0
        assert all(_canonical(env, z) == _staged(env, z) for z in ZONES)


class TestCanonicalRemovedBeforeTheConditionalPut:
    """Another writer removes canonical between the read that verified it and the If-Match PUT,
    and the store refuses the PUT. The read that follows the refusal finds no object, which is
    not this run's bytes: a refusal, not a crash."""

    @staticmethod
    def _remove_on_if_match_put(env, zone_id: str) -> str:
        key = op._CANONICAL.format(zone_id=zone_id)
        real_put = env.fake.put_object

        def _put(**request):
            if request["Key"] == key and request.get("IfMatch"):
                env.fake.objects.pop(key, None)
            return real_put(**request)

        env.fake.put_object = _put
        return key

    def test_a_promote_refused_because_canonical_was_removed_touches_nothing(self, env, capsys):
        _prepare(env, "niamey")
        key = self._remove_on_if_match_put(env, "niamey")
        capsys.readouterr()

        assert _run(env, "promote", "niamey", extra=("--if-match",)) == 2

        err = capsys.readouterr().err
        assert "does not hold this run's staged bytes" in err
        assert "touching 0 zone(s)" in err
        assert "CHANGED BY ANOTHER WRITER" not in err
        assert key not in env.fake.objects
        assert _journal(env, "niamey") == []

    def test_a_rollback_refused_because_canonical_was_removed_restores_nothing(self, env, capsys):
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        key = self._remove_on_if_match_put(env, "niamey")
        capsys.readouterr()

        assert _run(env, "rollback", "niamey", extra=("--if-match",)) == 2

        assert "changed after this rollback verified it" in capsys.readouterr().err
        assert key not in env.fake.objects
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote"]


class TestAZoneWithNoCanonicalBefore:
    def test_the_absence_is_marked_and_its_rollback_is_refused_rather_than_deleting(
        self, env, capsys
    ):
        del env.fake.objects[op._CANONICAL.format(zone_id="niamey")]
        _prepare(env, "niamey")
        plan = json.loads((env.root / op._PLAN.format(run_id="r1")).read_text())
        assert plan["zones"]["niamey"]["canonical"]["exists"] is False

        assert _run(env, "promote", "niamey") == 0

        absent = op._ABSENT.format(run_id="r1", zone_id="niamey")
        assert absent in env.fake.objects
        assert op._ROLLBACK.format(run_id="r1", zone_id="niamey") not in env.fake.objects
        canonical_put = next(
            r for r in env.fake.put_requests if r["Key"] == op._CANONICAL.format(zone_id="niamey")
        )
        assert canonical_put["IfNoneMatch"] == "*"

        assert _run(env, "rollback", "niamey") == 2
        assert "does not delete" in capsys.readouterr().err
        assert _canonical(env, "niamey") == _staged(env, "niamey")


class TestOnlyTheBaselineWindowReachesCanonical:
    """Canonical is the WMO 1991-2020 baseline `climate_normals` serves. A complete one-year
    artifact passes every coverage and schema check, so the window itself is the guard (Codex P1,
    PR #224)."""

    def test_the_guard_is_the_wmo_baseline(self):
        assert op._CANONICAL_WINDOW != {"start": YEAR_START, "end": YEAR_END}  # the fixture's
        assert {"start": op.WMO_START, "end": op.WMO_END} == {
            "start": "1991-01-01",
            "end": "2020-12-31",
        }

    def test_another_window_is_not_materialized(self, env, monkeypatch, capsys):
        monkeypatch.setattr(op, "_CANONICAL_WINDOW", {"start": op.WMO_START, "end": op.WMO_END})
        assert _run(env, "fetch", "niamey") == 0

        assert _run(env, "materialize", "niamey") == 2
        assert "different artifact" in capsys.readouterr().err
        assert not (env.root / op._RUN_RECORD.format(run_id="r1")).exists()
        assert not (env.root / op._STAGING.format(run_id="r1", zone_id="niamey")).exists()

    def test_a_run_record_for_another_window_is_neither_planned_nor_promoted(
        self, env, monkeypatch, capsys
    ):
        _prepare(env, "niamey")
        monkeypatch.setattr(op, "_CANONICAL_WINDOW", {"start": op.WMO_START, "end": op.WMO_END})
        capsys.readouterr()

        assert _run(env, "plan", "niamey") == 2
        assert "different artifact" in capsys.readouterr().err
        assert _run(env, "promote", "niamey") == 2
        assert "different artifact" in capsys.readouterr().err
        assert op._CANONICAL.format(zone_id="niamey") not in env.fake.puts()


def _die_once_on(env, key: str) -> None:
    """The PUT of ``key`` fails once, as a connection reset, then the store behaves."""
    failed: list[str] = []

    def _on_put(put_key, body):
        if put_key == key and not failed:
            failed.append(put_key)
            raise ConnectionError("connection reset")
        return body

    env.fake.on_put = _on_put


class TestTheRunEventIsFinishedOnRetry:
    """Every zone event landed and the run event did not. Re-running the same step writes only
    that event; before, promote said "already promoted" and rollback "not marked promoted", and the
    run journal could never be completed (Greptile P1, Codex P2, PR #224)."""

    def test_a_promotion_that_stopped_before_its_run_event_finishes_it(self, env, capsys):
        _prepare(env, *ZONES)
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=1))
        assert _run(env, "promote", *ZONES) == 1
        assert _run_events(env) == []
        assert all(_canonical(env, z) == _staged(env, z) for z in ZONES)
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 0

        assert "wrote only the run event" in capsys.readouterr().out
        assert env.fake.puts() == [op._RUN_EVENT.format(run_id="r1", seq=1)]
        (event,) = _run_events(env)
        assert event["kind"] == "promoted"
        assert set(event["zones"]) == set(ZONES)
        assert _run(env, "promote", *ZONES) == 2
        assert "already promoted" in capsys.readouterr().err

    def test_a_rollback_that_stopped_before_its_run_event_finishes_it(self, env, capsys):
        _prepare(env, *ZONES)
        assert _run(env, "promote", *ZONES) == 0
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=2))
        assert _run(env, "rollback", *ZONES) == 1
        assert [e["kind"] for e in _run_events(env)] == ["promoted"]
        assert all(_canonical(env, z) == env.seed[z]["v1"] for z in ZONES)
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "rollback", *ZONES) == 0

        assert "wrote only the run event" in capsys.readouterr().out
        assert env.fake.puts() == [op._RUN_EVENT.format(run_id="r1", seq=2)]
        event = _run_events(env)[-1]
        assert event["kind"] == "rolled_back"
        assert {z: e["kind"] for z, e in event["zones"].items()} == dict.fromkeys(ZONES, "rollback")
        assert _run(env, "rollback", *ZONES) == 2
        assert "not marked promoted" in capsys.readouterr().err

    def test_only_a_rollback_in_every_zone_finishes_one(self, env, capsys):
        """Nothing left to restore is not enough: a journal whose latest entry for this run is
        not its rollback (here an out-of-protocol unwind) does not get a rolled_back run event,
        which would say canonical went back when it still holds the v2. The operator's own unwind
        never leaves this run's v2 in canonical, so these bytes are stranded, as after a promote
        that stopped between its PUT and its event, and rollback says so."""
        _prepare(env, *ZONES)
        assert _run(env, "promote", *ZONES) == 0
        for zone_id in ZONES:
            entries = _journal(env, zone_id)
            key = op._ZONE_EVENT.format(zone_id=zone_id, seq=len(entries) + 1)
            env.fake.objects[key] = op._record_bytes(
                {"kind": "unwind", "run_id": "r1", "zone_id": zone_id, "seq": len(entries) + 1}
            )
        capsys.readouterr()

        assert _run(env, "rollback", *ZONES) == 2

        assert "stopped between its PUT and its event" in capsys.readouterr().err
        assert [e["kind"] for e in _run_events(env)] == ["promoted"]

    def test_a_promotion_waits_for_an_unfinished_rollback(self, env, capsys):
        """Promoting again before the rollback's run event is written would leave the run
        journal saying promoted, promoted: a rollback nobody could see."""
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=2))
        assert _run(env, "rollback", "niamey") == 1
        assert _run(env, "plan", "niamey") == 0
        capsys.readouterr()

        assert _run(env, "promote", "niamey") == 2
        assert "Run rollback again" in capsys.readouterr().err
        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]

        assert _run(env, "rollback", "niamey") == 0
        assert _run(env, "plan", "niamey") == 0
        assert _run(env, "promote", "niamey") == 0
        assert [e["kind"] for e in _run_events(env)] == ["promoted", "rolled_back", "promoted"]

    def test_a_promotion_waits_for_a_rollback_whose_promotion_had_no_run_event(self, env, capsys):
        """The run journal is empty, so it never says promoted: the zone journals say that a
        rollback stands without its run event, and promote used to go ahead and bury it."""
        _prepare(env, *ZONES)
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=1))
        assert _run(env, "promote", *ZONES) == 1
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=1))
        assert _run(env, "rollback", *ZONES) == 1
        assert _run_events(env) == []
        assert _run(env, "plan", *ZONES) == 0
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 2

        assert "Run rollback again" in capsys.readouterr().err
        assert all(_canonical(env, z) == env.seed[z]["v1"] for z in ZONES)
        assert _run(env, "rollback", *ZONES) == 0
        assert _run(env, "plan", *ZONES) == 0
        assert _run(env, "promote", *ZONES) == 0
        assert [e["kind"] for e in _run_events(env)] == ["rolled_back", "promoted"]

    def test_a_promotion_waits_for_an_unfinished_rollback_in_a_second_cycle(self, env, capsys):
        """The last run event is the first cycle's rolled_back, which does not record the second
        cycle's rollback."""
        _prepare(env, *ZONES)
        assert _run(env, "promote", *ZONES) == 0
        assert _run(env, "rollback", *ZONES) == 0
        assert _run(env, "plan", *ZONES) == 0
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=3))
        assert _run(env, "promote", *ZONES) == 1
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=3))
        assert _run(env, "rollback", *ZONES) == 1
        assert _run(env, "plan", *ZONES) == 0
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 2

        assert "Run rollback again" in capsys.readouterr().err
        assert [e["kind"] for e in _run_events(env)] == ["promoted", "rolled_back"]

    def test_a_run_event_names_a_zone_a_dead_promotion_never_reached(self, env):
        """The process died after the first zone's event, so no unwind ran. Rollback restores
        that zone, and the run event says the second has no entry rather than failing on it
        after the restore."""
        _prepare(env, *ZONES)
        first, second = ZONES

        def _die(key, body):
            if key == op._ROLLBACK.format(run_id="r1", zone_id=second):
                raise KeyboardInterrupt  # not an Exception: nothing unwinds, as when killed
            return body

        env.fake.on_put = _die
        with pytest.raises(KeyboardInterrupt):
            _run(env, "promote", *ZONES)
        env.fake.on_put = None
        assert _canonical(env, first) == _staged(env, first)
        assert [e["kind"] for e in _journal(env, second)] == []

        assert _run(env, "rollback", *ZONES) == 0

        assert _canonical(env, first) == env.seed[first]["v1"]
        zones = _run_events(env)[-1]["zones"]
        assert zones[first]["kind"] == "rollback"
        assert zones[second] == {"seq": None, "kind": None, "canonical_sha256_after": None}


class TestARollbackOfAnUnrecordedPromotionIsFinishedOnRetry:
    """The promotion before the rollback never wrote its run event: its own PUT failed, or the
    process died before some zone. The rollback's run event is still written on a retry, instead
    of the retry being refused as "not marked promoted" (PR #224 review)."""

    def test_after_a_promotion_whose_run_event_failed(self, env, capsys):
        _prepare(env, *ZONES)
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=1))
        assert _run(env, "promote", *ZONES) == 1
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=1))
        assert _run(env, "rollback", *ZONES) == 1
        assert _run_events(env) == []
        assert all(_canonical(env, z) == env.seed[z]["v1"] for z in ZONES)
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "rollback", *ZONES) == 0

        assert "wrote only the run event" in capsys.readouterr().out
        assert env.fake.puts() == [op._RUN_EVENT.format(run_id="r1", seq=1)]
        (event,) = _run_events(env)
        assert event["kind"] == "rolled_back"
        assert {z: e["kind"] for z, e in event["zones"].items()} == dict.fromkeys(ZONES, "rollback")
        assert _run(env, "rollback", *ZONES) == 2
        assert "not marked promoted" in capsys.readouterr().err

    def test_after_a_dead_promotion_that_never_reached_a_zone(self, env, capsys):
        _prepare(env, *ZONES)
        first, second = ZONES

        def _die(key, body):
            if key == op._ROLLBACK.format(run_id="r1", zone_id=second):
                raise KeyboardInterrupt  # not an Exception: nothing unwinds, as when killed
            return body

        env.fake.on_put = _die
        with pytest.raises(KeyboardInterrupt):
            _run(env, "promote", *ZONES)
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=1))
        assert _run(env, "rollback", *ZONES) == 1
        assert _canonical(env, first) == env.seed[first]["v1"]
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "rollback", *ZONES) == 0

        assert env.fake.puts() == [op._RUN_EVENT.format(run_id="r1", seq=1)]
        zones = _run_events(env)[-1]["zones"]
        assert zones[first]["kind"] == "rollback"
        assert zones[second] == {"seq": None, "kind": None, "canonical_sha256_after": None}

    def test_in_a_second_cycle(self, env, capsys):
        _prepare(env, *ZONES)
        assert _run(env, "promote", *ZONES) == 0
        assert _run(env, "rollback", *ZONES) == 0
        assert _run(env, "plan", *ZONES) == 0
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=3))
        assert _run(env, "promote", *ZONES) == 1
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=3))
        assert _run(env, "rollback", *ZONES) == 1
        capsys.readouterr()

        assert _run(env, "rollback", *ZONES) == 0

        assert [e["kind"] for e in _run_events(env)] == ["promoted", "rolled_back", "rolled_back"]
        assert _run(env, "rollback", *ZONES) == 2
        assert "not marked promoted" in capsys.readouterr().err

    def test_an_unwound_promotion_is_still_not_rolled_back(self, env, capsys):
        """Not every latest entry finishes one: after an unwind that returned every zone to its
        v1, there is nothing this run replaced, and no rolled_back event is written."""
        _prepare(env, *ZONES)
        _fail_once_on_read(env, op._ZONE_EVENT.format(zone_id="fos_sur_mer", seq=1))
        assert _run(env, "promote", *ZONES) == 1
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote", "unwind"]
        capsys.readouterr()

        assert _run(env, "rollback", *ZONES) == 2

        assert "not marked promoted" in capsys.readouterr().err
        assert _run_events(env) == []

    def test_in_a_second_cycle_that_never_reached_a_zone(self, env, capsys):
        """A zone the second promotion never reached keeps cycle 1's rollback as its latest entry,
        which cycle 1's rolled_back event names: the retry still writes the second one."""
        _prepare(env, *ZONES)
        assert _run(env, "promote", *ZONES) == 0
        assert _run(env, "rollback", *ZONES) == 0
        assert _run(env, "plan", *ZONES) == 0
        fos = op._CANONICAL.format(zone_id="fos_sur_mer")

        def _die(key, body):
            if key == fos:
                raise KeyboardInterrupt  # killed on fos's canonical PUT, after niamey's event
            return body

        env.fake.on_put = _die
        with pytest.raises(KeyboardInterrupt):
            _run(env, "promote", *ZONES)
        assert [e["kind"] for e in _journal(env, "fos_sur_mer")] == ["promote", "rollback"]
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=3))
        assert _run(env, "rollback", *ZONES) == 1
        env.fake.on_put = None
        capsys.readouterr()

        assert _run(env, "rollback", *ZONES) == 0

        assert [e["kind"] for e in _run_events(env)] == ["promoted", "rolled_back", "rolled_back"]
        assert all(_canonical(env, z) == env.seed[z]["v1"] for z in ZONES)

    def test_after_a_resume_that_unwound_another_zone(self, env, capsys):
        """The first attempt died after niamey's event. The resume promoted fos_sur_mer and failed
        after its event, so the unwind returned fos_sur_mer and left niamey, which it did not
        touch. Rolling back restores niamey and loses its run event: the retry used to be refused
        because fos_sur_mer's latest entry is an unwind, not a rollback."""
        niamey, fos = ZONES
        _prepare(env, *ZONES)

        def _die(key, body):
            if key == op._ROLLBACK.format(run_id="r1", zone_id=fos):
                raise KeyboardInterrupt  # not an Exception: nothing unwinds, as when killed
            return body

        env.fake.on_put = _die
        with pytest.raises(KeyboardInterrupt):
            _run(env, "promote", *ZONES)
        env.fake.on_put = None
        _fail_once_on_read(env, op._ZONE_EVENT.format(zone_id=fos, seq=1))
        assert _run(env, "promote", *ZONES) == 1
        assert [e["kind"] for e in _journal(env, fos)] == ["promote", "unwind"]
        _die_once_on(env, op._RUN_EVENT.format(run_id="r1", seq=1))
        assert _run(env, "rollback", *ZONES) == 1
        assert _canonical(env, niamey) == env.seed[niamey]["v1"]
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "rollback", *ZONES) == 0

        assert env.fake.puts() == [op._RUN_EVENT.format(run_id="r1", seq=1)]
        (event,) = _run_events(env)
        assert event["kind"] == "rolled_back"
        assert {z: e["kind"] for z, e in event["zones"].items()} == {
            niamey: "rollback",
            fos: "unwind",
        }
        assert _run(env, "rollback", *ZONES) == 2
        assert "not marked promoted" in capsys.readouterr().err

    def test_an_unwind_after_a_recorded_rollback_is_still_not_rolled_back(self, env, capsys):
        """fos_sur_mer's latest entry is the rollback the first cycle's run event records, and
        niamey's is the second cycle's unwind: nothing is missing, and nothing is written."""
        niamey, fos = ZONES
        _prepare(env, *ZONES)
        assert _run(env, "promote", *ZONES) == 0
        assert _run(env, "rollback", *ZONES) == 0
        assert _run(env, "plan", *ZONES) == 0
        niamey_event = op._ZONE_EVENT.format(zone_id=niamey, seq=3)

        def _move(key, body):
            # fos_sur_mer moves once niamey's event lands: it refuses before any write, and only
            # niamey is unwound.
            if key == niamey_event:
                env.fake.objects[op._CANONICAL.format(zone_id=fos)] = FOREIGN
            return body

        env.fake.on_put = _move
        assert _run(env, "promote", *ZONES) == 2
        env.fake.on_put = None
        assert [e["kind"] for e in _journal(env, niamey)][-1] == "unwind"
        assert [e["kind"] for e in _journal(env, fos)][-1] == "rollback"
        capsys.readouterr()

        assert _run(env, "rollback", *ZONES) == 2

        assert "not marked promoted" in capsys.readouterr().err
        assert [e["kind"] for e in _run_events(env)] == ["promoted", "rolled_back"]


class TestRollbackAndVerifyAreBoundToTheStore:
    """Rollback and verify may run under their own authorization comment, which their events
    record, but never against another bucket or endpoint than the run's (Codex P1, PR #224)."""

    def _retarget_the_record(self, env, **fields) -> None:
        key = op._RUN_RECORD.format(run_id="r1")
        record = json.loads(env.fake.objects[key])
        env.fake.objects[key] = op._record_bytes({**record, **fields})

    @pytest.mark.parametrize("field", ["bucket", "endpoint"])
    def test_a_run_made_under_another_store_is_not_rolled_back_or_verified(
        self, env, capsys, field
    ):
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        self._retarget_the_record(env, **{field: "elsewhere"})
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "rollback", "niamey") == 2
        assert f"another {field}" in capsys.readouterr().err
        assert _run(env, "verify", "niamey") == 2
        assert f"another {field}" in capsys.readouterr().err
        assert env.fake.puts() == []
        assert _canonical(env, "niamey") == _staged(env, "niamey")

    def test_a_rollback_under_its_own_authorization_is_run_and_recorded(self, env):
        _prepare(env, "niamey")
        assert _run(env, "promote", "niamey") == 0
        other = "https://github.com/theseedship/deposium_geoai/issues/223#issuecomment-2"
        argv = _with(_argv(env, "rollback", "niamey"), "--authorization", other)

        assert op.main(argv) == 0

        assert _canonical(env, "niamey") == env.seed["niamey"]["v1"]
        assert _journal(env, "niamey")[-1]["authorization"] == other
        assert _run_events(env)[-1]["authorization"] == other


#: The (variable, month) key the parity tests move. Any of the 72 would do.
KEY = ("temperature_2m", 7)
LABEL = f"{KEY[0]}/{KEY[1]}"
_ON_KEY = "variable = ? AND month = ?"


def _rewritten(tmp_path, source: bytes, name: str, select: str) -> Path:
    """``source`` rewritten through ``select``, which reads it as the view ``t`` and may use the
    two parameters of :data:`_ON_KEY` (bound to :data:`KEY`) where it has one."""
    import duckdb

    folder = tmp_path / "parity"
    folder.mkdir(exist_ok=True)
    src, dst = folder / f"{name}.source.parquet", folder / f"{name}.parquet"
    src.write_bytes(source)
    con = duckdb.connect(":memory:")
    try:
        con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{src}')")
        con.execute(
            f"COPY ({select}) TO '{dst}' (FORMAT PARQUET)", list(KEY) if "?" in select else []
        )
    finally:
        con.close()
    return dst


def _moved(field: str, delta: float | int) -> str:
    return (
        f"SELECT * REPLACE (CASE WHEN {_ON_KEY} THEN {field} + {delta} ELSE {field} END "
        f"AS {field}) FROM t"
    )


def _canonical_is(env, tmp_path, zone_id: str, select: str) -> None:
    """Replace the zone's v1 canonical in the fake store with the v1 rewritten through ``select``."""
    rewritten = _rewritten(tmp_path, env.seed[zone_id]["v1"], f"{zone_id}-canonical", select)
    env.fake.objects[op._CANONICAL.format(zone_id=zone_id)] = rewritten.read_bytes()


def _plan(env, run_id: str = "r1") -> dict:
    return json.loads((env.root / op._PLAN.format(run_id=run_id)).read_text())


def _parity_of(env, zone_id: str) -> dict:
    return _plan(env)["zones"][zone_id]["parity"]


class TestTheParityRuleIsExact:
    """The parity rule of #223 (decision 5, 2026-09-27): EXACT. Per zone, at plan time, the
    all-days p10, p50, p90, n and mean of v2 must equal v1's on exactly 72 unique (variable, month)
    keys. Any deviation is a STOP that promote enforces for the whole run, before any write, with no
    flag and no tolerance. It does not apply where there is no canonical."""

    def test_the_fixture_v1_is_the_all_days_part_of_v2_so_parity_is_exact(self, env, capsys):
        """HONEST ABOUT THE FIXTURE. `seed` builds v1 as the v2 of the same days minus its v2
        columns, so v1 IS the all-days part of v2 by construction: the premise is checked below
        on its own, by EXCEPT ALL both ways. What this test proves is therefore no false STOP —
        the wet-day columns v2 adds, and a file re-encoded without them, are not read as a
        difference. That the verdict catches a difference is what the tests after it prove."""
        import duckdb

        _prepare(env, *ZONES)
        out = capsys.readouterr().out

        fields = "zone_id, variable, month, p10, p50, p90, n, mean"
        for zone_id in ZONES:
            staged = env.root / op._STAGING.format(run_id="r1", zone_id=zone_id)
            v1 = env.root / f"{zone_id}.v1.parquet"
            v1.write_bytes(env.seed[zone_id]["v1"])
            con = duckdb.connect(":memory:")
            columns = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{v1}'").fetchall()}
            assert not columns & set(op._WET_DAY_COLUMNS), "the fixture's v1 is not a v1"
            for a, b in ((staged, v1), (v1, staged)):
                left = f"SELECT {fields} FROM read_parquet('{a}')"
                right = f"SELECT {fields} FROM read_parquet('{b}')"
                assert con.execute(f"{left} EXCEPT ALL {right}").fetchall() == []
            con.close()

            parity = _parity_of(env, zone_id)
            assert parity["status"] == "exact"
            assert parity["exact"] is True
            assert parity["unique_keys"] == {"staged": 72, "canonical": 72}
            assert parity["identical_keys"] == 72
            assert parity["duplicated_keys"] == {"staged": [], "canonical": []}
            assert parity["only_in_staged"] == parity["only_in_canonical"] == []
            assert parity["differing_keys"] == {} and parity["reasons"] == []
            assert parity["fields"] == ["p10", "p50", "p90", "n", "mean"]
            assert sorted(parity["by_variable"]) == sorted(op.NORMALIZED_VARS)
            assert all(v["exact"] for v in parity["by_variable"].values())
            assert re.search(rf"^  exact\s+{zone_id}\s+72/72 keys identical$", out, re.M)
        assert "STOP" not in out

        assert _run(env, "promote", *ZONES) == 0

    def test_a_p50_moved_by_1e_4_on_one_month_differs_and_promote_refuses_before_any_put(
        self, env, tmp_path, capsys
    ):
        _canonical_is(env, tmp_path, "niamey", _moved("p50", 1e-4))
        before = _canonical(env, "niamey")

        _prepare(env, *ZONES)

        captured = capsys.readouterr()
        parity = _parity_of(env, "niamey")
        assert parity["status"] == "differs"
        assert parity["exact"] is False
        assert list(parity["differing_keys"]) == [LABEL]
        moved = parity["differing_keys"][LABEL]
        assert list(moved) == ["p50"]
        assert moved["p50"]["canonical"] - moved["p50"]["staged"] == pytest.approx(1e-4)
        assert parity["unique_keys"] == {"staged": 72, "canonical": 72}
        assert parity["identical_keys"] == 71
        assert parity["by_variable"][KEY[0]]["differing_fields"] == {"p50": 1}
        assert parity["by_variable"][KEY[0]]["exact"] is False
        assert _parity_of(env, "fos_sur_mer")["status"] == "exact"
        # The report plan wrote before the rule is kept, for anything that reads it.
        legacy = _plan(env)["zones"]["niamey"]["percentiles_all_days"]
        assert legacy["max_abs_diff"]["p50"] == pytest.approx(1e-4)
        assert "parity" not in legacy
        assert re.search(rf"^  DIFFERS\s+niamey\s+STOP: .*{LABEL}", captured.out, re.M)
        assert "STOP: parity is not exact for ['niamey']" in captured.err
        env.fake.calls.clear()

        assert _run(env, "promote", *ZONES) == 2

        err = capsys.readouterr().err
        assert "parity is not exact for ['niamey']" in err
        assert "No flag overrides it" in err
        assert "parity stop procedure in docs/runbooks/normals-v2-promotion.md" in err
        assert env.fake.puts() == []
        assert _canonical(env, "niamey") == before
        assert _canonical(env, "fos_sur_mer") == env.seed["fos_sur_mer"]["v1"]

    @pytest.mark.parametrize(
        "field, delta", [("p10", 1e-4), ("p50", 1e-4), ("p90", 1e-4), ("n", 1), ("mean", 1e-4)]
    )
    def test_a_difference_in_one_field_alone_differs(self, env, tmp_path, field, delta):
        """Each compared field on its own, the integer ``n`` included: dropping any one of them
        from the comparison is caught."""
        _canonical_is(env, tmp_path, "niamey", _moved(field, delta))

        _prepare(env, "niamey")

        parity = _parity_of(env, "niamey")
        assert parity["status"] == "differs"
        assert {k: list(v) for k, v in parity["differing_keys"].items()} == {LABEL: [field]}
        assert parity["unique_keys"] == {"staged": 72, "canonical": 72}
        assert f"differ on {field}: {LABEL}" in parity["reasons"][0]
        assert _run(env, "promote", "niamey") == 2

    def test_a_duplicated_all_days_row_in_canonical_differs_even_with_identical_values(
        self, env, tmp_path, capsys
    ):
        """72 unique keys and no value differs: only the duplicate says no, because
        `climate_normals` would serve two rows for that month."""
        _canonical_is(
            env, tmp_path, "niamey", f"SELECT * FROM t UNION ALL SELECT * FROM t WHERE {_ON_KEY}"
        )

        _prepare(env, "niamey")

        parity = _parity_of(env, "niamey")
        assert parity["status"] == "differs"
        assert parity["duplicated_keys"] == {"staged": [], "canonical": [LABEL]}
        assert parity["unique_keys"] == {"staged": 72, "canonical": 72}
        assert parity["differing_keys"] == {}
        assert parity["only_in_staged"] == parity["only_in_canonical"] == []
        env.fake.calls.clear()
        capsys.readouterr()
        assert _run(env, "promote", "niamey") == 2
        assert "parity is not exact for ['niamey']" in capsys.readouterr().err
        assert env.fake.puts() == []

    def test_a_key_missing_from_canonical_differs(self, env, tmp_path):
        _canonical_is(env, tmp_path, "niamey", f"SELECT * FROM t WHERE NOT ({_ON_KEY})")

        _prepare(env, "niamey")

        parity = _parity_of(env, "niamey")
        assert parity["status"] == "differs"
        assert parity["only_in_staged"] == [LABEL]
        assert parity["unique_keys"] == {"staged": 72, "canonical": 71}
        assert _run(env, "promote", "niamey") == 2

    def test_a_key_on_one_side_only_differs_even_with_72_keys_on_each(self, env, tmp_path):
        """Month 7 relabelled 13 in canonical: 72 unique keys on each side, no duplicate, no
        paired value differs, and still two keys are unpaired — each side's own guard."""
        _canonical_is(
            env,
            tmp_path,
            "niamey",
            f"SELECT * REPLACE (CASE WHEN {_ON_KEY} THEN 13 ELSE month END AS month) FROM t",
        )

        _prepare(env, "niamey")

        parity = _parity_of(env, "niamey")
        assert parity["status"] == "differs"
        assert parity["unique_keys"] == {"staged": 72, "canonical": 72}
        assert parity["only_in_staged"] == [LABEL]
        assert parity["only_in_canonical"] == [f"{KEY[0]}/13"]
        assert parity["duplicated_keys"] == {"staged": [], "canonical": []}
        assert parity["differing_keys"] == {}
        assert [r.split(":")[0] for r in parity["reasons"]] == [
            "1 key(s) only in staged",
            "1 key(s) only in canonical",
        ]

    def test_72_unique_keys_are_required_on_each_side(self, env, tmp_path):
        """The same key missing on BOTH sides leaves nothing unpaired and nothing different: only
        the count says no. Staged cannot lose a key through materialize, so the comparison is
        called on the two files directly."""
        _prepare(env, "niamey")
        drop = f"SELECT * FROM t WHERE NOT ({_ON_KEY})"
        staged = _rewritten(tmp_path, _staged(env, "niamey"), "staged", drop)
        canonical = _rewritten(tmp_path, env.seed["niamey"]["v1"], "canonical", drop)

        parity = op._percentile_comparison(staged, canonical, zone_id="niamey")["parity"]

        assert parity["status"] == "differs"
        assert parity["unique_keys"] == {"staged": 71, "canonical": 71}
        assert parity["only_in_staged"] == parity["only_in_canonical"] == []
        assert parity["differing_keys"] == {}
        assert parity["reasons"] == [
            "staged has 71 unique (variable, month) keys, not 72",
            "canonical has 71 unique (variable, month) keys, not 72",
        ]

    @pytest.mark.parametrize("staged_null, expected", [(True, "exact"), (False, "differs")])
    def test_null_equals_null_and_nothing_else(self, env, tmp_path, staged_null, expected):
        """IS NOT DISTINCT FROM: a mean that is NULL on both sides is equal, and NULL against a
        value is a difference. With ``=`` the first would be a STOP and the second unknown."""
        _prepare(env, "niamey")
        null_mean = f"SELECT * REPLACE (CASE WHEN {_ON_KEY} THEN NULL ELSE mean END AS mean) FROM t"
        staged = _rewritten(
            tmp_path,
            _staged(env, "niamey"),
            "staged",
            null_mean if staged_null else "SELECT * FROM t",
        )
        canonical = _rewritten(tmp_path, env.seed["niamey"]["v1"], "canonical", null_mean)

        parity = op._percentile_comparison(staged, canonical, zone_id="niamey")["parity"]

        assert parity["status"] == expected
        assert parity["exact"] is (expected == "exact")
        if expected == "differs":
            assert parity["differing_keys"][LABEL]["mean"]["canonical"] is None

    def test_an_unreadable_canonical_differs_rather_than_not_applicable(self, env, capsys):
        """Not a parquet at all: the older p10/p50/p90 report already fails, and the verdict is
        its refusal (the first of the comparison's two guards)."""
        env.fake.objects[op._CANONICAL.format(zone_id="niamey")] = b"not a parquet"

        _prepare(env, "niamey")

        parity = _parity_of(env, "niamey")
        assert parity["status"] == "differs"
        assert parity["exact"] is False
        assert parity["error"] == "canonical is unreadable as a normals parquet"
        env.fake.calls.clear()
        capsys.readouterr()
        assert _run(env, "promote", "niamey") == 2
        assert "parity is not exact for ['niamey']" in capsys.readouterr().err
        assert env.fake.puts() == []

    @pytest.mark.parametrize("column", ["mean", "n", "zone_id"])
    def test_a_canonical_the_older_report_reads_but_parity_cannot_differs(
        self, env, tmp_path, capsys, column
    ):
        """A canonical with p10/p50/p90 and without ``mean``, ``n`` or ``zone_id``: the older
        report reads it, and the parity query cannot (the second guard). That is a STOP, never an
        exact verdict by default: the fields the rule compares are not there to compare."""
        _canonical_is(env, tmp_path, "niamey", f"SELECT * EXCLUDE ({column}) FROM t")

        _prepare(env, "niamey")

        legacy = _plan(env)["zones"]["niamey"]["percentiles_all_days"]
        assert "error" not in legacy and legacy["pairs"] == 72  # past the first guard
        parity = _parity_of(env, "niamey")
        assert parity["status"] == "differs"
        assert parity["exact"] is False
        assert "are unreadable on one side" in parity["error"]
        env.fake.calls.clear()
        capsys.readouterr()
        assert _run(env, "promote", "niamey") == 2
        assert "parity is not exact for ['niamey']" in capsys.readouterr().err
        assert env.fake.puts() == []

    def test_a_duplicated_staged_key_differs_even_with_identical_values(self, env, tmp_path):
        """Materialize cannot write a duplicated key, so the comparison is called on the files
        directly: each side's duplicate is a reason on its own, staged's as much as canonical's."""
        staged = _rewritten(
            tmp_path,
            env.seed["niamey"]["v1"],
            "staged-dup",
            f"SELECT * FROM t UNION ALL SELECT * FROM t WHERE {_ON_KEY}",
        )
        canonical = tmp_path / "canonical.parquet"
        canonical.write_bytes(env.seed["niamey"]["v1"])

        parity = op._percentile_comparison(staged, canonical, zone_id="niamey")["parity"]

        assert parity["duplicated_keys"] == {"staged": [LABEL], "canonical": []}
        assert parity["unique_keys"] == {"staged": 72, "canonical": 72}
        assert parity["differing_keys"] == {}
        assert parity["status"] == "differs"
        assert parity["exact"] is False
        assert parity["reasons"] == [f"1 key(s) duplicated in staged: {LABEL}"]

    def test_no_canonical_is_not_applicable_and_promote_proceeds(self, env, capsys):
        del env.fake.objects[op._CANONICAL.format(zone_id="niamey")]

        _prepare(env, *ZONES)

        out = capsys.readouterr().out
        parity = _parity_of(env, "niamey")
        assert parity["status"] == "not_applicable"
        assert parity["exact"] is False
        assert _plan(env)["zones"]["niamey"]["percentiles_all_days"] is None
        assert _parity_of(env, "fos_sur_mer")["status"] == "exact"
        assert re.search(r"^  not applicable\s+niamey\s+no canonical before this run$", out, re.M)

        assert _run(env, "promote", *ZONES) == 0

        assert _canonical(env, "niamey") == _staged(env, "niamey")

    @pytest.mark.parametrize(
        "verdict",
        [
            None,  # a plan made before the rule
            {"rule": "exact", "status": "not_applicable", "exact": False},
            {"rule": "exact", "status": "exact", "exact": False},
            {"rule": "exact", "status": "not_applicable", "exact": True},
        ],
        ids=[
            "none",
            "not_applicable-with-a-canonical",
            "exact-status-with-exact-false",
            "not_applicable-with-exact-true",
        ],
    )
    def test_a_zone_with_a_canonical_and_no_exact_verdict_is_refused(self, env, capsys, verdict):
        _prepare(env, *ZONES)
        path = env.root / op._PLAN.format(run_id="r1")
        plan = json.loads(path.read_text())
        if verdict is None:
            del plan["zones"]["fos_sur_mer"]["parity"]
        else:
            plan["zones"]["fos_sur_mer"]["parity"] = verdict
        path.write_text(json.dumps(plan))
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 2

        err = capsys.readouterr().err
        assert "no exact parity verdict for ['fos_sur_mer']" in err
        assert "re-run plan" in err
        assert env.fake.puts() == []

    @pytest.mark.parametrize(
        "field, delta, forged",
        [("p50", 1e-4, "minimal"), ("n", 1, "minimal"), ("p50", 1e-4, "copied")],
        ids=["p50-minimal", "n-minimal", "p50-copied-from-an-exact-zone"],
    )
    def test_a_forged_exact_verdict_is_refused_on_the_verdict_promote_recomputes(
        self, env, tmp_path, capsys, field, delta, forged
    ):
        """plan.json is local and can be edited, so promote does not take its verdict on trust.
        It recomputes the verdict on the staged bytes it writes and the canonical they replace,
        read again from the store, as it re-checks the canonical's etag and sha256.

        niamey's plan says `differs`, and its parity block is overwritten with an exact one: the
        minimal one the plan-side check reads, or fos_sur_mer's genuine 72/72 block. Without the
        recompute, promote wrote v2 over the non-exact canonical and exited 0."""
        _canonical_is(env, tmp_path, "niamey", _moved(field, delta))
        before = _canonical(env, "niamey")
        _prepare(env, *ZONES)
        assert _parity_of(env, "niamey")["status"] == "differs"
        path = env.root / op._PLAN.format(run_id="r1")
        plan = json.loads(path.read_text())
        plan["zones"]["niamey"]["parity"] = (
            {"rule": "exact", "status": "exact", "exact": True}
            if forged == "minimal"
            else plan["zones"]["fos_sur_mer"]["parity"]
        )
        path.write_text(json.dumps(plan))
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 2

        err = capsys.readouterr().err
        assert "parity is not exact for ['niamey']" in err
        assert "the plan's exact verdict for ['niamey'] is not the one promote recomputes" in err
        assert "Do not edit plan.json" in err
        assert env.fake.puts() == []
        assert _canonical(env, "niamey") == before
        assert _canonical(env, "fos_sur_mer") == env.seed["fos_sur_mer"]["v1"]

    def test_the_recompute_reads_canonical_from_the_store_not_from_the_work_root(
        self, env, tmp_path, capsys
    ):
        """The canonical readback plan left in the work root is replaced with the unmoved v1,
        next to a forged verdict. The recompute reads canonical again from the store, so the
        edited readback does not make the moved canonical look exact."""
        _canonical_is(env, tmp_path, "niamey", _moved("p50", 1e-4))
        before = _canonical(env, "niamey")
        _prepare(env, "niamey")
        path = env.root / op._PLAN.format(run_id="r1")
        plan = json.loads(path.read_text())
        readback = env.root / op._READBACK_CANONICAL.format(
            run_id="r1", zone_id="niamey", sha256=plan["zones"]["niamey"]["canonical"]["sha256"]
        )
        assert readback.read_bytes() == before
        readback.write_bytes(env.seed["niamey"]["v1"])
        plan["zones"]["niamey"]["parity"] = {"rule": "exact", "status": "exact", "exact": True}
        path.write_text(json.dumps(plan))
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", "niamey") == 2

        assert "parity is not exact for ['niamey']" in capsys.readouterr().err
        assert env.fake.puts() == []
        assert _canonical(env, "niamey") == before

    def test_the_recompute_is_on_the_canonical_the_plan_saw_and_refuses_before_any_write(
        self, env, capsys
    ):
        """The canonical promote recomputes against must carry the sha256 the plan recorded, and
        that is checked for every zone before the first write. The plan's sha256 for fos_sur_mer
        (promoted second) is edited: before, niamey was written, then unwound, when fos's turn
        came."""
        _prepare(env, *ZONES)
        path = env.root / op._PLAN.format(run_id="r1")
        plan = json.loads(path.read_text())
        plan["zones"]["fos_sur_mer"]["canonical"]["sha256"] = "0" * 64
        path.write_text(json.dumps(plan))
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 2

        assert f"canonical for 'fos_sur_mer' changed since plan (sha256 {'0' * 64}" in (
            capsys.readouterr().err
        )
        assert env.fake.puts() == []

    def test_each_promote_event_records_the_verdict_promote_recomputed_and_the_plan(self, env):
        """The zone journal says which verdict each promotion relied on, the one promote
        recomputed, and which plan.json it acted on, so a promotion over a canonical that was
        not exact cannot pass unseen."""
        del env.fake.objects[op._CANONICAL.format(zone_id="niamey")]
        _prepare(env, *ZONES)
        plan_path = env.root / op._PLAN.format(run_id="r1")
        plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

        assert _run(env, "promote", *ZONES) == 0

        fos, niamey = _journal(env, "fos_sur_mer")[-1], _journal(env, "niamey")[-1]
        assert fos["kind"] == niamey["kind"] == "promote"
        assert fos["parity"] == {
            "rule": "exact",
            "status": "exact",
            "exact": True,
            "identical_keys": 72,
        }
        assert niamey["parity"] == {
            "rule": "exact",
            "status": "not_applicable",
            "exact": False,
            "identical_keys": None,
        }
        assert fos["plan_sha256"] == niamey["plan_sha256"] == plan_sha

    def test_a_verdict_on_staged_bytes_promote_does_not_write_is_refused(
        self, env, tmp_path, capsys
    ):
        """The verdict is bound to the staged bytes it judged, as it is to the canonical's etag.

        Canonical's p50 is moved by 1e-4, so the recorded v2 differs from it. A writer outside
        the protocol swaps the staging object for a lookalike with the same move, which plan
        judges exact, then puts the recorded bytes back before promote. Promote's own check of the
        staging object against the run record then passes, and without this binding it would
        write bytes whose parity against the canonical they replace is `differs`."""
        _canonical_is(env, tmp_path, "niamey", _moved("p50", 1e-4))
        before = _canonical(env, "niamey")
        for step in ("fetch", "materialize", "publish"):
            assert _run(env, step, "niamey") == 0
        key = op._STAGING.format(run_id="r1", zone_id="niamey")
        recorded = env.fake.objects[key]
        lookalike = _rewritten(tmp_path, recorded, "lookalike", _moved("p50", 1e-4))
        env.fake.objects[key] = lookalike.read_bytes()
        capsys.readouterr()

        assert _run(env, "plan", "niamey") == 0

        captured = capsys.readouterr()
        zone = _plan(env)["zones"]["niamey"]
        assert zone["parity"]["status"] == "exact"  # about the lookalike
        assert zone["staged"]["matches_record"] is False
        assert re.search(
            r"^  NOT RECORDED\s+niamey\s+STOP: plan read staged sha256", captured.out, re.M
        )
        assert not re.search(r"^  exact\s+niamey", captured.out, re.M)
        stop = (
            "STOP: the staged object plan read for ['niamey'] is not the one the run record holds"
        )
        assert stop in captured.out
        assert stop in captured.err
        env.fake.objects[key] = recorded
        # The restored bytes are the recorded ones, and they are not exact against canonical.
        restored = env.root / "restored.parquet"
        restored.write_bytes(recorded)
        current = env.root / "current.parquet"
        current.write_bytes(before)
        assert (
            op._percentile_comparison(restored, current, zone_id="niamey")["parity"]["status"]
            == "differs"
        )
        env.fake.calls.clear()

        assert _run(env, "promote", "niamey") == 2

        err = capsys.readouterr().err
        assert (
            "the plan judged staged bytes other than the ones promote writes for ['niamey']" in err
        )
        assert "re-run plan" in err
        assert env.fake.puts() == []
        assert _canonical(env, "niamey") == before

    @pytest.mark.parametrize(
        "field, value",
        [
            ("sha256", "0" * 64),
            ("recorded_sha256", "0" * 64),
            ("matches_record", False),
            ("matches_record", None),
        ],
        ids=["sha256", "recorded_sha256", "matches_record-false", "matches_record-missing"],
    )
    def test_a_plan_whose_staged_block_is_not_the_run_records_is_refused(
        self, env, capsys, field, value
    ):
        """Each part of the binding on its own, on a plan edited by hand: the sha256 plan read,
        the one it recorded, and its own match flag must all say the run record's bytes."""
        _prepare(env, *ZONES)
        path = env.root / op._PLAN.format(run_id="r1")
        plan = json.loads(path.read_text())
        staged = plan["zones"]["fos_sur_mer"]["staged"]
        if value is None:
            del staged[field]
        else:
            staged[field] = value
        path.write_text(json.dumps(plan))
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 2

        err = capsys.readouterr().err
        assert "staged bytes other than the ones promote writes for ['fos_sur_mer']" in err
        assert env.fake.puts() == []

    def test_a_zone_this_run_already_promoted_is_exempt_when_the_promotion_resumes(self, env):
        """niamey is promoted and its event written, then the process dies on fos's rollback copy
        (a KeyboardInterrupt is not an Exception, so nothing unwinds). The resume runs on the plan
        it started from, where niamey's canonical already holds this run's v2. Its verdict is
        replaced with `differs` so that only the exemption can let the resume through; fos, not
        promoted yet, is still judged."""
        _prepare(env, *ZONES)
        fos_kept = op._ROLLBACK.format(run_id="r1", zone_id="fos_sur_mer")

        def _die(key, body):
            if key == fos_kept:
                raise KeyboardInterrupt
            return body

        env.fake.on_put = _die
        with pytest.raises(KeyboardInterrupt):
            _run(env, "promote", *ZONES)
        env.fake.on_put = None
        assert [e["kind"] for e in _journal(env, "niamey")] == ["promote"]
        assert _journal(env, "fos_sur_mer") == []
        path = env.root / op._PLAN.format(run_id="r1")
        plan = json.loads(path.read_text())
        plan["zones"]["niamey"]["parity"] = {"rule": "exact", "status": "differs", "exact": False}
        path.write_text(json.dumps(plan))

        assert _run(env, "promote", *ZONES) == 0

        for zone_id in ZONES:
            assert _canonical(env, zone_id) == _staged(env, zone_id)

    def test_an_exact_verdict_whose_details_were_edited_is_refused(self, env, capsys):
        """Parity really is exact, but the reviewed report is not the one promote recomputes: its
        key count was edited. The report the maintainer reviewed must be the one the rule is
        enforced on, field for field (Greptile P2, PR #225)."""
        _prepare(env, "niamey")
        assert _parity_of(env, "niamey")["status"] == "exact"
        path = env.root / op._PLAN.format(run_id="r1")
        plan = json.loads(path.read_text())
        plan["zones"]["niamey"]["parity"]["identical_keys"] = 71
        path.write_text(json.dumps(plan))
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", "niamey") == 2

        assert "is not the one promote recomputes" in capsys.readouterr().err
        assert env.fake.puts() == []

    @pytest.mark.parametrize(
        "parity",
        [
            None,
            {"rule": "exact", "status": "differs", "exact": False},
            # not_applicable is evidence only where there was no canonical before.
            {"rule": "exact", "status": "not_applicable", "exact": False},
        ],
        ids=["no-verdict", "differs", "not-applicable-over-a-canonical"],
    )
    def test_a_resumed_zone_whose_promote_event_lacks_an_exact_verdict_is_refused(
        self, env, capsys, parity
    ):
        """A promote event written by code older than the rule carries no verdict, so a resume
        cannot rely on the zone having been judged when it was promoted (Codex P2, PR #225)."""
        _prepare(env, *ZONES)
        fos_kept = op._ROLLBACK.format(run_id="r1", zone_id="fos_sur_mer")

        def _die(key, body):
            if key == fos_kept:
                raise KeyboardInterrupt
            return body

        env.fake.on_put = _die
        with pytest.raises(KeyboardInterrupt):
            _run(env, "promote", *ZONES)
        env.fake.on_put = None
        key = op._ZONE_EVENT.format(zone_id="niamey", seq=1)
        event = json.loads(env.fake.objects[key])
        assert event["canonical_sha256_before"] is not None
        event.pop("parity")
        if parity is not None:
            event["parity"] = parity
        env.fake.objects[key] = op._record_bytes(event)
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 2

        assert "with no parity verdict in their promote event" in capsys.readouterr().err
        assert env.fake.puts() == []

    def test_a_resumed_zone_that_had_no_canonical_is_exempt(self, env):
        """niamey had no canonical, so its promote event records `not_applicable` with no
        canonical before: that is the evidence the rule asks of it, and the resume goes on."""
        del env.fake.objects[op._CANONICAL.format(zone_id="niamey")]
        _prepare(env, *ZONES)
        fos_kept = op._ROLLBACK.format(run_id="r1", zone_id="fos_sur_mer")

        def _die(key, body):
            if key == fos_kept:
                raise KeyboardInterrupt
            return body

        env.fake.on_put = _die
        with pytest.raises(KeyboardInterrupt):
            _run(env, "promote", *ZONES)
        env.fake.on_put = None
        (event,) = _journal(env, "niamey")
        assert event["parity"]["status"] == "not_applicable"
        assert event["canonical_sha256_before"] is None

        assert _run(env, "promote", *ZONES) == 0

        for zone_id in ZONES:
            assert _canonical(env, zone_id) == _staged(env, zone_id)

    def test_the_refusal_names_every_zone_whose_parity_differs(self, env, tmp_path, capsys):
        for zone_id in ZONES:
            _canonical_is(env, tmp_path, zone_id, _moved("p50", 1e-4))
        _prepare(env, *ZONES)
        env.fake.calls.clear()
        capsys.readouterr()

        assert _run(env, "promote", *ZONES) == 2

        assert f"parity is not exact for {list(ZONES)}" in capsys.readouterr().err
        assert env.fake.puts() == []


def test_the_local_mode_is_the_default_and_constructs_no_client(env, tmp_path):
    """No `--store` means the local mode, whose own tests are unchanged."""
    root = Path(tmp_path)
    code = op.main(["rollback", "--root", root.as_posix(), "--run-id", "r1", "--zone-id", "niamey"])

    assert code == 2  # no local manifest: the local refusal, not a store call
    assert env.clients == 0
