#!/usr/bin/env python3
"""Extract the rule15 historical Fos temperature forecast from one archived NOAA GFS cycle.

Decision: https://github.com/theseedship/deposium_geoai/issues/30#issuecomment-6424428459

Single cycle 2023-06-30 00 UTC, field ``TMP:2 m above ground``, three-hourly leads f024..f141
(00..21 UTC on 1-5 July 2023), the GFS cell containing the historical Fos point. For every
lead it keeps the raw ``.idx``, the one GRIB2 message fetched by byte range, their SHA-256
digests and three separate timestamps (model initialization read from the GRIB message, S3
``Last-Modified``, retrieval time). Each day is the mean of its eight instantaneous samples; a
day with a missing sample stays null. Humidity and ET0 stay null: nothing is imputed.

    uv run --no-sync python scripts/rule15_gfs_extraction.py --output-dir NEW_DIR

The output directory is create-only. ``forecast.json`` is the saved forecast for
``b6_historical_case preflight --forecast-profile rule15-temperature``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.patterns.historical_case import ZONES_SHA256  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

BUCKET_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
CYCLE = datetime(2023, 6, 30, 0, tzinfo=UTC)
LEADS = tuple(range(24, 142, 3))  # f024..f141: 00..21 UTC on 1-5 July 2023
FIELD = ":TMP:2 m above ground:"
LAT, LON = 43.445, 4.92
KELVIN_OFFSET = 273.15
MAX_IDX_BYTES = 1024 * 1024
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_OBJECT_BYTES = 2 * 1024 * 1024 * 1024


def object_url(lead: int) -> str:
    return (
        f"{BUCKET_URL}/gfs.{CYCLE:%Y%m%d}/{CYCLE:%H}/atmos/gfs.t{CYCLE:%H}z.pgrb2.0p25.f{lead:03d}"
    )


def message_range(idx_text: str, object_size: int) -> tuple[str, int, int]:
    """Return the matching idx line and the inclusive byte range of its GRIB message."""
    lines = [line for line in idx_text.splitlines() if line]
    hits = [i for i, line in enumerate(lines) if FIELD in line]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one {FIELD!r} idx line, found {len(hits)}")
    i = hits[0]
    start = int(lines[i].split(":")[1])
    end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else object_size - 1
    if not 0 <= start <= end < object_size:
        raise ValueError("idx byte range outside the object")
    return lines[i], start, end


def daily_means(samples: list[dict]) -> list[dict]:
    """Mean of the eight 3-hourly samples per UTC day; any missing sample makes the day null."""
    days = []
    for offset in range(5):
        day = (CYCLE + timedelta(days=1 + offset)).date()
        values = [s["celsius"] for s in samples if s["valid_time"][:10] == day.isoformat()]
        present = [v for v in values if v is not None]
        complete = len(values) == 8 and len(present) == 8
        days.append(
            {
                "date": day.isoformat(),
                "temperature_2m": round(sum(present) / 8, 4) if complete else None,
                "samples_expected": 8,
                "samples_present": len(present),
            }
        )
    return days


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_point(path: Path) -> dict:
    import rasterio

    # GDAL converts GRIB temperatures to Celsius by default; keep the source Kelvin instead.
    with rasterio.Env(GRIB_NORMALIZE_UNITS="NO"), rasterio.open(path) as ds:
        tags = ds.tags(1)
        row, col = ds.index(LON, LAT)  # cell containing the point = nearest 0.25° grid node
        grid_lon, grid_lat = ds.xy(row, col)
        kelvin = float(ds.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
    return {
        "tags": {k: tags.get(k) for k in sorted(tags) if k.startswith("GRIB_")},
        "grid": {"row": int(row), "col": int(col), "lat": float(grid_lat), "lon": float(grid_lon)},
        "kelvin": kelvin,
    }


def _bounded_body(response, limit: int) -> bytes:
    """Reject oversized declarations before reading; retain at most limit bytes."""
    declared = response.headers.get("content-length")
    if declared is not None and not 0 <= int(declared) <= limit:
        raise ValueError("response Content-Length exceeds byte limit")
    # One chunk exposes at most limit + 1 bytes before rejection, including when
    # Content-Length is absent or false. The context manager closes on every exit.
    chunks = response.iter_raw(chunk_size=limit + 1)
    body = next(chunks, b"")
    if len(body) > limit:
        raise ValueError("response body exceeds byte limit")
    if declared is not None and len(body) != int(declared):
        raise ValueError("response body does not match Content-Length")
    return body


def extract_lead(client, lead: int, out: Path) -> dict:
    url = object_url(lead)
    sample = {"lead_hours": lead, "valid_time": (CYCLE + timedelta(hours=lead)).isoformat()}
    sample["valid_time"] = sample["valid_time"].replace("+00:00", "Z")
    sample.update(url=url, idx_url=f"{url}.idx", celsius=None, status="missing")
    try:
        head = client.head(url)
        head.raise_for_status()
        size = int(head.headers["content-length"])
        if not 0 < size <= MAX_OBJECT_BYTES:
            raise ValueError("object Content-Length outside byte limit")
        etag = head.headers.get("etag")
        if not etag:
            raise ValueError("object HEAD requires ETag")
        sample["object"] = {
            "last_modified": head.headers.get("last-modified"),
            "etag": head.headers.get("etag"),
            "content_length": size,
            "retrieved_at": _now(),
        }
        with client.stream(
            "GET", sample["idx_url"], headers={"Accept-Encoding": "identity"}
        ) as idx:
            idx.raise_for_status()
            idx_raw = _bounded_body(idx, MAX_IDX_BYTES)
            idx_modified = idx.headers.get("last-modified")
        idx_path = out / "raw/idx" / f"f{lead:03d}.idx"
        idx_path.write_bytes(idx_raw)
        line, start, end = message_range(idx_raw.decode("utf-8"), size)
        sample["idx"] = {
            "path": idx_path.relative_to(out).as_posix(),
            "sha256": _sha(idx_raw),
            "bytes": len(idx_raw),
            "last_modified": idx_modified,
            "retrieved_at": _now(),
            "line": line,
        }
        expected = end - start + 1
        if expected > MAX_MESSAGE_BYTES:
            raise ValueError("idx message range exceeds byte limit")
        # If-Match ties the range to the object described by HEAD.
        with client.stream(
            "GET",
            url,
            headers={
                "Range": f"bytes={start}-{end}",
                "If-Match": etag,
                "Accept-Encoding": "identity",
            },
        ) as msg:
            if msg.status_code != 206:
                raise ValueError(f"range request returned {msg.status_code}")
            content_range = msg.headers.get("content-range")
            if content_range != f"bytes {start}-{end}/{size}":
                raise ValueError("range response Content-Range mismatch")
            message_etag = msg.headers.get("etag")
            if message_etag != etag:
                raise ValueError("range response ETag mismatch")
            declared = msg.headers.get("content-length")
            if declared is not None and int(declared) != expected:
                raise ValueError("range response Content-Length mismatch")
            msg_raw = _bounded_body(msg, expected)
            if len(msg_raw) != expected:
                raise ValueError("range response body length mismatch")
        msg_path = out / "raw/messages" / f"f{lead:03d}.grib2"
        msg_path.write_bytes(msg_raw)
        sample["message"] = {
            "path": msg_path.relative_to(out).as_posix(),
            "byte_range": [start, end],
            "bytes": len(msg_raw),
            "sha256": _sha(msg_raw),
            "digest_scope": "this retrieved GRIB2 message only, not the whole GRIB object",
            "etag": message_etag,
            "content_range": content_range,
            "retrieved_at": _now(),
        }
        point = _read_point(msg_path)
        tags = point["tags"]
        init = datetime.fromtimestamp(int(tags["GRIB_REF_TIME"]), UTC)
        valid = datetime.fromtimestamp(int(tags["GRIB_VALID_TIME"]), UTC)
        if init != CYCLE or valid != CYCLE + timedelta(hours=lead):
            raise ValueError(f"GRIB times {init}/{valid} do not match cycle/lead")
        if (tags.get("GRIB_ELEMENT"), tags.get("GRIB_SHORT_NAME"), tags.get("GRIB_UNIT")) != (
            "TMP",
            "2-HTGL",
            "[K]",
        ):
            raise ValueError(
                f"unexpected GRIB field {tags.get('GRIB_ELEMENT')} {tags.get('GRIB_UNIT')}"
            )
        sample.update(
            model_initialization=init.isoformat().replace("+00:00", "Z"),
            grib=tags,
            grid=point["grid"],
            kelvin=point["kelvin"],
            celsius=round(point["kelvin"] - KELVIN_OFFSET, 4),
            status="ok",
        )
    except Exception as exc:  # a failed lead stays visible as missing, never interpolated
        sample["error"] = f"{type(exc).__name__}: {exc}"
    return sample


def _environment() -> dict:
    import httpx
    import rasterio

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    return {
        "code_revision": git("rev-parse", "HEAD"),
        "worktree_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rasterio": rasterio.__version__,
        "gdal": rasterio.__gdal_version__,
        "httpx": httpx.__version__,
        "gdal_options": {"GRIB_NORMALIZE_UNITS": "NO"},
    }


def run(out: Path) -> dict:
    import httpx

    out.mkdir(parents=True)  # create-only: never overwrite retained evidence
    (out / "raw/idx").mkdir(parents=True)
    (out / "raw/messages").mkdir(parents=True)
    started = _now()
    with httpx.Client(timeout=60, follow_redirects=False) as client:
        samples = [extract_lead(client, lead, out) for lead in LEADS]
    days = daily_means(samples)
    grids = {json.dumps(s["grid"], sort_keys=True) for s in samples if s["status"] == "ok"}
    forecast = {
        "zone_id": "fos_sur_mer",
        "forecast": [
            {
                "date": d["date"],
                "temperature_2m": d["temperature_2m"],
                "humidity": None,
                "evapotranspiration": None,
            }
            for d in days
        ],
        "metadata": {
            "synthetic": False,
            "source": f"NOAA GFS 0.25 deg {CYCLE:%Y-%m-%d %H} UTC cycle, TMP 2 m above ground, "
            "noaa-gfs-bdp-pds; mean of eight 3-hourly samples per UTC day; see manifest.json",
            "model_issued_at": CYCLE.isoformat().replace("+00:00", "Z"),
            "issued_at": started,
            "lat": LAT,
            "lon": LON,
            "horizon_days": 5,
            "registry_sha256": ZONES_SHA256,
        },
    }
    (out / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    (out / "forecast.json").write_text(json.dumps(forecast, indent=2) + "\n")
    manifest = {
        "schema_version": "geoai-rule15-gfs-extraction/v1",
        "decision": "https://github.com/theseedship/deposium_geoai/issues/30#issuecomment-6424428459",
        "case_rule_id": "cross_zone/tgrad/15",
        "model_cycle": forecast["metadata"]["model_issued_at"],
        "field": {
            "idx_match": FIELD,
            "source_units": "K",
            "output_units": "degC",
            "conversion": f"celsius = kelvin - {KELVIN_OFFSET}",
        },
        "point": {"lat": LAT, "lon": LON, "registry_sha256": ZONES_SHA256},
        "grid": [json.loads(g) for g in sorted(grids)],
        "extraction_method": "rasterio index of the 0.25 deg GFS cell containing the point "
        "(its centre is the nearest grid node); no interpolation",
        "aggregation": "daily value = mean of the eight instantaneous samples at 00..21 UTC; "
        "declared aggregation, not an hourly daily mean; null if any sample is missing",
        "leads": list(LEADS),
        "samples_ok": sum(s["status"] == "ok" for s in samples),
        "samples_missing": [s["lead_hours"] for s in samples if s["status"] != "ok"],
        "daily": days,
        "timestamps": "model_initialization from GRIB_REF_TIME; object.last_modified from S3 HEAD; "
        "retrieved_at per request (UTC)",
        "retrieval_started_at": started,
        "retrieval_finished_at": _now(),
        "command": " ".join(sys.argv),
        "environment": _environment(),
        "files": {
            p.relative_to(out).as_posix(): _sha(p.read_bytes())
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    manifest = run(parser.parse_args(argv).output_dir)
    print(
        json.dumps({k: manifest[k] for k in ("samples_ok", "samples_missing", "daily")}, indent=2)
    )
    return 0 if not manifest["samples_missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
