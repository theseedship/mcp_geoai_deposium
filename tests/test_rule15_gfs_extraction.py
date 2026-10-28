"""Hermetic tests for scripts/rule15_gfs_extraction.py — no network."""

from __future__ import annotations

import pytest

from scripts import rule15_gfs_extraction as gfs

IDX = (
    "580:423500000:d=2023063000:SPFH:2 m above ground:48 hour fcst:\n"
    "581:424031287:d=2023063000:TMP:2 m above ground:48 hour fcst:\n"
    "582:424550067:d=2023063000:SPFH:2 m above ground:48 hour fcst:\n"
    "585:427099350:d=2023063000:APTMP:2 m above ground:48 hour fcst:\n"
)


def test_leads_cover_three_hourly_samples_of_1_to_5_july():
    assert len(gfs.LEADS) == 40
    assert (gfs.LEADS[0], gfs.LEADS[-1]) == (24, 141)
    assert gfs.object_url(24).endswith("/gfs.20230630/00/atmos/gfs.t00z.pgrb2.0p25.f024")


def test_message_range_selects_only_2m_temperature():
    line, start, end = gfs.message_range(IDX, 500_000_000)
    assert ":TMP:2 m above ground:" in line and ":APTMP:" not in line
    assert (start, end) == (424031287, 424550066)


def test_message_range_last_line_ends_at_object_size():
    line, start, end = gfs.message_range(IDX.splitlines()[1], 424_100_000)
    assert (start, end) == (424031287, 424099999)


@pytest.mark.parametrize("idx", ["", IDX + IDX.splitlines()[1] + "\n"])
def test_message_range_refuses_missing_or_ambiguous_field(idx):
    with pytest.raises(ValueError, match="exactly one"):
        gfs.message_range(idx, 500_000_000)


def samples(missing=()):
    return [
        {
            "valid_time": f"2023-07-{1 + (lead - 24) // 24:02d}T{(lead % 24):02d}:00:00Z",
            "celsius": None if lead in missing else float(lead),
        }
        for lead in gfs.LEADS
    ]


def test_daily_mean_of_eight_samples():
    days = gfs.daily_means(samples())
    assert [d["date"] for d in days] == [f"2023-07-0{i}" for i in range(1, 6)]
    assert days[0]["temperature_2m"] == sum(range(24, 48, 3)) / 8
    assert all(d["samples_present"] == 8 for d in days)


def test_missing_sample_nulls_its_day_only():
    days = gfs.daily_means(samples(missing={51}))
    assert days[1]["temperature_2m"] is None and days[1]["samples_present"] == 7
    assert all(d["temperature_2m"] is not None for i, d in enumerate(days) if i != 1)


class CountingBody:
    """Make iteration and closure observable without network access."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.reads = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    def close(self):
        self.closed = True


def extraction_response(
    tmp_path,
    monkeypatch,
    *,
    status=206,
    headers=None,
    chunks=(b"0123456789",),
    idx_chunks=None,
    idx_headers=None,
    head_headers=None,
):
    import httpx

    class Stream(CountingBody, httpx.SyncByteStream):
        pass

    message = Stream(chunks)
    index = Stream(
        idx_chunks
        if idx_chunks is not None
        else [
            b"1:0:d=2023063000:TMP:2 m above ground:24 hour fcst:\n"
            b"2:10:d=2023063000:SPFH:2 m above ground:24 hour fcst:\n"
        ]
    )
    requested = []

    def handler(request):
        requested.append(request)
        if request.method == "HEAD":
            return httpx.Response(
                200,
                headers=head_headers
                if head_headers is not None
                else {"content-length": "100", "etag": '"object"'},
            )
        if str(request.url).endswith(".idx"):
            return httpx.Response(200, headers=idx_headers, stream=index)
        return httpx.Response(
            status,
            headers=headers
            if headers is not None
            else {"content-range": "bytes 0-9/100", "etag": '"object"', "content-length": "10"},
            stream=message,
        )

    (tmp_path / "raw/idx").mkdir(parents=True)
    (tmp_path / "raw/messages").mkdir()
    monkeypatch.setattr(
        gfs,
        "_read_point",
        lambda path: {
            "tags": {
                "GRIB_REF_TIME": "1688083200",
                "GRIB_VALID_TIME": "1688169600",
                "GRIB_ELEMENT": "TMP",
                "GRIB_SHORT_NAME": "2-HTGL",
                "GRIB_UNIT": "[K]",
            },
            "grid": {"lat": 43.5, "lon": 5.0},
            "kelvin": 293.15,
        },
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        sample = gfs.extract_lead(client, 24, tmp_path)
    return sample, message, index, requested


def test_valid_streamed_range_preserves_extraction(tmp_path, monkeypatch):
    sample, body, index, requests = extraction_response(tmp_path, monkeypatch)
    assert sample["status"] == "ok" and sample["celsius"] == 20.0
    assert (tmp_path / sample["message"]["path"]).read_bytes() == b"0123456789"
    assert requests[-1].headers["if-match"] == '"object"'
    assert requests[-1].headers["range"] == "bytes=0-9"
    assert body.closed and index.closed


@pytest.mark.parametrize(
    "status,headers",
    [
        (200, {}),
        (206, {"content-range": "bytes 1-10/100", "etag": '"object"'}),
        (206, {"content-range": "bytes 0-9/100", "etag": '"other"'}),
        (206, {"content-range": "bytes 0-9/100", "etag": '"object"', "content-length": "11"}),
    ],
)
def test_rejects_invalid_range_headers_without_reading(tmp_path, monkeypatch, status, headers):
    sample, body, _, _ = extraction_response(tmp_path, monkeypatch, status=status, headers=headers)
    assert sample["status"] == "missing"
    assert body.reads == 0 and body.closed
    assert not list((tmp_path / "raw/messages").iterdir())


def test_range_without_length_stops_at_first_excess_byte(tmp_path, monkeypatch):
    sample, body, _, _ = extraction_response(
        tmp_path,
        monkeypatch,
        headers={"content-range": "bytes 0-9/100", "etag": '"object"'},
        chunks=[b"x"] * 100,
    )
    assert sample["status"] == "missing" and "byte limit" in sample["error"]
    assert body.reads == 11 and body.closed


@pytest.mark.parametrize("declared", [True, False])
def test_idx_is_bounded_before_range_request(tmp_path, monkeypatch, declared):
    monkeypatch.setattr(gfs, "MAX_IDX_BYTES", 10)
    sample, body, index, requests = extraction_response(
        tmp_path,
        monkeypatch,
        idx_chunks=[b"x"] * 100,
        idx_headers={"content-length": "100"} if declared else {},
    )
    assert sample["status"] == "missing"
    assert index.reads == (0 if declared else 11) and index.closed
    assert len(requests) == 2 and body.reads == 0


@pytest.mark.parametrize(
    "headers",
    [
        {"content-length": "100"},
        {"content-length": "0", "etag": '"object"'},
        {"content-length": str(gfs.MAX_OBJECT_BYTES + 1), "etag": '"object"'},
    ],
)
def test_invalid_head_refuses_gets(tmp_path, monkeypatch, headers):
    sample, _, _, requests = extraction_response(tmp_path, monkeypatch, head_headers=headers)
    assert sample["status"] == "missing" and len(requests) == 1


def test_oversize_idx_range_refused_before_message_get(tmp_path, monkeypatch):
    monkeypatch.setattr(gfs, "MAX_MESSAGE_BYTES", 9)
    sample, body, _, requests = extraction_response(tmp_path, monkeypatch)
    assert sample["status"] == "missing" and "range exceeds" in sample["error"]
    assert len(requests) == 2 and body.reads == 0


def test_short_announced_range_length_rejected_without_reading(tmp_path, monkeypatch):
    sample, body, _, _ = extraction_response(
        tmp_path,
        monkeypatch,
        headers={"content-range": "bytes 0-9/100", "etag": '"object"', "content-length": "9"},
    )
    assert sample["status"] == "missing"
    assert body.reads == 0 and body.closed
