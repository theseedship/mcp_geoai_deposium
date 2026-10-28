"""Network-free handler tests for change_detect edge cases."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.tools.change import register_change_tools


class _CaptureMCP:
    """Records tool handlers without a live FastMCP server."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *_args, **_kwargs):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


def _load_change_tools() -> dict:
    cap = _CaptureMCP()
    register_change_tools(cap)
    return cap.tools


@dataclass
class _Item:
    id: str
    date: str
    ndvi: float
    cloud_cover: float = 0.0

    @property
    def properties(self) -> dict:
        return {"eo:cloud_cover": self.cloud_cover}


def test_change_detect_rejects_selected_images_out_of_order(monkeypatch):
    def _fake_search(_bbox, datetime_str, _cloud_cover_max, _limit):
        if datetime_str.startswith("2023-12-02"):
            return [_Item("before", "2024-01-05T00:00:00+00:00", 0.8)]
        return [_Item("after", "2024-01-03T00:00:00+00:00", 0.2)]

    def _fake_best(items, _geometry):
        item = items[0]
        return (
            {
                "date": item.date,
                "ndvi_mean": item.ndvi,
                "ndvi_std": 0.0,
                "ndvi_min": item.ndvi,
                "ndvi_max": item.ndvi,
                "cloud_cover": item.cloud_cover,
                "item_id": item.id,
            },
            {"item_id": item.id, "date": item.date, "cloud_cover": item.cloud_cover},
        )

    # _search_sentinel2 now lives in (and is called through) src.tools._ndvi — its single
    # honest patch home; _find_best_ndvi is still defined in change.py.
    monkeypatch.setattr("src.tools._ndvi._search_sentinel2", _fake_search)
    monkeypatch.setattr("src.tools.change._find_best_ndvi", _fake_best)

    result = asyncio.run(
        _load_change_tools()["change_detect"](
            bbox=[0.0, 0.0, 0.1, 0.1],
            date_before="2024-01-01",
            date_after="2024-01-02",
        )
    )

    assert "error" in result
    assert "not earlier" in result["error"]
    assert result["metadata"]["before_image"]["item_id"] == "before"
    assert result["metadata"]["after_image"]["item_id"] == "after"
