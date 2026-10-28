import hashlib
import zipfile

import pandas as pd
import pytest

from scripts import o3_prepare_pilots as prepare


def test_archive_hash_is_checked_before_zip_read(tmp_path):
    source = tmp_path / "input.zip"
    source.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare.read_archive(source, "0" * 64)


def test_archive_member_traversal_rejected(tmp_path):
    source = tmp_path / "input.zip"
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("../escape", "invalid")
    with pytest.raises(ValueError, match="member"):
        prepare.read_archive(source, hashlib.sha256(source.read_bytes()).hexdigest())


def test_alignment_cannot_cross_temporal_donor_boundary():
    era5 = pd.DataFrame(
        {"date": pd.to_datetime(["2023-01-01"]), "zone_id": ["pilot"], "temperature_2m": [8.0]}
    )
    ndvi = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-12-31", "2023-01-01"]),
            "zone_id": ["pilot", "pilot"],
            "ndvi_mean": [0.9, 0.5],
        }
    )
    result = prepare.align(era5, ndvi, "pilot", 3)
    assert len(result) == 1 and result.iloc[0]["ndvi_mean"] == 0.5
    assert result.iloc[0]["day_delta"] == 0


def test_alignment_nearest_and_collision_policy():
    era5 = pd.DataFrame(
        {"date": pd.to_datetime(["2021-01-01"]), "zone_id": ["pilot"], "temperature_2m": [8.0]}
    )
    ndvi = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "zone_id": ["pilot", "pilot"],
            "ndvi_mean": [0.2, 0.9],
        }
    )
    result = prepare.align(era5, ndvi, "pilot", 3)
    assert len(result) == 1 and result.iloc[0]["ndvi_mean"] == 0.2
    assert result.iloc[0]["day_delta"] == 0
