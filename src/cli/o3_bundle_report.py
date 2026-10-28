"""Generate an offline report from a SHA-256-pinned O3 ZIP bundle."""

from __future__ import annotations

import argparse
import json
import sys

from src.patterns.offline_bundle import (
    OfflineBundleError,
    load_offline_bundle,
    write_bundle_reports,
)
from src.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="ZIP local geoai-o3-offline-handoff/v1")
    parser.add_argument("--expected-sha256", required=True, help="SHA-256 attendu de l'archive")
    parser.add_argument("--output-dir", required=True, help="Répertoire local des trois rapports")
    parser.add_argument(
        "--zones-geojson", help="GeoJSON local dont le SHA correspond à tous les manifests B4"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        report = load_offline_bundle(args.archive, args.expected_sha256, args.zones_geojson)
        paths = write_bundle_reports(report, args.output_dir)
    except OfflineBundleError as exc:
        print(
            json.dumps({"status": "refused", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": "written",
                "patterns": len(report["patterns"]),
                "archive_sha256": report["archive"]["sha256"],
                "files": [str(path) for path in paths],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
