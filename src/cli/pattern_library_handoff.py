"""Export, verify or project a D-A package without mining, forecasting or network calls."""

from __future__ import annotations

import argparse
import json
import sys

from src.patterns.library_handoff import export_library_handoff, verify_library_handoff
from src.patterns.scientific_projection import project_library_handoff
from src.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Authenticate sources and create a new package")
    export.add_argument("--config", required=True)
    export.add_argument("--output-dir", required=True)
    verify = commands.add_parser("verify", help="Verify an externally pinned package")
    verify.add_argument("directory")
    verify.add_argument("--expected-manifest-sha256", required=True)
    project = commands.add_parser(
        "project", help="Verify a package and write an offline MCP projection"
    )
    project.add_argument("directory")
    project.add_argument("--expected-manifest-sha256", required=True)
    project.add_argument("--package-id", required=True)
    project.add_argument("--output", required=True)
    project.add_argument("--package-version")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            digest = export_library_handoff(args.config, args.output_dir)
            print(json.dumps({"output_dir": args.output_dir, "manifest_sha256": digest}))
        elif args.command == "project":
            print(
                json.dumps(
                    project_library_handoff(
                        args.directory,
                        args.expected_manifest_sha256,
                        args.package_id,
                        args.output,
                        args.package_version,
                    )
                )
            )
        else:
            manifest, libraries = verify_library_handoff(
                args.directory, args.expected_manifest_sha256
            )
            print(
                json.dumps(
                    {
                        "library_version": manifest.library_version,
                        "manifest_sha256": args.expected_manifest_sha256,
                        "verified_files": len(manifest.files),
                        "libraries": len(libraries),
                        "patterns": sum(len(library.patterns) for library in libraries),
                        "areas": [row.model_dump() for row in manifest.areas],
                    }
                )
            )
        return 0
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
