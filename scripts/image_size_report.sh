#!/usr/bin/env bash
# Build the root Dockerfile natively for linux/arm64 and publish its sizes with provenance (#173).
#
#   scripts/image_size_report.sh [--no-cache]
#
# - Refuses a dirty working tree, and builds from `git archive HEAD`, so the report names
#   exactly what was built. Refuses a non-arm64 host or engine: an emulated build is never
#   reported as native.
# - Three size semantics, kept apart:
#     archive proxy  set -o pipefail; docker save --platform linux/arm64 IMAGE | gzip -1 -c | wc -c
#     local size     docker image inspect IMAGE --format '{{.Size}}'  (meaning depends on the store)
#     registry       not measured: nothing is pushed
# - Runs scripts/image_smoke.sh on the built image.
# - Writes report.json, report.md and *.log.txt evidence into
#   outputs/measurements/bf-image-size/<UTC date>-<12-char sha>-native-arm64/.
# No size threshold is applied: the report is evidence, not a gate.
set -euo pipefail

NO_CACHE=false
case "${1:-}" in
    "") ;;
    --no-cache) NO_CACHE=true ;;
    *) echo "usage: scripts/image_size_report.sh [--no-cache]" >&2; exit 2 ;;
esac

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

if [[ -n $(git status --porcelain --untracked-files=all) ]]; then
    echo "refusing: the working tree is dirty; commit or remove changes first" >&2
    git status --short >&2
    exit 1
fi

HOST_ARCH=$(uname -m)
ENGINE_ARCH=$(docker version --format '{{.Server.Arch}}')
ENGINE_OS=$(docker version --format '{{.Server.Os}}')
if [[ $HOST_ARCH != arm64 && $HOST_ARCH != aarch64 ]] || [[ $ENGINE_ARCH != arm64 ]]; then
    echo "refusing: host ($HOST_ARCH) and engine ($ENGINE_ARCH) must both be arm64" >&2
    exit 1
fi

COMMIT=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short=12 HEAD)
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# Date and commit both in the name: a later run never overwrites a committed report.
OUT="outputs/measurements/bf-image-size/$(date -u +%Y-%m-%d)-$SHORT-native-arm64"
IMAGE="geoai-bf-size:$SHORT"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
RAW="$OUT/.raw"
rm -rf "$RAW"
mkdir -p "$RAW"

SRC=$(mktemp -d)
cleanup() { rm -rf "$SRC" "$RAW"; }
trap cleanup EXIT
git archive "$COMMIT" | tar -x -C "$SRC"

# --- build ------------------------------------------------------------------------------ #
BUILD_ARGS=(build --platform linux/arm64 --progress=plain -f Dockerfile -t "$IMAGE")
$NO_CACHE && BUILD_ARGS+=(--no-cache)
BUILD_ARGS+=(.)
echo "building $IMAGE from $COMMIT: docker ${BUILD_ARGS[*]}" >&2
build_start=$(date +%s)
(cd "$SRC" && docker "${BUILD_ARGS[@]}") >"$OUT/build.log.txt" 2>&1 || {
    echo "build failed, see $OUT/build.log.txt" >&2
    exit 1
}
build_seconds=$(( $(date +%s) - build_start ))
printf '%s\n' "docker ${BUILD_ARGS[*]}" >"$RAW/build_command"
grep -c ' CACHED$' "$OUT/build.log.txt" >"$RAW/cached_steps" || true
echo "$build_seconds" >"$RAW/build_seconds"

# --- sizes ------------------------------------------------------------------------------ #
echo "measuring sizes" >&2
docker save --platform linux/arm64 "$IMAGE" | gzip -1 -c | wc -c | tr -d ' ' >"$RAW/gzip_proxy"
docker image inspect "$IMAGE" --format '{{.Size}}' >"$RAW/local_size"
docker image inspect "$IMAGE" --format '{{.Id}} {{.Os}}/{{.Architecture}}' \
    >"$RAW/image_identity"
docker image ls --tree "$IMAGE" >"$OUT/image-tree.log.txt" 2>&1 || true
docker history --no-trunc --format '{{.Size}}\t{{.CreatedBy}}' "$IMAGE" >"$OUT/history.log.txt"

# --- tooling and store ------------------------------------------------------------------ #
docker version >"$OUT/docker-version.log.txt" 2>&1
docker version --format '{{.Client.Version}}' >"$RAW/client_version"
docker version --format '{{.Server.Version}} {{.Server.APIVersion}}' >"$RAW/server_version"
docker buildx version >"$RAW/buildx_version" 2>&1 || echo unavailable >"$RAW/buildx_version"
docker buildx inspect >"$OUT/buildx-inspect.log.txt" 2>&1 || true
docker info --format '{{.Driver}}' >"$RAW/storage_driver"
docker info --format '{{json .DriverStatus}}' >"$RAW/driver_status"
uname -srm >"$RAW/host_uname"

# --- base digests ----------------------------------------------------------------------- #
BASE_REF=$(sed -n 's/^FROM \([^ ]*@sha256:[0-9a-f]*\).*/\1/p' "$SRC/Dockerfile" | sort -u)
echo "$BASE_REF" >"$RAW/base_refs"
INDEX_DIGEST=${BASE_REF##*@}
platform_digest=unresolved
if token=$(curl -fsS --max-time 20 \
        "https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/python:pull" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])'); then
    platform_digest=$(curl -fsS --max-time 20 -H "Authorization: Bearer $token" \
        -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json' \
        "https://registry-1.docker.io/v2/library/python/manifests/$INDEX_DIGEST" \
        | python3 -c 'import json,sys; print(next(m["digest"] for m in json.load(sys.stdin)["manifests"] if m["platform"].get("architecture") == "arm64"))' \
        ) || platform_digest=unresolved
fi
echo "$platform_digest" >"$RAW/platform_digest"

# --- versions inside the image ---------------------------------------------------------- #
docker run --rm --network none "$IMAGE" dpkg-query -W -f '${Package}\t${Version}\n' \
    >"$OUT/dpkg-packages.log.txt"
docker run --rm --network none "$IMAGE" gdalinfo --version >"$RAW/gdalinfo"
docker run --rm --network none "$IMAGE" python -c '
import json, os, platform, duckdb, rasterio, torch
con = duckdb.connect(":memory:")
con.execute("LOAD spatial"); con.execute("LOAD httpfs")
ext = con.execute("SELECT extension_name, extension_version FROM duckdb_extensions() WHERE installed AND extension_name IN (\x27spatial\x27, \x27httpfs\x27) ORDER BY 1").fetchall()
def du(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total
site = "/opt/venv/lib/python3.14/site-packages"
pkgs = sorted(((du(os.path.join(site, d)), d) for d in os.listdir(site) if os.path.isdir(os.path.join(site, d))), reverse=True)[:12]
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "rasterio": rasterio.__version__,
    "rasterio_gdal": rasterio.__gdal_version__,
    "duckdb": duckdb.__version__,
    "duckdb_extensions": {n: v for n, v in ext},
    "apparent_bytes": {"/opt/venv": du("/opt/venv"), "/opt/duckdb-home": du("/opt/duckdb-home")},
    "largest_site_packages": [{"package": d, "bytes": b} for b, d in pkgs],
}))
' >"$RAW/image_versions"

# --- functional checks ------------------------------------------------------------------ #
echo "running scripts/image_smoke.sh $IMAGE" >&2
smoke_exit=0
scripts/image_smoke.sh "$IMAGE" "$OUT" >"$OUT/smoke-summary.log.txt" 2>&1 || smoke_exit=$?
echo "$smoke_exit" >"$RAW/smoke_exit"

# --- report ----------------------------------------------------------------------------- #
python3 - "$RAW" "$OUT" "$COMMIT" "$STAMP" "$IMAGE" "$NO_CACHE" "$ENGINE_OS/$ENGINE_ARCH" \
    "$(shasum -a 256 "$SRC/Dockerfile" | cut -d' ' -f1)" \
    "$(shasum -a 256 "$SRC/uv.lock" | cut -d' ' -f1)" \
    "$(git log -1 --format=%cI "$COMMIT")" <<'PY'
import json
import sys
from pathlib import Path

raw, out, commit, stamp, image, no_cache, engine, dockerfile_sha, lock_sha, commit_date = sys.argv[1:]
r = lambda name: Path(raw, name).read_text().strip()  # noqa: E731
driver_status = r("driver_status")
store = "containerd" if "io.containerd.snapshotter.v1" in driver_status else "classic"
image_id, platform = r("image_identity").split()
versions = json.loads(r("image_versions"))
smoke = json.loads(Path(out, "smoke.json").read_text())
runtime_pkgs = dict(
    line.split("\t") for line in Path(out, "dpkg-packages.log.txt").read_text().splitlines()
)
report = {
    "schema": "geoai-image-size-report/v1",
    "measured_at_utc": stamp,
    "native": True,
    "commit": commit,
    "commit_date": commit_date,
    "tree": "clean (git archive of the commit)",
    "recipe": {"dockerfile": "Dockerfile", "dockerfile_sha256": dockerfile_sha,
               "uv_lock_sha256": lock_sha, "build_args": []},
    "base": {"refs": r("base_refs").split(), "index_digest": r("base_refs").split("@")[-1],
             "linux_arm64_manifest_digest": r("platform_digest")},
    "build": {"command": r("build_command"), "no_cache": no_cache == "true", "pull": False,
              "cached_steps": int(r("cached_steps") or 0), "seconds": int(r("build_seconds"))},
    "image": {"tag": image, "id": image_id, "platform": platform},
    "host": {"uname": r("host_uname"), "engine": engine},
    "tooling": {"docker_client": r("client_version"), "docker_server_api": r("server_version"),
                "buildx": r("buildx_version")},
    "store": {"image_store": store, "storage_driver": r("storage_driver"),
              "driver_status": json.loads(driver_status)},
    "sizes": {
        "gzip_archive_proxy_bytes": int(r("gzip_proxy")),
        "gzip_archive_proxy_command":
            f"set -o pipefail; docker save --platform linux/arm64 {image} | gzip -1 -c | wc -c",
        "local_size_bytes": int(r("local_size")),
        "local_size_command": f"docker image inspect {image} --format '{{{{.Size}}}}'",
        "local_size_semantics": f"{store} image store",
        "registry_transfer_bytes": None,
        "registry_transfer_note": "not measured: the image was not pushed",
    },
    "versions": {**versions, "gdalinfo": r("gdalinfo"),
                 "apt": {p: runtime_pkgs.get(p) for p in ("gdal-bin", "libgdal36", "curl")}},
    "smoke": {"exit": int(r("smoke_exit")), **smoke["summary"]},
}
Path(out, "report.json").write_text(json.dumps(report, indent=2) + "\n")

s = report["sizes"]
md = [
    f"# Native ARM64 image size report — `{commit[:12]}`",
    "",
    "Generated by `scripts/image_size_report.sh`. Evidence, not a gate: no threshold applies.",
    "",
    "## Sizes",
    "",
    "| metric | bytes | semantics |",
    "|---|---:|---|",
    f"| gzip archive proxy | {s['gzip_archive_proxy_bytes']:,} | `docker save --platform linux/arm64` piped to `gzip -1`; not a registry size |",
    f"| local size (`.Size`) | {s['local_size_bytes']:,} | {s['local_size_semantics']} |",
    "| registry transfer | not measured | the image was not pushed |",
    "",
    "## Provenance",
    "",
    "| | |",
    "|---|---|",
    f"| commit | `{commit}` ({commit_date}), built from `git archive` |",
    f"| measured | {stamp} |",
    f"| Dockerfile sha256 | `{dockerfile_sha}` |",
    f"| uv.lock sha256 | `{lock_sha}` |",
    f"| base (both stages) | `{' '.join(report['base']['refs'])}` |",
    f"| base linux/arm64 manifest | `{report['base']['linux_arm64_manifest_digest']}` |",
    f"| image | `{image}` `{image_id}` ({platform}) |",
    f"| build | `{report['build']['command']}`; `CACHED` steps: {report['build']['cached_steps']} (the digest-pinned base `FROM` counts when it is already local); {report['build']['seconds']} s |",
    f"| host / engine | {report['host']['uname']} / {engine} |",
    f"| Docker | client {report['tooling']['docker_client']}, server/API {report['tooling']['docker_server_api']}; {report['tooling']['buildx']} |",
    f"| image store / driver | {store} / {report['store']['storage_driver']} |",
    f"| Python / Torch | {versions['python']} / {versions['torch']} |",
    f"| GDAL | CLI: {report['versions']['gdalinfo']}; rasterio {versions['rasterio']} bundles GDAL {versions['rasterio_gdal']} |",
    f"| DuckDB / extensions | {versions['duckdb']} / {', '.join(f'{k} {v}' for k, v in versions['duckdb_extensions'].items())} |",
    f"| apt | {', '.join(f'{k} {v}' for k, v in report['versions']['apt'].items())} |",
    "",
    "## Largest site-packages entries (apparent bytes)",
    "",
    "| package | bytes |",
    "|---|---:|",
    *[f"| `{p['package']}` | {p['bytes']:,} |" for p in versions["largest_site_packages"]],
    f"| **`/opt/venv` total** | {versions['apparent_bytes']['/opt/venv']:,} |",
    f"| **`/opt/duckdb-home` total** | {versions['apparent_bytes']['/opt/duckdb-home']:,} |",
    "",
    "## Offline functional checks (`scripts/image_smoke.sh`)",
    "",
    "| check | status | limitation |",
    "|---|---|---|",
    *[
        f"| `{c['check']}` | {c['status']} | {c['detail'].get('limitation', '')} |"
        for c in smoke["checks"]
    ],
    "",
    "Full outputs: `smoke.json`; logs: `*.log.txt` in this directory.",
    "",
]
Path(out, "report.md").write_text("\n".join(md))
print(json.dumps({"sizes": s, "smoke": report["smoke"], "out": out}, indent=2))
PY
exit "$smoke_exit"
