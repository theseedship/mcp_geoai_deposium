#!/usr/bin/env bash
# Offline functional checks of a built GeoAI image (#173).
#
#   scripts/image_smoke.sh IMAGE [OUT_DIR]
#
# 1. Normal entrypoint and healthcheck: the image's own CMD runs in a detached
#    `--network none` container with the data plane enabled (so startup also prepares the
#    cached DuckDB extensions offline); /health must answer and Docker must report the
#    container `healthy` through the image HEALTHCHECK.
# 2. In-image checks: scripts/image_smoke_checks.py and tests/fixtures/image-smoke are
#    mounted read-only into a disposable `--network none` container and run from the image
#    WORKDIR with the image's own Python. Nothing is installed into the image: no dev
#    tooling, no extra package.
#
# Writes OUT_DIR/smoke.json (default: a temporary directory) and OUT_DIR/smoke-*.log.txt,
# prints one line per check, and exits non-zero when any check fails. The in-image checks
# are stopped and recorded as failed after SMOKE_CHECKS_TIMEOUT_S seconds (default 900).
set -euo pipefail

IMAGE=${1:?usage: scripts/image_smoke.sh IMAGE [OUT_DIR]}
OUT_DIR=${2:-$(mktemp -d)}
CHECKS_TIMEOUT_S=${SMOKE_CHECKS_TIMEOUT_S:-900}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$OUT_DIR"

NAME="geoai-image-smoke-$$"
CHECKS_NAME="geoai-image-smoke-checks-$$"
cleanup() { docker rm -f "$NAME" "$CHECKS_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

# --- 1. entrypoint + healthcheck ------------------------------------------------------- #
# A throwaway bearer token, generated here and never recorded: an HTTP data-plane start
# refuses to run without one.
TOKEN=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
health_body=""
health_seconds=""
docker_health=""
docker_health_seconds=0
entrypoint_launch_exit=0
if docker run -d --name "$NAME" --network none \
    -e ALLOW_LOCAL_PARQUET_SOURCE=true \
    -e LOCAL_PARQUET_SOURCE_ROOT=/data \
    -e GEOAI_MESH_AUTH_TOKEN="$TOKEN" \
    "$IMAGE" >/dev/null 2>"$OUT_DIR/smoke-entrypoint.log.txt"; then
    start=$SECONDS
    while (( SECONDS - start < 90 )); do
        if health_body=$(docker exec "$NAME" curl -fsS http://localhost:5001/health 2>/dev/null); then
            health_seconds=$(( SECONDS - start ))
            break
        fi
        if [[ $(docker inspect -f '{{.State.Running}}' "$NAME") != true ]]; then
            break
        fi
        sleep 1
    done

    # Docker probes the image HEALTHCHECK during its start period at the engine's start
    # interval (5 s by default), not only every 30 s: `healthy` was observed at ~6 s.
    while (( SECONDS - start < 150 )); do
        docker_health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$NAME")
        [[ $docker_health == healthy || $docker_health == unhealthy ]] && break
        [[ $(docker inspect -f '{{.State.Running}}' "$NAME") != true ]] && break
        sleep 2
    done
    docker_health_seconds=$(( SECONDS - start ))
else
    entrypoint_launch_exit=$?
fi
docker inspect -f '{{json .Config.Cmd}} {{json .Config.Healthcheck}} {{json .Config.ExposedPorts}}' \
    "$IMAGE" >"$OUT_DIR/smoke-image-config.log.txt" 2>&1 || true
docker logs "$NAME" >>"$OUT_DIR/smoke-entrypoint.log.txt" 2>&1 || true
cleanup

# --- 2. in-image functional checks ----------------------------------------------------- #
# A named, detached container so it can be bounded in time and always removed.
checks_timed_out=false
if docker run -d --name "$CHECKS_NAME" --network none \
    -v "$REPO_ROOT/scripts/image_smoke_checks.py:/smoke/image_smoke_checks.py:ro" \
    -v "$REPO_ROOT/tests/fixtures/image-smoke:/smoke/fixtures:ro" \
    "$IMAGE" python /smoke/image_smoke_checks.py /smoke/fixtures \
    >/dev/null 2>"$OUT_DIR/smoke-checks.log.txt"; then
    checks_start=$SECONDS
    while [[ $(docker inspect -f '{{.State.Running}}' "$CHECKS_NAME") == true ]]; do
        if (( SECONDS - checks_start >= CHECKS_TIMEOUT_S )); then
            checks_timed_out=true
            docker kill "$CHECKS_NAME" >/dev/null 2>&1 || true
            break
        fi
        sleep 1
    done
    docker logs "$CHECKS_NAME" >>"$OUT_DIR/smoke-checks.log.txt" 2>&1 || true
    checks_exit=$(docker inspect -f '{{.State.ExitCode}}' "$CHECKS_NAME")
    $checks_timed_out && checks_exit=124
    docker rm -f "$CHECKS_NAME" >/dev/null 2>&1 || true
else
    checks_exit=$?
fi

python3 - "$OUT_DIR" "$IMAGE" "$health_body" "$health_seconds" "$docker_health" \
    "$docker_health_seconds" "$checks_exit" "$checks_timed_out" "$CHECKS_TIMEOUT_S" \
    "$entrypoint_launch_exit" <<'PY'
import json
import sys
from pathlib import Path

out, image, body, body_s, health, health_s, checks_exit, timed_out, timeout_s, launch_exit = sys.argv[1:]
log = Path(out, "smoke-checks.log.txt").read_text(encoding="utf-8", errors="replace")
lines = [x for x in log.splitlines() if x.startswith("IMAGE_SMOKE_JSON=")]
report = json.loads(lines[-1].split("=", 1)[1]) if lines else {"checks": []}
entry_log = Path(out, "smoke-entrypoint.log.txt").read_text(encoding="utf-8", errors="replace")
prepared = "Prepared DuckDB extensions for isolated mesh workers" in entry_log
entrypoint = {
    "check": "entrypoint_healthcheck",
    "status": "pass" if body == "OK" and health == "healthy" and prepared else "fail",
    "detail": {
        "launch_exit": int(launch_exit),
        "health_body": body,
        "health_after_s": int(body_s) if body_s else None,
        "docker_health_status": health,
        "docker_health_after_s": int(health_s),
        "data_plane_extensions_prepared_offline": prepared,
        "image_config": Path(out, "smoke-image-config.log.txt").read_text().strip(),
    },
}
checks = [entrypoint, *report["checks"]]
# The container's own outcome is a check: a crash or a timeout after the JSON line was
# printed must not read as success.
if not lines or int(checks_exit) != 0 or timed_out == "true":
    checks.append(
        {
            "check": "in_image_checks",
            "status": "fail",
            "detail": {
                "exit": int(checks_exit),
                "timed_out": timed_out == "true",
                "timeout_s": int(timeout_s),
                "json_line_found": bool(lines),
            },
        }
    )
for c in checks:
    c["detail"].pop("trace", None) if c["status"] == "pass" else None
failed = [c["check"] for c in checks if c["status"] != "pass"]
summary = {
    "image": image,
    "pass": len(checks) - len(failed),
    "fail": len(failed),
    "failed": failed,
    "with_limitation": [c["check"] for c in checks if "limitation" in c["detail"]],
}
Path(out, "smoke.json").write_text(
    json.dumps({"summary": summary, "checks": checks}, indent=2, sort_keys=True) + "\n"
)
for c in checks:
    note = " (limitation recorded)" if "limitation" in c["detail"] else ""
    print(f"{c['status']:4}  {c['check']}{note}")
print(json.dumps(summary))
sys.exit(1 if failed else 0)
PY
