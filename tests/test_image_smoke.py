"""Exercise the image smoke supervisor offline, without Docker or an image."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "mode,failed",
    [
        ("pass", []),
        ("entrypoint_launch_error", ["entrypoint_healthcheck"]),
        ("checks_launch_error", ["in_image_checks"]),
        ("failed_check", ["fixture"]),
        ("crash_after_json", ["in_image_checks"]),
        ("missing_json", ["in_image_checks"]),
        ("timeout", ["in_image_checks"]),
    ],
)
def test_image_smoke_records_container_outcomes(tmp_path: Path, mode: str, failed: list[str]):
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        + """import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
mode = os.environ["FAKE_DOCKER_MODE"]
with Path(os.environ["FAKE_DOCKER_CALLS"]).open("a") as log:
    # Only record the command and container name, never the throwaway token.
    name = args[args.index("--name") + 1] if "--name" in args else args[-1]
    log.write(json.dumps([args[0], " ".join(args[2:]) if args[0] == "rm" else name]) + "\\n")
checks = "smoke-checks-" in name
if args[0] == "run":
    if mode == ("checks_launch_error" if checks else "entrypoint_launch_error"):
        print("docker: Error response from daemon: invalid mount config", file=sys.stderr)
        sys.exit(125)
    print("fake-container-id")
elif args[0] == "exec":
    print("OK")
elif args[0] == "inspect":
    template = args[2]
    if ".Config.Cmd" in template:
        if mode == "entrypoint_launch_error":
            print("docker: image metadata unavailable", file=sys.stderr)
            sys.exit(125)
        print('["python", "-m", "src.server"] {} {"5001/tcp":{}}')
    elif ".State.Running" in template:
        print("true" if checks and mode == "timeout" else "false")
    elif ".State.ExitCode" in template:
        print(9 if mode == "crash_after_json" else 0)
    elif ".State.Health" in template:
        print("healthy")
    else:
        sys.exit(2)
elif args[0] == "logs":
    if not checks:
        print("Prepared DuckDB extensions for isolated mesh workers")
    elif mode != "missing_json":
        print("IMAGE_SMOKE_JSON=" + json.dumps({"checks": [{
            "check": "fixture", "status": "fail" if mode == "failed_check" else "pass",
            "detail": {},
        }]}))
elif args[0] not in ("rm", "kill"):
    sys.exit(2)
"""
    )
    docker.chmod(0o755)
    calls_path = tmp_path / "calls.jsonl"
    out = tmp_path / "out"
    result = subprocess.run(
        [
            "bash",
            str(Path(__file__).resolve().parents[1] / "scripts/image_smoke.sh"),
            "fake:image",
            str(out),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "FAKE_DOCKER_MODE": mode,
            "FAKE_DOCKER_CALLS": str(calls_path),
            "SMOKE_CHECKS_TIMEOUT_S": "0",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == (1 if failed else 0), result.stderr
    # This is the JSON contract consumed by image_size_report.sh, even on launch errors.
    report = json.loads((out / "smoke.json").read_text())
    assert report["summary"]["failed"] == failed
    assert report["summary"]["fail"] == len(failed)
    checks = {check["check"]: check for check in report["checks"]}
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    launched_names = [name for command, name in calls if command == "run"]
    assert len(launched_names) == 2
    for name in launched_names:
        assert any(command == "rm" and name in removed for command, removed in calls)
    if mode.endswith("launch_error"):
        entrypoint = mode == "entrypoint_launch_error"
        name = launched_names[0 if entrypoint else 1]
        assert ["inspect", name] not in calls
        log = out / ("smoke-entrypoint.log.txt" if entrypoint else "smoke-checks.log.txt")
        assert "invalid mount config" in log.read_text()
        detail = checks["entrypoint_healthcheck" if entrypoint else "in_image_checks"]["detail"]
        assert detail["launch_exit" if entrypoint else "exit"] == 125
    if mode in ("crash_after_json", "missing_json", "timeout"):
        detail = checks["in_image_checks"]["detail"]
        assert detail["exit"] == {"crash_after_json": 9, "missing_json": 0, "timeout": 124}[mode]
        assert detail["timed_out"] == (mode == "timeout")
        assert detail["json_line_found"] == (mode != "missing_json")
    if mode == "timeout":
        assert ["kill", launched_names[1]] in calls
