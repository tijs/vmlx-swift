"""Run one owned model process with preserved physical-memory/host evidence."""

import argparse
import ctypes
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path


def host():
    pressure = int(
        subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    )
    swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    return {
        "pressure": pressure,
        "swap_mib": float(re.search(r"used = ([0-9.]+)M", swap)[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=int, default=900)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    assert command
    args.out.mkdir(parents=True, exist_ok=False)
    inventory = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True)
    owners = [
        line
        for line in inventory.splitlines()
        if Path(line.strip().split(maxsplit=1)[-1]).name
        in {"mei", "llama-server", "llama-cli", "SenseNovaProbe"}
    ]
    if owners:
        raise RuntimeError(f"existing inference owner: {owners}")
    baseline = []
    for _ in range(3):
        baseline.append(host())
        time.sleep(1)
    assert all(p["pressure"] == 1 for p in baseline), baseline
    assert (
        max(p["swap_mib"] for p in baseline) - min(p["swap_mib"] for p in baseline) < 64
    )
    assert shutil.disk_usage(args.out).free >= 20 * 1024**3
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "command": command,
                "baseline": baseline,
                "limits": {
                    "physical_footprint_gib": 24,
                    "swap_growth_mib": 256,
                    "warning_samples": 2,
                    "critical_immediate": True,
                    "deadline_s": args.deadline,
                },
            },
            indent=2,
        )
    )
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"}
    }
    reason = None
    proc = None
    began = time.monotonic()
    with (
        (args.out / "stdout.log").open("w") as log,
        (args.out / "resources.jsonl").open("w") as journal,
    ):
        try:
            proc = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
            warnings = 0
            while proc.poll() is None:
                sample = host()
                sample["elapsed_s"] = time.monotonic() - began
                buf = ctypes.create_string_buffer(4096)
                if lib.proc_pid_rusage(proc.pid, 0, ctypes.byref(buf)) != 0:
                    if proc.poll() is not None:
                        break
                    raise RuntimeError("physical-footprint observation failed")
                sample["pid"] = proc.pid
                sample["phys_footprint"] = int.from_bytes(buf.raw[72:80], "little")
                journal.write(json.dumps(sample) + "\n")
                journal.flush()
                warnings = warnings + 1 if sample["pressure"] != 1 else 0
                if sample["pressure"] >= 4:
                    reason = "critical pressure"
                elif warnings >= 2:
                    reason = "sustained warning pressure"
                elif sample["swap_mib"] - baseline[0]["swap_mib"] >= 256:
                    reason = "swap growth >=256MiB"
                elif sample["phys_footprint"] > 24 * 1024**3:
                    reason = "physical footprint >24GiB"
                elif shutil.disk_usage(args.out).free < 20 * 1024**3:
                    reason = "disk floor <20GiB"
                elif sample["elapsed_s"] > args.deadline:
                    reason = "deadline"
                if reason:
                    break
                time.sleep(1)
        finally:
            if proc is not None:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=5)
                (args.out / "outcome.json").write_text(
                    json.dumps(
                        {
                            "returncode": proc.returncode,
                            "guard_stop": reason,
                            "pid": proc.pid,
                            "elapsed_s": time.monotonic() - began,
                            "terminal_host": host(),
                        },
                        indent=2,
                    )
                )
    if reason or proc is None or proc.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
