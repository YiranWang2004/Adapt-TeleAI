"""Watch a specific Stage-1 process, evaluate checkpoints, then launch Stage 2.

Run with the system Python; evaluation and training use the project's venv.
State and reports are durable. A file lock and launch marker prevent duplicates.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def process(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return {"start_ticks": stat[19], "state": stat[0], "command": cmd}
    except (FileNotFoundError, ProcessLookupError):
        return None


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def passes(result, expected):
    s = result["summary"]
    rows = result["episodes"]
    return (
        s["episodes"] == expected
        and len(rows) == expected
        and len({r["motion"] for r in rows}) == expected
        and s["full_clip_completion_rate"] == 1
        and s["completed_episode_without_failure_rate"] == 1
        and s["failure_termination_rate"] == 0
        and s["unfinished_at_evaluation_limit"] == 0
        and all(r["episode_ended"] and r["completed_full_reference_clip"]
                and not r["failed_termination"] for r in rows)
        and math.isfinite(s["relative_body_position_error_m"])
        and 0 <= s["relative_body_position_error_m"] <= 0.04
        and math.isfinite(s["joint_position_rmse_rad"])
        and 0 <= s["joint_position_rmse_rad"] <= 0.18
        and math.isfinite(s["mean_return"])
    )


def run_command(command, log):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLCONFIGDIR": "/tmp/adapt-mpl",
           "CUDA_VISIBLE_DEVICES": "0"}
    with log.open("a") as output:
        subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                       stderr=subprocess.STDOUT, check=True, timeout=3600)


def watch(config, check_only=False):
    cfg = json.loads(config.read_text())
    run = ROOT / cfg["run_dir"]
    directory = ROOT / cfg["watch_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "status.json"
    with (directory / "watch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(phase, **details):
            data = {"phase": phase, "updated_at": dt.datetime.now().isoformat(),
                    "watcher_pid": os.getpid(), "stage1_pid": cfg["pid"], **details}
            write_json(state_path, data)
            print(json.dumps(data), flush=True)

        marker = directory / "stage2_launch.json"
        if marker.exists():
            print(marker.read_text(), flush=True)
            return
        current = process(cfg["pid"])
        if check_only:
            print(json.dumps({"process": current, "config": cfg,
                              "checkpoints": [p.name for p in run.glob("model_*.pt")]}))
            return
        try:
            while current and current["state"] != "Z" and current["start_ticks"] == cfg["start_ticks"]:
                if cfg["stage1_task"] not in current["command"]:
                    raise RuntimeError("Stage-1 PID command no longer matches")
                status("waiting_for_stage1", checkpoints=len(list(run.glob("model_*.pt"))))
                time.sleep(cfg["poll_seconds"])
                current = process(cfg["pid"])

            final = run / "model_9999.pt"
            final_jit = run / "jit/modeljit_9999.pt"
            if not final.is_file() or not final_jit.is_file():
                raise RuntimeError("Stage 1 exited without final model_9999.pt/JIT; no Stage 2 launched")
            candidates = sorted(
                [p for p in run.glob("model_*.pt") if p.stem[6:].isdigit()
                 and (run / "jit" / f"modeljit_{p.stem[6:]}.pt").is_file()],
                key=lambda p: int(p.stem[6:]), reverse=True)
            python = str(ROOT / ".venv/bin/python")
            accepted = []
            for start in range(0, len(candidates), 5):
                batch = candidates[start:start + 5]
                report = directory / f"rollouts_{start // 5}.json"
                status("evaluating", candidates=[str(p) for p in batch])
                run_command([python, "scripts/tools/evaluate_a3_pingpong.py",
                             "--task", cfg["stage1_task"], "--checkpoint",
                             *map(str, batch), "--seed", "42", "--device", "cuda:0",
                             "--output", str(report)], directory / "evaluation.log")
                results = json.loads(report.read_text())
                if results["num_motions"] != 43 or results["repeats"] != 1:
                    raise RuntimeError("Unexpected motion library/evaluation protocol")
                accepted = [r for r in results["checkpoints"] if passes(r, 43)]
                if accepted:
                    break
            if not accepted:
                status("acceptance_failed", reason="No checkpoint satisfies acceptance thresholds")
                return
            best = min(accepted, key=lambda r: (
                r["summary"]["relative_body_position_error_m"],
                r["summary"]["joint_position_rmse_rad"], -r["summary"]["mean_return"]))
            tracker = Path(best["jit_policy"])
            write_json(directory / "selected_checkpoint.json", best)
            status("strike_diagnostics", checkpoint=best["resolved_checkpoint"])
            run_command([python, "scripts/tools/diagnose_a3_strike.py",
                         "--task", cfg["stage1_task"], "--checkpoint",
                         best["resolved_checkpoint"], "--seed", "42", "--device", "cuda:0",
                         "--output", str(directory / "strike_diagnostics.json")],
                        directory / "evaluation.log")
            for entry in Path("/proc").iterdir():
                if entry.name.isdigit():
                    try:
                        other = process(int(entry.name))
                    except (PermissionError, ProcessLookupError):
                        continue
                    if other and cfg["stage2_task"] in other["command"] and other["state"] != "Z":
                        raise RuntimeError(f"Stage 2 already running, PID {entry.name}; refusing duplicate")
            digest = hashlib.sha256(tracker.read_bytes()).hexdigest()
            if digest != best["jit_sha256"]:
                raise RuntimeError("Selected JIT changed after evaluation")
            run_name = "g1_pingpong_stage2_auto_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            log = ROOT / "logs/full_training" / f"{run_name}.log"
            command = [str(ROOT / ".venv/bin/train"), cfg["stage2_task"],
                       "--env.scene.num-envs", "1024",
                       "--env.actions.joint-pos.tracker-file", str(tracker),
                       "--agent.max-iterations", "10000", "--agent.save-interval", "500",
                       "--agent.run-name", run_name, "--agent.logger", "tensorboard",
                       "--gpu-ids", "[0]"]
            # Test the selected tracker in the actual next-stage training path.
            # This short run has its own name; it never resumes the formal run.
            preflight_name = run_name + "_preflight"
            preflight_command = command.copy()
            for option, value in (("--env.scene.num-envs", "16"),
                                  ("--agent.max-iterations", "2"),
                                  ("--agent.save-interval", "1"),
                                  ("--agent.run-name", preflight_name)):
                preflight_command[preflight_command.index(option) + 1] = value
            status("stage2_preflight", checkpoint=best["resolved_checkpoint"],
                   command=preflight_command)
            run_command(preflight_command, directory / "stage2_preflight.log")
            preflight_runs = list((ROOT / "logs/rsl_rl/g1_pingpong_receive_stage2").glob(
                "*_" + preflight_name))
            if len(preflight_runs) != 1:
                raise RuntimeError("Cannot uniquely locate Stage-2 preflight artifacts")
            preflight = preflight_runs[0]
            if not (preflight / "jit/modeljit_1.pt").is_file():
                raise RuntimeError("Stage-2 preflight did not export final JIT")
            verification = '''import hashlib, sys, torch
from pathlib import Path
run = Path(sys.argv[1])
expected = sys.argv[2]
assert hashlib.sha256((run / "tracker.pt").read_bytes()).hexdigest() == expected
payload = torch.load(run / "model_1.pt", map_location="cpu", weights_only=False)
assert payload["iter"] == 1
assert payload["infos"]["pingpong_tracker_sha256"] == expected
def verify(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), "Nonfinite preflight checkpoint tensor"
    elif isinstance(value, dict):
        for child in value.values(): verify(child)
    elif isinstance(value, (tuple, list)):
        for child in value: verify(child)
verify(payload)
print("Stage-2 tracker loading, two training updates, checkpoint/JIT saving, dependency hash and tensor finiteness passed.")
'''
            run_command([python, "-c", verification, str(preflight), digest],
                        directory / "stage2_preflight.log")
            write_json(directory / "stage2_preflight_passed.json", {
                "run_dir": str(preflight), "tracker_sha256": digest,
                "checkpoint": best["resolved_checkpoint"], "updates": 2,
                "num_envs": 16, "passed_at": dt.datetime.now().isoformat()})
            launch = {"command": command, "log": str(log), "tracker_sha256": digest,
                      "checkpoint": best["resolved_checkpoint"], "phase": "launching"}
            # Persist intent before spawning: on an uncertain restart, do not launch twice.
            write_json(marker, launch)
            with log.open("a") as output:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=output, stderr=subprocess.STDOUT,
                                         start_new_session=True, env={**os.environ,
                                             "PYTHONUNBUFFERED": "1", "MPLCONFIGDIR": "/tmp/adapt-mpl"})
            launch.update(pid=child.pid, phase="running")
            write_json(marker, launch)
            (ROOT / "logs/full_training/g1_pingpong_stage2_auto.pid").write_text(str(child.pid) + "\n")
            (ROOT / "logs/full_training/g1_pingpong_stage2_auto.logpath").write_text(str(log) + "\n")
            status("stage2_running", launch=launch)
            code = child.wait()
            status("stage2_exited", returncode=code, stage2_pid=child.pid, log=str(log))
        except Exception as error:
            status("blocked", reason=str(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    watch(args.config, args.check_only)
