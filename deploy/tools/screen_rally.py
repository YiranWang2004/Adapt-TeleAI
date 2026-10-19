"""Run independent long rally cases with unchanged deployment physics/control."""

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import torch

from deploy.g1_pingpong.configuration import add_policy_arguments, from_arguments
from deploy.g1_pingpong.simulation import run_rally

_deployment = None
_args = None


def initialize(args):
  global _deployment, _args
  torch.set_num_threads(1)
  _args = args
  _deployment = from_arguments(args)


def run_case(case):
  assert _args is not None and _deployment is not None
  destination = _args.output / case["name"]
  destination.mkdir()
  started = time.monotonic()
  with (destination / "run.log").open("w") as stream:
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
      result = run_rally(
        _deployment,
        _args.real_config,
        _args.mocap_config,
        simulation_config=_args.simulation_config,
        motion_index=case.get("home", 19),
        motion_select="auto",
        motion_allow=case["motions"],
        episodes=case.get("swings", 50),
        duration=case.get("duration", 1200),
        seed=case["seed"],
        final_hold_seconds=5,
        output=destination,
      )
  sequence = [
    event["motion_index"]
    for event in result["events"]
    if event["event"] == "motion_selected"
  ]
  recovered = []
  active = None
  for event in result["events"]:
    if event["event"] == "motion_selected":
      active = event["motion_index"]
    elif event["event"] == "recovered":
      recovered.append(active)
  return {
    **case,
    **{
      key: result[key]
      for key in (
        "passed",
        "completed_swings",
        "recovered_swings",
        "failure",
        "termination",
        "simulation_time",
        "max_control_tilt_degrees",
        "robot_resets",
      )
    },
    "sequence": sequence,
    "recovered_sequence": recovered,
    "switches": sum(a != b for a, b in zip(sequence, sequence[1:], strict=False)),
    "hits": sum(swing["hit"] for swing in result["swings"]),
    "valid_returns": sum(swing["success"] for swing in result["swings"]),
    "wall_seconds": time.monotonic() - started,
    "summary": str(destination / "summary.json"),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  add_policy_arguments(parser)
  parser.add_argument("--cases", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--real-config", default="deploy/configs/real.yaml")
  parser.add_argument("--mocap-config", default="deploy/configs/mocap.yaml")
  parser.add_argument("--simulation-config", default="deploy/configs/mujoco.yaml")
  args = parser.parse_args()
  cases = json.loads(args.cases.read_text())
  names = [case["name"] for case in cases]
  if len(set(names)) != len(names) or any(
    Path(name).name != name or name in (".", "..") for name in names
  ):
    parser.error("Case names must be unique plain directory names")
  args.output.mkdir(parents=True, exist_ok=False)
  (args.output / "cases.json").write_text(json.dumps(cases, indent=2))
  for name in ("real_config", "mocap_config", "simulation_config"):
    snapshot = args.output / f"{name}.yaml"
    snapshot.write_text(Path(getattr(args, name)).read_text())
    setattr(args, name, snapshot)
  hashes = {
    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
    for folder in ("g1_pingpong", "backends", "io", "tools")
    for path in sorted((Path("deploy") / folder).glob("*.py"))
  }
  report = {"config": str(args.config), "source_sha256": hashes, "cases": []}
  with concurrent.futures.ProcessPoolExecutor(
    max_workers=args.workers,
    mp_context=multiprocessing.get_context("spawn"),
    initializer=initialize,
    initargs=(args,),
  ) as pool:
    futures = {pool.submit(run_case, case): case for case in cases}
    for future in concurrent.futures.as_completed(futures):
      row = future.result()
      report["cases"].append(row)
      (args.output / "report.json").write_text(json.dumps(report, indent=2))
      print(json.dumps(row), flush=True)
  if any(not row["passed"] for row in report["cases"]):
    raise SystemExit(1)


if __name__ == "__main__":
  main()
