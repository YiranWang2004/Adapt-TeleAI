"""Test every motion through preparation/swing/recovery, then continuous selection."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path

import torch

from deploy.g1_pingpong.configuration import add_policy_arguments, from_arguments
from deploy.g1_pingpong.simulation import run_rally


def validate(deployment, args):
  output = args.output
  output.mkdir(parents=True, exist_ok=False)
  # Each sweep keeps its startup configuration even while a later correction
  # is being developed in the workspace.
  for name in ("real_config", "mocap_config", "simulation_config"):
    snapshot = output / f"{name}.yaml"
    snapshot.write_text(Path(getattr(args, name)).read_text())
    setattr(args, name, snapshot)
  indices = (
    list(range(len(deployment.motions)))
    if args.motions == "all"
    else [int(value) for value in args.motions.split(",")]
  )
  report = {
    "checkpoint": str(deployment.checkpoint),
    "source_sha256": {
      str(path): hashlib.sha256(path.read_bytes()).hexdigest()
      for path in sorted(Path("deploy").glob("**/*.py"))
      if "recordings" not in path.parts
    },
    "seed": args.seed,
    "final_hold_seconds": args.hold_seconds,
    "single": [],
    "continuous": None,
  }

  def run(name, home, selection, allow, episodes, duration):
    destination = output / name
    destination.mkdir()
    with (destination / "run.log").open("w") as stream:
      with contextlib.redirect_stdout(stream):
        result = run_rally(
          deployment,
          args.real_config,
          args.mocap_config,
          simulation_config=args.simulation_config,
          motion_index=home,
          motion_select=selection,
          motion_allow=allow,
          episodes=episodes,
          duration=duration,
          seed=args.seed,
          final_hold_seconds=args.hold_seconds,
          output=destination,
        )
    sequence = [
      event["motion_index"]
      for event in result["events"]
      if event["event"] == "motion_selected"
    ]
    row = {
      "case": name,
      "home": home,
      "passed": result["passed"],
      "completed": result["completed_swings"],
      "recovered": result["recovered_swings"],
      "hits": sum(swing["hit"] for swing in result["swings"]),
      "motion_sequence": sequence,
      "motion_switches": sum(
        a != b for a, b in zip(sequence, sequence[1:], strict=False)
      ),
      "max_tilt": result["max_control_tilt_degrees"],
      "termination": result["termination"],
      "failure": result["failure"],
    }
    print(json.dumps(row), flush=True)
    return row

  if args.phase in ("single", "both"):
    for index in indices:
      row = run(
        f"motion_{index:02d}",
        index if args.single_home is None else args.single_home,
        "fixed" if args.single_home is None else "auto",
        None if args.single_home is None else [index],
        1,
        args.single_duration,
      )
      row["motion_index"] = index
      report["single"].append(row)
      (output / "report.json").write_text(json.dumps(report, indent=2))
  if args.phase in ("continuous", "both"):
    report["continuous"] = run(
      "continuous", args.home, "auto", indices, args.swings, args.duration
    )
    report["continuous"]["switching_passed"] = (
      report["continuous"]["passed"] and report["continuous"]["motion_switches"] > 0
    )
  report["single_passed"] = sum(row["passed"] for row in report["single"])
  report["single_failed"] = [
    row["motion_index"] for row in report["single"] if not row["passed"]
  ]
  (output / "report.json").write_text(json.dumps(report, indent=2))
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  add_policy_arguments(parser)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--phase", choices=("single", "continuous", "both"), default="both"
  )
  parser.add_argument("--motions", default="all")
  parser.add_argument("--home", type=int, default=5)
  parser.add_argument(
    "--single-home",
    type=int,
    help="Use one fixed ready pose for every single-motion test",
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--swings", type=int, default=30)
  parser.add_argument("--duration", type=float, default=600)
  parser.add_argument("--single-duration", type=float, default=45)
  parser.add_argument("--hold-seconds", type=float, default=5)
  parser.add_argument("--real-config", default="deploy/configs/real.yaml")
  parser.add_argument("--mocap-config", default="deploy/configs/mocap.yaml")
  parser.add_argument("--simulation-config", default="deploy/configs/mujoco.yaml")
  args = parser.parse_args()
  torch.set_num_threads(1)
  report = validate(from_arguments(args), args)
  if report["single_failed"] or (
    report["continuous"] is not None and not report["continuous"]["switching_passed"]
  ):
    raise SystemExit(1)


if __name__ == "__main__":
  main()
