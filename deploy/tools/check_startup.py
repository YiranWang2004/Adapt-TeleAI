"""Validate staged real startup using a MuJoCo plant, never robot IO.

The plant models suspended PD positioning and operator placement on the ground.
Virtual time does not establish hardware timing or real contact conditions.
"""

import argparse
import json
from pathlib import Path

import torch

from deploy.g1_pingpong.configuration import add_policy_arguments, from_arguments
from deploy.tools.check_real_loop import check


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  add_policy_arguments(parser)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(1)
  deployment = from_arguments(args)
  args.output.mkdir(parents=True, exist_ok=False)
  results = [
    check(deployment, args.output / scenario, scenario, swings=2)
    for scenario in (
      "no_ball",
      "normal",
      "switch_back",
      "root_dropout",
      "dds_dropout",
      "stop",
    )
  ]
  report = {
    "passed": all(row["passed"] for row in results),
    "actual_robot_commands_sent": False,
    "virtual_time": True,
    "startup_condition": "suspended positioning, then operator placement",
    "results": results,
  }
  (args.output / "validation.json").write_text(json.dumps(report, indent=2))
  print(json.dumps(report, indent=2))
  raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
  main()
