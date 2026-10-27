"""Offline ideal-paddle counterfactual; never changes actual rollout results."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deploy.benchmarks.racket_command.physical_ball import PhysicalBall


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--recording", type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(1)
  report = json.loads((args.recording / "report.json").read_text())
  cases = []
  for row in report["results"]:
    command = row["command"]
    if not command:
      continue
    feed = row["feed"]
    ball = PhysicalBall(
      feed["initial_position"], feed["initial_velocity"], report["physics"]
    )
    p, n, v = map(
      np.asarray, (command["position"], command["normal"], command["velocity"])
    )
    while ball.elapsed < feed["oracle_crossing_time"] + 2:
      if ball.elapsed + 1e-10 < command["issued_at"]:
        ball.step(0.001)
      else:
        if ball.previous_racket is None:
          ball.previous_racket = p + v * (ball.elapsed - command["deadline"])
        ball.step(0.001, p + v * (ball.elapsed + 0.001 - command["deadline"]), n)
      if ball.first_landing or ball.floor_hit:
        break
    b = ball.report()
    cases.append(
      dict(
        trial_id=row["trial_id"],
        actual_contact=row["ball_hit"],
        actual_legal_return=row["legal_return"],
        ideal_linear_paddle_contact=b["hit"],
        ideal_linear_paddle_legal_return=b["legal_return"],
        ideal_contact_time=ball.contact["time"] if ball.contact else None,
        ideal_net_hit=b["net_hit"],
        ideal_first_landing=b["first_landing"],
      )
    )
  output = dict(
    scope="Diagnostic only: ideal paddle follows the latched position/velocity/normal locally from command issue; no feedback into actual execution or ball sampling",
    cases=len(cases),
    ideal_contacts=sum(c["ideal_linear_paddle_contact"] for c in cases),
    ideal_legal_returns=sum(c["ideal_linear_paddle_legal_return"] for c in cases),
    actual_contacts=sum(c["actual_contact"] for c in cases),
    actual_legal_returns=sum(c["actual_legal_return"] for c in cases),
    results=cases,
  )
  (args.recording / "ideal_paddle_audit.json").write_text(json.dumps(output, indent=2))
  print(json.dumps({k: v for k, v in output.items() if k != "results"}, indent=2))


if __name__ == "__main__":
  main()
