"""Compare G1's planner interface with a read-only A3 source snapshot offline."""

import argparse
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import yaml

from deploy.g1_pingpong.perception import BallEstimate
from deploy.hope_g1.a3_parameters import PROFILE, ball_parameters
from deploy.hope_g1.a3_planner import A3Planner
from deploy.hope_g1.world import mujoco_ball_config, simulation_translation


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--source",
    type=Path,
    default=Path("deploy/hope_g1/artifacts/a3_planner_readonly_20261021"),
  )
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  package = types.ModuleType("independent_a3")
  package.__path__ = [str(args.source.resolve())]
  sys.modules[package.__name__] = package
  const = importlib.import_module("independent_a3.constants")
  pred = importlib.import_module("independent_a3.ball_trajectory_predictor")
  target = importlib.import_module("independent_a3.racket_target_planner")
  params = yaml.safe_load(PROFILE.read_text())["hope_planner_optimized"][
    "ros__parameters"
  ]
  cfg = yaml.safe_load(Path("deploy/hope_g1/configs/rally.yaml").read_text())
  translation = simulation_translation(cfg["world"])
  physical = const.BallPhysics(
    k=params["drag_k"], C_h=params["restitution_h"], C_v=params["restitution_v"]
  )
  geometry = const.TableParams()
  rng = np.random.default_rng(42)
  cases = []
  for i in range(64):
    p = rng.uniform([0.2, -1.3, 0.1], [1.5, -0.2, 0.8])
    v = rng.uniform([-4, -0.2, -2], [-1.5, 0.2, 1])
    plane = -0.25 if i % 2 else 0.0
    options = const.PlannerConfig(
      x_hit=plane,
      # Compare identical task inputs; the landing goal is configurable,
      # independently of the preserved A3 algorithm and physical parameters.
      target_land=np.r_[cfg["ball"]["target_xy"], cfg["ball"]["target_land_z"]],
      delta_t_flight=params["delta_t_flight"],
      C_r=params["restitution_racket"],
    )
    strike = pred.BallTrajectoryPredictor(physical, options, geometry).predict(
      p, v, 10.0
    )
    expected = target.RacketTargetPlanner(physical, options, geometry).plan(strike)
    for floor in [False, True]:
      offset = translation if floor else np.zeros(3)
      ball = mujoco_ball_config(cfg["ball"], offset) if floor else cfg["ball"]
      actual = A3Planner(ball, cfg["a3"]).plan(
        BallEstimate(p - offset, v, 100.0, 10.0), 10.03, plane - offset[0]
      )
      row = dict(
        id=i,
        frame="mujoco_floor" if floor else "hope_table",
        valid=expected.valid,
        same_validity=(actual is not None) == expected.valid,
      )
      if actual is not None and expected.valid:
        row.update(
          position_error=float(
            np.max(np.abs(actual.position + offset - expected.p_intercept))
          ),
          velocity_error=float(np.max(np.abs(actual.velocity - expected.v_racket))),
          normal_error=float(np.max(np.abs(actual.normal - expected.n_racket))),
          deadline_error=abs(actual.deadline - expected.t_strike),
          tts_error=abs(actual.diagnostic["remaining"] - (expected.t_strike - 10.03)),
        )
      cases.append(row)
  maxima = {
    k: max((c.get(k, 0) for c in cases), default=0)
    for k in [
      "position_error",
      "velocity_error",
      "normal_error",
      "deadline_error",
      "tts_error",
    ]
  }
  hashes = {}
  for name in [
    "constants.py",
    "ball_trajectory_predictor.py",
    "racket_target_planner.py",
  ]:
    source = (args.source / name).read_bytes()
    local = (Path("deploy/hope_g1/a3_reference") / name).read_bytes()
    hashes[name] = dict(
      sha256=hashlib.sha256(source).hexdigest(), byte_identical=source == local
    )
  passed = (
    all(c["same_validity"] for c in cases)
    and all(v < 1e-10 for v in maxima.values())
    and all(v["byte_identical"] for v in hashes.values())
  )
  report = dict(
    passed=passed,
    source=str(args.source),
    profile=str(PROFILE),
    physical_parameters=ball_parameters(),
    task_target_land=[*cfg["ball"]["target_xy"], cfg["ball"]["target_land_z"]],
    comparisons=len(cases),
    maxima=maxima,
    core_files=hashes,
    cases=cases,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2))
  print(
    json.dumps(
      {k: report[k] for k in ["passed", "comparisons", "maxima", "core_files"]},
      indent=2,
    )
  )
  if not passed:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
