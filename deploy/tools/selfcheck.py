"""Generate synthetic JIT policies and verify deploy end-to-end without ckpts."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from deploy.g1_pingpong.configuration import Deployment
from deploy.g1_pingpong.core import ACTOR_TERMS, TRACKER_TERMS
from deploy.g1_pingpong.policy import load_policy
from deploy.tools.compare_traces import compare
from deploy.tools.evaluate import run_mujoco
from deploy.tools.traces import replay_inputs


def synthetic_deployment(output, stage):
  """Build an in-memory test scene with separate synthetic policy/motion files."""
  from mjlab.scene import Scene
  from mjlab.tasks.g1_pingpong.config import g1_receiving_env_cfg, g1_tracking_env_cfg

  output = Path(output)
  if output.exists():
    raise FileExistsError(output)
  cfg = (
    g1_tracking_env_cfg(play=True) if stage == 1 else g1_receiving_env_cfg(play=True)
  )
  cfg.scene.num_envs = 1
  scene = Scene(cfg.scene, device="cpu")
  model = scene.compile()
  output.mkdir(parents=True)
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  names = list(cfg.actions["joint_pos"].actuator_names)
  joints = [model.joint("robot/" + name).id for name in names]
  acts = [
    int(np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0]) for joint in joints
  ]
  q = data.qpos[model.jnt_qposadr[joints]].copy()
  bodies = [i for i in range(model.nbody) if model.body(i).name.startswith("robot/")]
  body_names = [model.body(i).name.removeprefix("robot/") for i in bodies]
  (output / "motions").mkdir()
  np.savez_compressed(
    output / "motions/static.npz",
    joint_names=names,
    body_names=body_names,
    fps=50.0,
    strike_time_s=0.86,
    joint_pos=np.tile(q, (94, 1)),
    joint_vel=np.zeros((94, 29)),
    body_pos_w=np.tile(data.xpos[bodies], (94, 1, 1)),
    body_quat_w=np.tile(data.xquat[bodies], (94, 1, 1)),
  )
  for filename, inputs, outputs in [("tracker.pt", 132, 29)] + (
    [("receiving.pt", 186, 30)] if stage == 2 else []
  ):
    policy = torch.nn.Sequential(torch.nn.Linear(inputs, outputs)).eval()
    with torch.no_grad():
      for parameter in policy.parameters():
        parameter.zero_()
    torch.jit.script(policy).save(str(output / filename))
  scale = cfg.actions["joint_pos"].scale
  ball = asdict(cfg.commands["ball"]) if stage == 2 else {}
  manifest = {
    "synthetic": True,
    "stage": stage,
    "task": f"Mjlab-PingPong-{'Tracking' if stage == 1 else 'Receive'}-G1-{'Stage1-RandomDt' if stage == 1 else 'Stage2-Adaptive'}",
    "motions": ["static.npz"],
    "anchor": "torso_link",
    "control_dt": 0.02,
    "physics_dt": 0.005,
    "decimation": 4,
    "sim_options": asdict(cfg.sim.mujoco),
    "episode_length_s": 0.12,
    "tracker_terms": TRACKER_TERMS,
    "actor_terms": TRACKER_TERMS if stage == 1 else ACTOR_TERMS,
    "robot": {
      "joint_names": names,
      "default_q": q.tolist(),
      "kp": model.actuator_gainprm[acts, 0].tolist(),
      "kd": (-model.actuator_biasprm[acts, 2]).tolist(),
      "torque_limits": model.actuator_forcerange[acts, 1].tolist(),
      "joint_limits": model.jnt_range[joints].tolist(),
      "action_scale": [scale[name] for name in names],
    },
    "action": {
      "min_speed": 0.5,
      "max_speed": 2.0,
      "residual_clip": 2.0,
      "residual_scales": [0.25] * 22 + [1.0] * 7,
    },
    "ball": ball,
  }
  tracker = load_policy(output / "tracker.pt", 132, 29)
  actor = load_policy(output / "receiving.pt", 186, 30) if stage == 2 else None
  return Deployment(
    manifest,
    model,
    [output / "motions/static.npz"],
    tracker,
    actor,
    output / ("receiving.pt" if stage == 2 else "tracker.pt"),
    output / "tracker.pt",
  )


def selfcheck(output):
  output = Path(output)
  if output.exists():
    raise FileExistsError(f"Use a new selfcheck output directory: {output}")
  output.mkdir(parents=True)
  report = {"synthetic_only": True, "robot_commands_sent": False, "stages": {}}
  for stage in (1, 2):
    deployment = synthetic_deployment(output / f"stage{stage}", stage)
    validation = deployment.report()
    records = output / f"rollout{stage}"
    run_mujoco(deployment, output=records)
    reference = records / "trace_0_0.npz"
    candidate = records / "replayed.npz"
    samples = replay_inputs(deployment, reference, candidate)
    fields = [
      "tracker_obs",
      "base_action",
      "joint_action",
      "target_q",
      "reference_time",
      "reference_delta",
    ]
    if stage == 2:
      fields += ["actor_obs", "high_action"]
    comparison = compare(reference, candidate, fields)
    if not all(row["passed"] for row in comparison.values()):
      raise AssertionError("Synthetic deployment replay differs")
    report["stages"][str(stage)] = {
      "policy": validation,
      "samples": samples,
      "comparison": comparison,
    }
  (output / "selfcheck.json").write_text(json.dumps(report, indent=2))
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  print(json.dumps(selfcheck(args.output), indent=2))


if __name__ == "__main__":
  main()
