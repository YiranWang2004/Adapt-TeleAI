"""Stress staged deployment with asynchronous sensors and commands, offline only.

Uses the real Python control loop and the same plant as check_startup. All DDS
and mocap factories are replaced before run_real: no network or motor output.
"""

import argparse
import contextlib
import heapq
import json
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

from deploy.backends import unitree
from deploy.g1_pingpong.configuration import add_policy_arguments, from_arguments
from deploy.g1_pingpong.core import quaternion_multiply
from deploy.tools.check_real_loop import OfflinePlant


class DelayedPlant(OfflinePlant):
  def __init__(self, deployment, condition):
    self.condition = condition
    self.feedback_history = deque()
    self.feedback_ages = []
    self.live_buttons = 0
    self.root_pending, self.commands_pending = [], []
    self.sequence = 0
    self.previous_targets = {}
    self.metrics = defaultdict(
      lambda: dict(count=0, max_dq=0.0, max_target_step=0.0, steps=deque(maxlen=100))
    )
    super().__init__(deployment, "no_ball", 901, 2)
    self.entered_policy = condition.get("stable_only", False)
    self.minimum_z = 10.0
    step = self.robot.substep
    self.physics_substep = step

    def delayed_substep():
      # Apply the newest command whose delivery time has arrived. MuJoCo still
      # computes joint PD each physics step, with its configured torque limits.
      while self.commands_pending and self.commands_pending[0][0] <= self.now() + 1e-9:
        _, _, command = heapq.heappop(self.commands_pending)
        OfflinePlant.send(self, *command)
      self.physics_substep()
      state = self.robot.state()
      sampled = self.now() + self.cfg["physics_dt"]
      packet = (
        state.q,
        state.dq,
        state.gyro,
        self.live_buttons,
        5,
        state.root_quat,
        sampled,
      )
      self.feedback_history.append((sampled, packet))
      self.minimum_z = min(self.minimum_z, float(self.robot.state().root_pos[2]))

    self.robot.substep = delayed_substep

  def publish_root(self):
    state = self.robot.state()
    pos = state.root_pos + np.array([0.0, 0.0, self.condition.get("root_z_bias", 0.0)])
    angle = np.radians(self.condition.get("pitch_bias_deg", 0.0))
    roll = np.radians(self.condition.get("roll_bias_deg", 0.0))
    quat = quaternion_multiply(
      state.root_quat,
      quaternion_multiply(
        np.array([np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0]),
        np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0]),
      ),
    )
    self.sequence += 1
    heapq.heappush(
      self.root_pending,
      (
        self.now() + self.condition.get("root_delay", 0.0),
        self.sequence,
        self.message(pos, quat),
      ),
    )
    while self.root_pending and self.root_pending[0][0] <= self.now() + 1e-9:
      _, _, msg = heapq.heappop(self.root_pending)
      self.publish(msg, "root")

  def feedback(self):
    current = super().feedback()
    self.live_buttons = current[3]
    if self.feedback_history and abs(self.feedback_history[-1][0] - self.now()) < 1e-9:
      self.feedback_history.pop()
    self.feedback_history.append((self.now(), current))
    oldest_allowed = self.now() - self.condition.get("feedback_delay", 0.0)
    while (
      len(self.feedback_history) > 1
      and self.feedback_history[1][0] <= oldest_allowed + 1e-9
    ):
      self.feedback_history.popleft()
    if self.feedback_history[0][0] > oldest_allowed + 1e-9:
      return None
    self.feedback_ages.append(self.now() - self.feedback_history[0][1][-1])
    return self.feedback_history[0][1]

  def send(self, target, kp=None, kd=None):
    phase = self.engine.lifecycle.state
    if phase == "ARMED" and not self.engine.receiving:
      phase = "PING_HANDOFF"
    metrics = self.metrics[phase]
    metrics["count"] += 1
    metrics["max_dq"] = max(
      metrics["max_dq"], float(np.max(np.abs(self.robot.state().dq)))
    )
    if phase in self.previous_targets:
      delta = np.asarray(target) - self.previous_targets[phase]
      metrics["max_target_step"] = max(
        metrics["max_target_step"], float(np.max(np.abs(delta)))
      )
      metrics["steps"].append(float(np.mean(delta**2)))
    self.previous_targets[phase] = np.asarray(target).copy()
    self.sequence += 1
    heapq.heappush(
      self.commands_pending,
      (
        self.now() + self.condition.get("command_delay", 0.0),
        self.sequence,
        (target.copy(), kp.copy(), kd.copy()),
      ),
    )


def check_condition(deployment, runtime, condition, duration, folder):
  folder.mkdir(parents=True, exist_ok=False)
  plant = DelayedPlant(deployment, condition)
  controller = unitree.RallyController

  def make_controller(*args, **kwargs):
    plant.engine = controller(*args, **kwargs)
    return plant.engine

  error = None
  with contextlib.ExitStack() as stack:
    stack.enter_context(
      contextlib.redirect_stdout(stack.enter_context((folder / "run.log").open("w")))
    )
    stack.enter_context(patch.object(unitree, "RallyController", make_controller))
    stack.enter_context(patch.object(unitree, "DDSBackend", lambda *args: plant))
    stack.enter_context(
      patch.object(unitree, "build_mocap", lambda config: plant.provider)
    )
    stack.enter_context(
      patch.object(
        unitree,
        "time",
        SimpleNamespace(
          monotonic=plant.now, sleep=plant.sleep, time=lambda: 10000 + plant.now()
        ),
      )
    )
    try:
      unitree.run_real(
        deployment,
        {**runtime, "io_process": False},
        "deploy/configs/mocap_hope.yaml",
        enable_motors=True,
        motion_index=19,
        motion_allow=[19, 24],
        duration=duration,
      )
    except (ValueError, RuntimeError, TimeoutError) as exc:
      error = str(exc)
  metrics = {}
  for phase, values in plant.metrics.items():
    steps = values.pop("steps")
    metrics[phase] = {
      **values,
      "tail_target_step_rms": float(np.sqrt(np.mean(steps))) if steps else 0.0,
    }
  result = dict(
    condition=condition,
    error=error,
    final_state=plant.engine.lifecycle.state,
    final_root_z=float(plant.robot.state().root_pos[2]),
    minimum_root_z=plant.minimum_z,
    simulated_seconds=plant.now(),
    metrics=metrics,
    actual_robot_commands_sent=False,
    actual_network_connections=False,
    virtual_time=True,
    startup_condition="suspended positioning, operator placement",
    completed_swings=plant.engine.completed,
  )
  result["feedback_age_quantiles"] = (
    np.quantile(plant.feedback_ages, [0.0, 0.5, 1.0]).tolist()
    if plant.feedback_ages
    else []
  )
  (folder / "result.json").write_text(json.dumps(result, indent=2))
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  add_policy_arguments(parser)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--duration", type=float, default=20.0)
  parser.add_argument(
    "--stable-only", action="store_true", help="Remain in AMP after A; never press UP"
  )
  args = parser.parse_args()
  torch.set_num_threads(1)
  deployment = from_arguments(args)
  runtime = yaml.safe_load(Path("deploy/configs/real.yaml").read_text())
  args.output.mkdir(parents=True, exist_ok=False)
  cases = [("sync", {})]
  for key in ["feedback_delay", "command_delay", "root_delay"]:
    for delay in [0.005, 0.01, 0.02, 0.04]:
      cases.append((f"{key}_{round(delay * 1000)}ms", {key: delay}))
  cases.extend(
    [
      ("feedback10_command10", dict(feedback_delay=0.01, command_delay=0.01)),
      ("marker_pitch_plus5", dict(pitch_bias_deg=5.0)),
      ("marker_pitch_minus5", dict(pitch_bias_deg=-5.0)),
      ("root_height_minus80cm", dict(root_z_bias=-0.8)),
    ]
  )
  results = []
  for name, case in cases:
    if args.stable_only:
      case = {**case, "stable_only": True}
    result = check_condition(
      deployment, runtime, case, args.duration, args.output / name
    )
    results.append(dict(name=name, **result))
    print(
      json.dumps(
        dict(
          name=name,
          error=result["error"],
          state=result["final_state"],
          z=result["final_root_z"],
          metrics=result["metrics"],
        )
      ),
      flush=True,
    )
    (args.output / "report.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
  main()
