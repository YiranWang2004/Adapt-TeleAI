"""Exercise run_real with a MuJoCo plant and HOPE packets; never open SDK/ROS IO.

Virtual time checks the control/data path, not hardware scheduling performance.
"""

import argparse
import contextlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch
import yaml

from deploy.backends import unitree
from deploy.backends.mujoco import MujocoBackend
from deploy.g1_pingpong.configuration import add_policy_arguments, from_arguments
from deploy.g1_pingpong.feeder import RandomBallFeeder
from deploy.g1_pingpong.perception import BallEstimator, RootPoseCache
from deploy.io.mocap import input_packet, read_config
from deploy.io.transport import UDPMocap


class OfflinePlant:
  def __init__(self, deployment, scenario, seed, swings):
    self.cfg = deployment.settings
    self.scenario, self.swings = scenario, swings
    self.robot = MujocoBackend(copy.copy(deployment.model), self.cfg)
    self.engine = None
    self.ticks = self.sent = self.hits = 0
    self.closed = self.provider_closed = self.halted = False
    self.armed_once = False
    self.entered_policy = False
    self.support_released = False
    self.switched_back = False
    self.staged = (
      yaml.safe_load(Path("deploy/configs/real.yaml").read_text())
      .get("stable_policy", {})
      .get("enabled", False)
    )
    self.ball = self.next_serve = self.first_swing = None
    self.previous_phase = None
    self.last_send = None
    self.mocap_cfg, _ = read_config("deploy/configs/mocap_hope.yaml")
    self.receiver = UDPMocap.__new__(UDPMocap)
    self.receiver.cfg, self.receiver.accepted = self.mocap_cfg, 0
    self.receiver.ball = BallEstimator(**self.mocap_cfg["estimator"])
    self.receiver.root = RootPoseCache([0, 0, 0], 0.1)
    self.options = yaml.safe_load(Path("deploy/configs/mujoco.yaml").read_text())
    self.feeder = RandomBallFeeder(
      self.cfg["ball"], self.options["feed"], np.random.default_rng(seed)
    )
    self.provider = SimpleNamespace(
      ball=self.receiver.ball, root=self.receiver.root, close=self.close_provider
    )
    self.publish_root()

  def now(self):
    return self.ticks * self.cfg["physics_dt"]

  def message(self, position, quaternion=None):
    # Reverse the template HOPE table-height translation at the sensor side.
    p = np.asarray(position) - self.mocap_cfg["transform"]["translation"]
    q = np.array([1, 0, 0, 0]) if quaternion is None else quaternion
    pose = SimpleNamespace(
      position=SimpleNamespace(x=p[0], y=p[1], z=p[2]),
      orientation=SimpleNamespace(w=q[0], x=q[1], y=q[2], z=q[3]),
    )
    sec, ns = divmod(round((10000 + self.now()) * 1e9), 10**9)
    return SimpleNamespace(
      header=SimpleNamespace(
        frame_id="world", stamp=SimpleNamespace(sec=sec, nanosec=ns)
      ),
      pose=pose,
      poses=[pose],
    )

  def publish(self, msg, kind):
    self.receiver.accept(
      input_packet(msg, kind, self.mocap_cfg),
      wall_time=10000 + self.now(),
      monotonic_time=self.now(),
    )

  def publish_root(self):
    state = self.robot.state()
    self.publish(self.message(state.root_pos, state.root_quat), "root")

  def feedback(self):
    assert self.engine is not None
    if self.scenario == "dds_dropout" and self.now() >= 8:
      raise TimeoutError("Injected DDS feedback expiration")
    if self.scenario == "stop" and self.now() >= 8:
      self.halted = True
    state = self.robot.state()
    phase = self.engine.lifecycle.state
    buttons = 0
    if (
      self.scenario == "switch_back"
      and self.engine.recovered >= self.swings
      and not self.switched_back
    ):
      buttons = 1 << 14
      self.switched_back = True
    elif phase == "WAIT":
      buttons = 1 << 2
    elif phase == "STABLE" and not self.entered_policy:
      buttons = 1 << 12
      self.entered_policy = True
    elif phase == "POSE_HOLD" and not self.armed_once:
      buttons = 1 << 8
      self.armed_once = True
    return state.q, state.dq, state.gyro, buttons, 5, state.root_quat, self.now()

  def send(self, target, kp=None, kd=None):
    assert not self.closed and not self.halted and self.scenario != "readonly"
    if kp is not None:
      self.robot.model.actuator_gainprm[self.robot.actuators, 0] = kp
      self.robot.model.actuator_biasprm[self.robot.actuators, 1] = -kp
      self.robot.model.actuator_biasprm[self.robot.actuators, 2] = -kd
    self.robot.set_target(target)
    self.sent += 1
    self.last_send = self.now()

  def close(self):
    self.closed = True
    self.robot.damp()

  def close_provider(self):
    self.provider_closed = True

  def sleep(self, delay):
    assert self.engine is not None
    phase = self.engine.lifecycle.state
    if self.engine.completed >= self.swings:
      self.engine.receiving = False
    if phase == "EXECUTE" and self.first_swing is None:
      self.first_swing = self.now()
    if self.previous_phase == "RECOVER" and phase == "READY":
      self.hits += bool(self.ball and self.ball.hit)
      self.ball, self.next_serve = None, None
    self.previous_phase = phase
    if (
      phase == "ARMED"
      and self.engine.receiving
      and self.scenario not in ("no_ball", "readonly")
    ):
      if self.next_serve is None:
        self.next_serve = self.now() + self.options["feed"]["serve_gap_seconds"]
      if self.ball is None and self.now() >= self.next_serve:
        self.ball = self.feeder.sample(self.engine.candidates(self.robot.state()))
        self.receiver.ball = BallEstimator(**self.mocap_cfg["estimator"])
        self.provider.ball = self.receiver.ball
    # Model the documented suspended positioning and operator placement.
    # This is an explicit test fixture, not a sensor-based landing detector.
    if self.staged and phase == "TO_STABLE" and not self.support_released:
      i, v = self.robot.free_qpos, self.robot.free_qvel
      self.robot.data.qpos[i : i + 3] = [0.0, 0.0, 0.783675]
      self.robot.data.qpos[i + 3 : i + 7] = [1.0, 0.0, 0.0, 0.0]
      self.robot.data.qvel[v : v + 6] = 0
      self.support_released = True
    for _ in range(max(1, round(delay / self.cfg["physics_dt"]))):
      if self.scenario != "readonly":
        self.robot.substep()
        if self.staged and not self.support_released:
          i, v = self.robot.free_qpos, self.robot.free_qvel
          self.robot.data.qpos[i : i + 3] = [0.0, 0.0, 0.90]
          self.robot.data.qpos[i + 3 : i + 7] = [1.0, 0.0, 0.0, 0.0]
          self.robot.data.qvel[v : v + 6] = 0
          mujoco.mj_forward(self.robot.model, self.robot.data)
      self.ticks += 1
      if self.ball is not None:
        self.ball.substep(self.robot.state(), self.cfg["physics_dt"])
      if self.ticks % 2 == 0:
        if not (self.scenario == "root_dropout" and self.now() >= 8):
          self.publish_root()
        drop_ball = (
          self.scenario == "ball_dropout"
          and self.first_swing is not None
          and self.now() > self.first_swing + 0.12
        )
        if self.ball is not None and not drop_ball:
          self.publish(self.message(self.ball.position), "ball")


def check(deployment, output, scenario, seed=901, swings=30):
  output.mkdir(parents=True, exist_ok=False)
  plant = OfflinePlant(deployment, scenario, seed, swings)
  runtime = yaml.safe_load(Path("deploy/configs/real.yaml").read_text())
  runtime["io_process"] = False
  controller = unitree.RallyController

  def create_controller(*args, **kwargs):
    plant.engine = controller(*args, **kwargs)
    return plant.engine

  duration = (
    260
    if scenario == "normal"
    else 60
    if scenario == "no_ball"
    else 30
    if scenario == "switch_back"
    else 20
  )
  error = None
  with contextlib.ExitStack() as stack:
    log = stack.enter_context((output / "run.log").open("w"))
    stack.enter_context(contextlib.redirect_stdout(log))
    stack.enter_context(patch.object(unitree, "RallyController", create_controller))
    stack.enter_context(patch.object(unitree, "DDSBackend", lambda *a: plant))
    stack.enter_context(
      patch.object(unitree, "build_mocap", lambda cfg: plant.provider)
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
        runtime,
        "deploy/configs/mocap_hope.yaml",
        enable_motors=scenario != "readonly",
        motion_index=19,
        motion_allow=[19, 24],
        duration=duration,
        output=output,
      )
    except (ValueError, TimeoutError, RuntimeError) as exc:
      error = str(exc)
  summary = json.loads((output / "summary.json").read_text())
  expected = {
    "root_dropout": "Robot mocap pose expired",
    "ball_dropout": "Ball positions expired during swing",
    "dds_dropout": "Injected DDS feedback expiration",
  }
  if scenario in expected:
    passed = error == expected[scenario] and summary["state"] == "HALT"
  elif scenario == "stop":
    passed = (
      error is None
      and summary["state"] == "HALT"
      and plant.last_send is not None
      and plant.last_send < 8
    )
  elif scenario == "readonly":
    passed = error is None and plant.sent == 0 and summary["state"] == "READ_ONLY"
  elif scenario == "switch_back":
    passed = (
      error is None
      and summary["state"] == "STABLE"
      and summary["recovered_swings"] == swings
    )
  elif scenario == "no_ball":
    passed = (
      error is None and summary["state"] == "ARMED" and not summary["completed_swings"]
    )
  else:
    recovered_times = [
      e["time"] for e in summary["events"] if e["event"] == "recovered"
    ]
    passed = (
      error is None
      and summary["recovered_swings"] == swings
      and summary["state"] == "READY"
      and bool(recovered_times)
      and plant.now() - recovered_times[-1] >= 5
    )
  passed &= plant.closed and plant.provider_closed
  report = dict(
    scenario=scenario,
    seed=seed,
    passed=bool(passed),
    error=error,
    completed=summary["completed_swings"],
    recovered=summary["recovered_swings"],
    hits=plant.hits,
    simulated_seconds=plant.now(),
    commands_to_simulator=plant.sent,
    actual_robot_commands_sent=False,
    actual_network_connections=False,
    virtual_time=True,
    closed=plant.closed,
    provider_closed=plant.provider_closed,
  )
  (output / "check.json").write_text(json.dumps(report, indent=2))
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  add_policy_arguments(parser)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(1)
  deployment = from_arguments(args)
  args.output.mkdir(parents=True, exist_ok=False)
  rows = []
  for scenario, seed in [("normal", s) for s in (901, 902, 903)] + [
    (s, 901)
    for s in (
      "no_ball",
      "readonly",
      "root_dropout",
      "ball_dropout",
      "dds_dropout",
      "stop",
    )
  ]:
    row = check(deployment, args.output / f"{scenario}_{seed}", scenario, seed)
    rows.append(row)
    print(json.dumps(row), flush=True)
    (args.output / "report.json").write_text(json.dumps(rows, indent=2))
  if not all(row["passed"] for row in rows):
    raise SystemExit(1)


if __name__ == "__main__":
  main()
