"""Offline G1 plant study. Every DDS/mocap factory is replaced before run_real.

Unknown plant parameters are sensitivity hypotheses, not hardware identification.
Use independent cases; production real.yaml and the policy are never modified.
"""

import argparse
import copy
import json
from collections import deque
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
import yaml

from deploy.g1_pingpong.configuration import load_deployment
from deploy.g1_pingpong.core import rotation
from deploy.tools import check_sim_real_gap as gap
from deploy.tools.analyze_latency import load_chunks


class RealismPlant(gap.DelayedPlant):
  def __init__(self, deployment, condition):
    self.records = []
    self.physics_records = []
    self.injected = False
    self.policy_start = None
    self.root_publish_time = None
    self.drive_history = deque()
    self.drive_effort = np.zeros(29)
    self.drive_velocity = None
    self.drive_velocity_history = deque()
    self.feedback_velocity = None
    self.filtered_target = None
    self.source_rows = None
    self.host_profile = None
    self.report_history = deque()
    if condition.get("recording"):
      folder = Path(condition["recording"])
      if not folder.exists():
        original = Path("deploy/recordings") / folder.name
        if original.exists():
          folder = original
      self.resolved_recording = str(folder.resolve())
      self.source_rows = [
        json.loads(x) for x in (folder / "trace.jsonl").read_text().splitlines()
      ]
      rows = [r for r in self.source_rows if r["state"] in ("TO_POLICY", "ARMED")]
      writes = load_chunks(folder / "command_writes", "commands_*.npy")
      first = {
        int(w["command_id"]): w for w in writes if w["first_write"] and not w["halted"]
      }
      self.host_profile = [
        dict(
          feedback_delay=r["latency"]["callback_to_consumption_seconds"],
          command_delay=max(
            0,
            float(
              first[r["latency"]["command_id"]]["write_started_time"]
              - r["command_generated_time"]
            ),
          ),
          root_delay=max(0, r.get("root_age", 0) or 0),
        )
        for r in rows
      ]
    super().__init__(deployment, condition)
    self.scenario = condition.get("scenario", "no_ball")
    self.swings = condition.get("swings", 3)
    self.feeder.rng = np.random.default_rng(condition.get("seed", 901))
    r = self.robot
    m = r.model
    self.waist = np.array(
      [
        i
        for i, n in enumerate(self.cfg["robot"]["joint_names"])
        if n.startswith("waist_")
      ]
    )
    groups = {
      "waist": ("waist_",),
      "legs": (
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
      ),
      "ankles": ("left_ankle", "right_ankle"),
      "arms": (
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
      ),
      "wrists": ("left_wrist", "right_wrist"),
    }
    group = (
      "waist" if condition.get("drive_waist_only") else condition.get("drive_group")
    )
    self.unfiltered = np.array(
      [
        i
        for i, name in enumerate(self.cfg["robot"]["joint_names"])
        if group and not name.startswith(groups[group])
      ],
      dtype=int,
    )
    m.dof_armature[r.qvel] *= condition.get("armature_scale", 1)
    m.dof_armature[r.qvel[self.waist]] *= condition.get("waist_armature_scale", 1)
    for name in ["waist_yaw_link", "waist_roll_link", "torso_link"]:
      b = m.body("robot/" + name).id
      m.body_inertia[b] *= condition.get("torso_inertia_scale", 1)
    m.body_ipos[m.body("robot/torso_link").id] += np.asarray(
      condition.get("torso_com_offset", [0, 0, 0])
    )
    if condition.get("wrist_mass_scale") is not None:
      b = m.body("robot/right_wrist_yaw_link").id
      m.body_mass[b] *= condition["wrist_mass_scale"]
      m.body_inertia[b] *= condition["wrist_mass_scale"]
    if condition.get("extra_payload_kg"):
      self.add_payload(condition["extra_payload_kg"])
    if condition.get("friction") is not None:
      m.geom_friction[:, 0] = condition["friction"]
    if condition.get("passive_damping"):
      m.dof_damping[r.qvel[self.waist]] += condition["passive_damping"]
    mujoco.mj_setConst(m, r.data)
    self.nominal_substep = self.physics_substep
    self.physics_substep = self.drive_substep

  def add_payload(self, mass):
    r = self.robot
    m = r.model
    b = int(m.site_bodyid[r.racket])
    point = m.site_pos[r.racket]
    old_mass = m.body_mass[b]
    old_com = m.body_ipos[b].copy()
    new_com = (old_mass * old_com + mass * point) / (old_mass + mass)
    R = rotation(m.body_iquat[b])
    inertia = R @ np.diag(m.body_inertia[b]) @ R.T
    for weight, offset in [(old_mass, old_com - new_com), (mass, point - new_com)]:
      inertia += weight * ((offset @ offset) * np.eye(3) - np.outer(offset, offset))
    eig, R = np.linalg.eigh(inertia)
    if np.linalg.det(R) < 0:
      R[:, 0] *= -1
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.reshape(-1))
    m.body_mass[b] = old_mass + mass
    m.body_ipos[b] = new_com
    m.body_inertia[b] = eig
    m.body_iquat[b] = quat

  def publish_root(self):
    # Keep mocap at the original 100Hz when physics is refined to 1ms.
    if (
      self.root_publish_time is not None
      and self.now() - self.root_publish_time < 0.01 - 1e-9
    ):
      return
    self.root_publish_time = self.now()
    return super().publish_root()

  def apply_profile(self):
    if self.engine is None or self.engine.lifecycle.state not in (
      "ARMED",
      "EXECUTE",
      "RECOVER",
    ):
      return
    if self.policy_start is None:
      self.policy_start = self.now()
    if self.host_profile and self.condition.get("measured_timing", False):
      i = int(round((self.now() - self.policy_start) / 0.02)) % len(self.host_profile)
      self.condition.update(self.host_profile[i])
      self.condition["feedback_delay"] += self.condition.get("extra_feedback_delay", 0)
      self.condition["command_delay"] += self.condition.get("extra_command_delay", 0)
      if self.condition.get("cap_feedback_age"):
        self.condition["feedback_delay"] = min(
          self.condition["feedback_delay"], self.condition["cap_feedback_age"]
        )

  def inject_entry(self):
    if self.injected or not self.source_rows:
      return
    stage = self.condition.get("entry_stage")
    if not stage:
      return
    phase = self.engine.lifecycle.state
    before_up = phase == "STABLE" and not self.entered_policy
    before_policy = phase == "STABLE" and any(
      x.get("state") == "POLICY_POSTURE" for x in self.records
    )
    if not ((stage == "up" and before_up) or (stage == "policy" and before_policy)):
      return
    rows = self.source_rows
    index = next(
      i
      for i, x in enumerate(rows)
      if x["state"] == ("POLICY_POSTURE" if stage == "up" else "TO_POLICY")
    )
    entry = rows[index]
    r = self.robot
    i, v = r.free_qpos, r.free_qvel
    r.data.qpos[r.qpos] = entry["q"]
    r.data.qvel[r.qvel] = entry["dq"]
    r.data.qpos[i : i + 3] = [0, 0, 1]
    r.data.qpos[i + 3 : i + 7] = entry["root_quat"]
    r.data.qvel[v : v + 3] = 0
    r.data.qvel[v + 3 : v + 6] = entry.get("imu_gyro", entry["gyro"])
    r.data.qacc_warmstart[:] = 0
    mujoco.mj_forward(r.model, r.data)
    r.data.qpos[i + 2] -= self.foot_bottom()
    mujoco.mj_forward(r.model, r.data)
    if hasattr(self, "stable"):
      history = [
        x["stable_policy"]["observation"]
        for x in rows[:index]
        if (x.get("stable_policy") or {}).get("observation") is not None
      ][-10:]
      self.stable.history[:] = np.array(history)[None]
      self.stable.action[self.stable.used] = rows[index - 1]["stable_policy"]["action"]
    self.feedback_history.clear()
    self.root_pending.clear()
    self.commands_pending.clear()
    self.root_publish_time = None
    self.publish_root()
    self.injected = True

  def foot_bottom(self):
    r = self.robot
    m, d = r.model, r.data
    bottom = []
    for g in range(m.ngeom):
      if m.geom_bodyid[g] not in r.feet or not (
        m.geom_contype[g] or m.geom_conaffinity[g]
      ):
        continue
      R = d.geom_xmat[g].reshape(3, 3)
      pos = d.geom_xpos[g]
      s = m.geom_size[g]
      kind = m.geom_type[g]
      if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        bottom.append(pos[2] - s[0] * np.linalg.norm(R[2, :2]) - s[1] * abs(R[2, 2]))
      elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        bottom.append(pos[2] - np.sum(abs(R[2]) * s))
      elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        bottom.append(pos[2] - s[0])
      elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        bottom.append(pos[2] - s[0] - s[1] * abs(R[2, 2]))
    return min(bottom)

  def feedback(self):
    if self.engine is not None:
      self.inject_entry()
    self.apply_profile()
    packet = super().feedback()
    if (
      packet is not None
      and self.condition.get("report_drive_velocity", False)
      and self.drive_velocity is not None
    ):
      values = list(packet)
      # q and dq belong to the SAME delayed packet, even though the drive's
      # velocity estimator runs locally at the physics rate. Using today's
      # filtered dq with yesterday's q would erase the measured host latency
      # from only one observation term and create an artificial plant.
      reported = self.reported_velocity_at(packet[-1])
      if reported is not None:
        values[1] = reported
      packet = tuple(values)
    if packet is not None and self.condition.get("sensor_velocity_delay", 0):
      # Internal encoder/velocity estimation latency, distinct from the host
      # packet timestamp. Preserve current q and metadata, delay only dq.
      self.report_history.append((self.now(), np.array(packet[1]).copy()))
      cutoff = self.now() - self.condition["sensor_velocity_delay"]
      while len(self.report_history) > 1 and self.report_history[1][0] <= cutoff:
        self.report_history.popleft()
      values = list(packet)
      values[1] = self.report_history[0][1].copy()
      packet = tuple(values)
    if packet is not None and self.condition.get("observation_velocity_tau", 0) > 0:
      values = list(packet)
      dq = np.array(packet[1])
      alpha = 1 - np.exp(
        -self.cfg["control_dt"] / self.condition["observation_velocity_tau"]
      )
      if self.feedback_velocity is None:
        self.feedback_velocity = dq.copy()
      self.feedback_velocity += alpha * (dq - self.feedback_velocity)
      dq[self.waist] = self.feedback_velocity[self.waist]
      values[1] = dq
      packet = tuple(values)
    return packet

  def reported_velocity_at(self, stamp):
    for time, velocity in reversed(self.drive_velocity_history):
      if time <= stamp + 1e-9:
        return velocity.copy()
    return None

  def send(self, target, kp=None, kd=None):
    self.apply_profile()
    target = np.array(target).copy()
    tau = self.condition.get("target_filter_tau", 0)
    phase = self.engine.lifecycle.state
    wait_only = self.condition.get("target_filter_wait_only", False)
    filtering = phase in ("ARMED", "EXECUTE", "RECOVER", "READY") and not (
      wait_only and phase in ("EXECUTE", "RECOVER")
    )
    if tau and filtering:
      if self.filtered_target is None:
        self.filtered_target = target.copy()
      self.filtered_target += (1 - np.exp(-self.cfg["control_dt"] / tau)) * (
        target - self.filtered_target
      )
      if self.condition.get("target_filter_all", False):
        target = self.filtered_target.copy()
      else:
        target[self.waist] = self.filtered_target[self.waist]
    elif tau and wait_only:
      # Re-entering waiting must start from the last command actually sent,
      # rather than the pre-swing filter state left several seconds ago.
      self.filtered_target = target.copy()
    super().send(target, kp, kd)
    s = self.robot.state()
    phase = self.engine.lifecycle.state
    self.records.append(
      dict(
        time=self.now(),
        state=phase,
        q=s.q.tolist(),
        dq=s.dq.tolist(),
        target_q=target.tolist(),
        root_z=float(s.root_pos[2]),
        kp=np.asarray(kp).tolist(),
        kd=np.asarray(kd).tolist(),
      )
    )

  def drive_substep(self):
    r = self.robot
    m, d = r.model, r.data
    dt = self.cfg["physics_dt"]
    if self.condition.get("explicit_drive", False):
      a = r.actuators
      kp = m.actuator_gainprm[a, 0].copy()
      bias = m.actuator_biasprm[a, :].copy()
      kd = -bias[:, 2]
      velocity = d.qvel[r.qvel].copy()
      tv = self.condition.get("drive_velocity_tau", 0)
      if self.drive_velocity is None:
        self.drive_velocity = velocity.copy()
      self.drive_velocity += (1 - np.exp(-dt / tv) if tv else 1) * (
        velocity - self.drive_velocity
      )
      self.drive_velocity[self.unfiltered] = velocity[self.unfiltered]
      effort = kp * (d.ctrl[a] - d.qpos[r.qpos]) - kd * self.drive_velocity
      effort *= self.condition.get("drive_gain", 1)
      self.drive_history.append((self.now(), effort.copy()))
      cutoff = self.now() - self.condition.get("torque_delay", 0)
      while len(self.drive_history) > 1 and self.drive_history[1][0] <= cutoff + 1e-9:
        self.drive_history.popleft()
      effort = (
        self.drive_history[0][1]
        if self.drive_history[0][0] <= cutoff + 1e-9
        else np.zeros(29)
      )
      tau = self.condition.get("torque_tau", 0)
      self.drive_effort += (1 - np.exp(-dt / tau) if tau else 1) * (
        effort - self.drive_effort
      )
      bounds = m.actuator_forcerange[a]
      force = np.clip(self.drive_effort, bounds[:, 0], bounds[:, 1])
      m.actuator_gainprm[a, 0] = 0
      m.actuator_biasprm[a, :] = 0
      d.qfrc_applied[r.qvel] = force
      self.nominal_substep()
      m.actuator_gainprm[a, 0] = kp
      m.actuator_biasprm[a, :] = bias
      self.drive_velocity_history.append((self.now() + dt, self.drive_velocity.copy()))
      while self.drive_velocity_history[0][0] < self.now() - 0.2:
        self.drive_velocity_history.popleft()
    else:
      self.nominal_substep()
    if (
      self.policy_start is not None
      and self.now() - self.policy_start >= 2
      and not getattr(self, "kicked", False)
      and self.condition.get("kick", False)
    ):
      d.qvel[r.qvel[12]] += 0.1
      self.kicked = True
    # Physics sampling avoids the 50Hz trace's Nyquist ambiguity.
    if self.policy_start is not None and (
      not self.physics_records
      or self.now() - self.physics_records[-1][0] >= 0.002 - 1e-9
    ):
      self.physics_records.append(
        (self.now(), d.qpos[r.qpos].copy(), d.qvel[r.qvel].copy())
      )


def assess(plant, result):
  rows = plant.records
  start = plant.policy_start
  result["entry_injected"] = plant.injected
  result["resolved_recording"] = getattr(plant, "resolved_recording", None)
  result["policy_start"] = start
  result["simulated_hits"] = plant.hits
  if not rows or start is None:
    return result
  active = [x for x in rows if x["time"] >= start + 2]
  physics = [x for x in plant.physics_records if x[0] >= start + 2]
  if len(physics) < 20:
    return result
  t = np.array([x[0] for x in physics])
  q = np.array([x[1] for x in physics])
  dq = np.array([x[2] for x in physics])
  target = np.array([x["target_q"] for x in active])
  tail = t >= max(start + 2, t[-1] - 5)
  grid = np.arange(t[0], t[-1], 0.002)
  yaw = np.interp(grid, t, dq[:, 12])
  yaw -= yaw.mean()
  freq = np.fft.rfftfreq(len(yaw), 0.002)
  power = abs(np.fft.rfft(yaw * np.hanning(len(yaw)))) ** 2
  peak = (freq >= 5) & (freq <= 80)
  band = (freq >= 20) & (freq <= 30)
  result["assessment"] = dict(
    waist_dq_rms=float(np.sqrt(np.mean(dq[:, 12] ** 2))),
    waist_tail_rms=float(np.sqrt(np.mean(dq[tail, 12] ** 2))),
    waist_dq_peak=float(abs(dq[:, 12]).max()),
    waist_q_peak_to_peak=float(np.ptp(q[:, 12])),
    all_tail_rms=float(np.sqrt(np.mean(dq[tail] ** 2))),
    max_target_step=float(abs(np.diff(target, axis=0)).max()) if len(target) > 1 else 0,
    waist_max_target_step=float(abs(np.diff(target[:, 12])).max())
    if len(target) > 1
    else 0,
    waist_peak_hz=float(freq[peak][np.argmax(power[peak])]),
    waist_20_30hz_fraction=float(power[band].sum() / max(power[peak].sum(), 1e-30)),
    min_active_height=float(min(x["root_z"] for x in active)),
    post_handoff_seconds=float(t[-1] - (start + 2)),
  )
  return result


def run_case(deployment, runtime, case, folder, duration):
  instances = []
  amp = gap.unitree.StableAMP

  class Capture(RealismPlant):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      instances.append(self)

  def capture_amp(*args, **kwargs):
    stable = amp(*args, **kwargs)
    instances[-1].stable = stable
    return stable

  cfg = copy.deepcopy(deployment.settings)
  cfg["physics_dt"] = case.get("physics_dt", 0.001)
  cfg["sim_options"]["timestep"] = cfg["physics_dt"]
  cfg["control_dt"] = case.get("control_dt", 0.02)
  for index, name in enumerate(cfg["robot"]["joint_names"]):
    if name.startswith("waist_"):
      cfg["robot"]["kp"][index] *= case.get("waist_kp_scale", 1)
  active = replace(deployment, settings=cfg)
  rt = copy.deepcopy(runtime)
  if case.get("policy_pd"):
    rt["policy_pd"] = case["policy_pd"]
  if case.get("handoff_seconds"):
    rt["stable_policy"]["transition_seconds"] = case["handoff_seconds"]
  with (
    patch.object(gap, "DelayedPlant", Capture),
    patch.object(gap.unitree, "StableAMP", capture_amp),
  ):
    result = gap.check_condition(active, rt, dict(case), duration, folder)
  plant = instances[-1]
  result = assess(plant, result)
  result["requested_condition"] = copy.deepcopy(case)
  result["requested_duration"] = duration
  result["study_revision"] = "20261015-realism-v2"
  np.savez_compressed(
    folder / "signals.npz",
    time=[x["time"] for x in plant.records],
    q=[x["q"] for x in plant.records],
    dq=[x["dq"] for x in plant.records],
    target=[x["target_q"] for x in plant.records],
    phase=[x["state"] for x in plant.records],
    physics_time=[x[0] for x in plant.physics_records],
    physics_q=[x[1] for x in plant.physics_records],
    physics_dq=[x[2] for x in plant.physics_records],
  )
  (folder / "result.json").write_text(json.dumps(result, indent=2))
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cases", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--duration", type=float, default=20)
  args = parser.parse_args()
  torch.set_num_threads(1)
  c = yaml.safe_load(Path("deploy/configs/g1_tele_gmr.yaml").read_text())
  deployment = load_deployment(**{k: v for k, v in c.items() if v is not None})
  runtime = yaml.safe_load(Path("deploy/configs/real.yaml").read_text())
  cases = json.loads(args.cases.read_text())
  args.output.mkdir(parents=True, exist_ok=True)
  reports = []
  for case in cases:
    name = case.pop("name")
    folder = args.output / name
    if (folder / "result.json").exists():
      result = json.loads((folder / "result.json").read_text())
      if (
        result.get("requested_condition") != case
        or result.get("requested_duration") != args.duration
      ):
        raise ValueError(
          f"Existing case has different or unversioned settings: {folder}; "
          "use a new output directory"
        )
    else:
      result = run_case(deployment, runtime, case, folder, args.duration)
    reports.append(dict(name=name, **result))
    (args.output / "results.json").write_text(json.dumps(reports, indent=2))
    print(
      json.dumps(
        dict(
          name=name,
          error=result["error"],
          injected=result.get("entry_injected"),
          metrics=result.get("assessment"),
        )
      ),
      flush=True,
    )


if __name__ == "__main__":
  main()
