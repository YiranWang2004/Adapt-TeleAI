"""Pure sampling: no robot, motion library, selector or policy dependencies."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace

import numpy as np

from deploy.benchmarks.racket_command import domain as coverage
from deploy.racket_command import RacketCommand


@dataclass(frozen=True)
class Trial:
  trial_id: int
  cell_y: int
  cell_z: int
  position: tuple
  normal: tuple
  velocity: tuple
  tts: float
  point_id: int | None = None
  direction_id: int | None = None
  speed_id: int | None = None
  tts_id: int | None = None

  def command(self, now):
    return RacketCommand(
      np.asarray(self.position),
      np.asarray(self.velocity),
      np.asarray(self.normal),
      now + self.tts,
      dict(
        source="independent_grid",
        trial_id=self.trial_id,
        position=list(self.position),
        striking_normal=list(self.normal),
        velocity=list(self.velocity),
        remaining=self.tts,
      ),
    )


def validate(config):
  if config.get("spatial_coverage", "support_union") not in (
    "support_union",
    "full_rectangle",
  ):
    raise ValueError("Unknown spatial_coverage")
  if config.get("spatial_coverage") == "full_rectangle" and (
    config.get("position_sampling") != "random"
    or config.get("rally_constraints", {}).get("enabled")
  ):
    raise ValueError("full_rectangle requires random positions without rally filtering")
  if "tts_values" in config:
    times = np.asarray(config["tts_values"], dtype=float)
    if (
      times.ndim != 1
      or not len(times)
      or not np.isfinite(times).all()
      or np.any(times < 0.15)
      or np.any(times >= config["episode_seconds"])
      or len(np.unique(times)) != len(times)
    ):
      raise ValueError("tts_values must be distinct finite TTS >= .15s within episode")
    if config.get("sweep"):
      raise ValueError("tts_values uses samples_per_cell per TTS; remove sweep")
  if (
    config.get("rally_constraints", {}).get("enabled")
    and config.get("position_sampling") != "fixed_per_cell"
  ):
    raise ValueError("Rally-constrained sampling requires fixed_per_cell positions")
  rally = config.get("rally_constraints", {})
  if rally.get("enabled"):
    for key in ("incoming_bank_size", "max_direction_attempts", "speed_probe_levels"):
      if not isinstance(rally.get(key), int) or rally[key] < 1:
        raise ValueError(f"rally_constraints.{key} must be a positive integer")
    if rally["speed_probe_levels"] < config.get("sweep", {}).get("speed_levels", 1):
      raise ValueError("speed_probe_levels must cover all requested speed levels")
    for key in ("landing_margin", "net_clearance", "minimum_speed_span"):
      if not np.isfinite(rally.get(key, 0)) or rally.get(key, 0) < 0:
        raise ValueError(f"rally_constraints.{key} must be finite and nonnegative")
    speed = np.asarray(rally.get("outgoing_speed_range"), dtype=float)
    if (
      speed.shape != (2,) or not np.isfinite(speed).all() or not 0 < speed[0] < speed[1]
    ):
      raise ValueError("Invalid rally outgoing_speed_range")
  if config.get("position_sampling", "random") not in ("random", "fixed_per_cell"):
    raise ValueError("position_sampling must be random or fixed_per_cell")
  sweep = config.get("sweep")
  if sweep:
    if any(
      not isinstance(sweep[k], int) or sweep[k] < 1
      for k in ("directions", "speed_levels", "tts_levels")
    ):
      raise ValueError("Sweep dimensions must be positive integers")
    if (
      np.prod([sweep[k] for k in ("directions", "speed_levels", "tts_levels")])
      != config["samples_per_cell"]
    ):
      raise ValueError(
        "samples_per_cell must equal directions * speed_levels * tts_levels"
      )
  for key in (
    "y_range",
    "z_range",
    "normal_yaw_degrees",
    "normal_pitch_degrees",
    "speed_mps",
    "tts_seconds",
  ):
    v = np.asarray(config[key], dtype=float)
    if v.shape != (2,) or not np.isfinite(v).all() or v[0] >= v[1]:
      raise ValueError(f"Invalid {key}: expected finite increasing pair")
  if config["world_frame"] != "hope_table" or not np.isfinite(config["plane_x"]):
    raise ValueError("Expected fixed plane in hope_table world")
  counts = [*config["grid"], config["samples_per_cell"]]
  if len(config["grid"]) != 2 or any(int(v) != v or v < 1 for v in counts):
    raise ValueError("Grid dimensions and samples_per_cell must be positive integers")
  if (
    config["speed_mps"][0] <= 0
    or config["speed_mps"][1] > 5
    or config["tts_seconds"][0] < 0.15
    or config["episode_seconds"] <= config["tts_seconds"][1]
    or not np.isfinite(config["episode_seconds"])
    or not np.isfinite(config["display_hold_seconds"])
    or config["display_hold_seconds"] < 0
  ):
    raise ValueError("Require speed in (0,5] m/s, TTS >= .15s and longer episode")
  if any(
    abs(v) >= 85
    for k in ("normal_yaw_degrees", "normal_pitch_degrees")
    for v in config[k]
  ):
    raise ValueError("Normals must face +X within 85 degrees")
  if any(not np.isfinite(v) or v <= 0 for v in config["tolerances"].values()):
    raise ValueError("Metric tolerances must be finite and positive")


def sample_suite(config):
  validate(config)
  if "tts_values" in config:
    # Pair identical spatial/vector targets across TTS for fair comparison.
    base = deepcopy(config)
    times = base.pop("tts_values")
    targets = sample_suite(base)
    if "rally_sampling_audit" in base:
      config["rally_sampling_audit"] = base["rally_sampling_audit"]
    return [
      replace(t, trial_id=i * len(targets) + j, tts=float(tts), tts_id=i)
      for i, tts in enumerate(times)
      for j, t in enumerate(targets)
    ]
  if (
    config.get("position_sampling") == "fixed_per_cell"
    or config.get("fixed_point") is not None
  ):
    return sample_fixed_points(config)
  rng = np.random.default_rng(config["seed"])
  ys, zs = coverage.grid_edges(config)
  trials = []
  domain = config.get("coverage_domain")
  full_rectangle = config.get("spatial_coverage") == "full_rectangle"
  for z in range(len(zs) - 1):
    for y in range(len(ys) - 1):
      pieces = (
        coverage.cell_regions(domain, (ys[y], ys[y + 1], zs[z], zs[z + 1]))
        if domain
        else None
      )
      if domain and not pieces and not full_rectangle:
        continue  # Geometric out-of-domain cell; never counted as policy failure.
      for _ in range(config["samples_per_cell"]):
        if domain:
          if full_rectangle:
            position = np.array(
              [
                config["plane_x"],
                rng.uniform(ys[y], ys[y + 1]),
                rng.uniform(zs[z], zs[z + 1]),
              ]
            )
            members = [r for r in domain["regions"] if coverage.contains(r, position)]
            # Keep out-of-support targets, drawing vectors from the same frozen library domain.
            members = members or domain["regions"]
          else:
            position, members = coverage.sample_position(pieces, config["plane_x"], rng)
          normal, velocity = coverage.sample_vectors(members, rng)
          trials.append(
            Trial(
              len(trials),
              y,
              z,
              tuple(position),
              tuple(normal),
              tuple(velocity),
              rng.uniform(*config["tts_seconds"]),
            )
          )
          continue
        position = (
          float(config["plane_x"]),
          rng.uniform(ys[y], ys[y + 1]),
          rng.uniform(zs[z], zs[z + 1]),
        )
        yaw, pitch = np.radians(
          [
            rng.uniform(*config["normal_yaw_degrees"]),
            rng.uniform(*config["normal_pitch_degrees"]),
          ]
        )
        normal = np.array(
          [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)]
        )
        velocity = normal * rng.uniform(*config["speed_mps"])
        trials.append(
          Trial(
            len(trials),
            y,
            z,
            position,
            tuple(normal),
            tuple(velocity),
            rng.uniform(*config["tts_seconds"]),
          )
        )
  if config.get("shuffle", True):
    rng.shuffle(trials)
  return trials


def _levels(bounds, count):
  return np.linspace(*bounds, count) if count > 1 else np.array([np.mean(bounds)])


def _direction(members, region_index, config, rng):
  if members:
    normal, velocity = coverage.sample_vectors([members[region_index]], rng)
    bounds = members[region_index]["speed_range"]
    return normal, velocity / np.linalg.norm(velocity), bounds
  yaw, pitch = np.radians(
    [
      rng.uniform(*config["normal_yaw_degrees"]),
      rng.uniform(*config["normal_pitch_degrees"]),
    ]
  )
  normal = np.array(
    [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)]
  )
  return normal, normal, config["speed_mps"]


def sample_fixed_points(config):
  rng = np.random.default_rng(config["seed"])
  ys, zs = coverage.grid_edges(config)
  domain = config.get("coverage_domain")
  explicit = config.get("fixed_point")
  points = []
  if explicit is not None:
    point = np.asarray([config["plane_x"], *explicit], dtype=float)
    if (
      point.shape != (3,)
      or not np.isfinite(point).all()
      or not ys[0] <= point[1] <= ys[-1]
      or not zs[0] <= point[2] <= zs[-1]
    ):
      raise ValueError("Fixed point lies outside configured plane region")
    members = (
      [r for r in domain["regions"] if coverage.contains(r, point)] if domain else []
    )
    if domain and not members:
      raise ValueError(
        "Fixed point is outside motion/training support; no automatic replacement"
      )
    y = min(int(np.searchsorted(ys, point[1], side="right") - 1), len(ys) - 2)
    z = min(int(np.searchsorted(zs, point[2], side="right") - 1), len(zs) - 2)
    points.append((y, z, point, members))
  else:
    for z in range(len(zs) - 1):
      for y in range(len(ys) - 1):
        if domain:
          pieces = coverage.cell_regions(domain, (ys[y], ys[y + 1], zs[z], zs[z + 1]))
          if not pieces:
            continue
          point, members = coverage.sample_position(pieces, config["plane_x"], rng)
        else:
          point = np.array(
            [
              config["plane_x"],
              rng.uniform(ys[y], ys[y + 1]),
              rng.uniform(zs[z], zs[z + 1]),
            ]
          )
          members = []
        points.append((y, z, point, members))
  groups = []
  index = 0
  sweep = config.get("sweep")
  rally = config.get("rally_constraints", {}).get("enabled", False)
  audit = []
  for point_id, (y, z, point, members) in enumerate(points):
    group = []
    point_audit = dict(
      point_id=point_id, cell_y=y, cell_z=z, position=point.tolist(), groups=[]
    )
    if rally:
      from deploy.benchmarks.racket_command.rally_constraints import (
        incoming_bank,
        sample_group,
      )

      bank = incoming_bank(
        point,
        config["physical_ball"],
        rng,
        config["rally_constraints"]["incoming_bank_size"],
      )
      if not len(bank):
        point_audit["status"] = "no_legal_incoming_witness"
        audit.append(point_audit)
        continue
    count = sweep["directions"] if sweep else config["samples_per_cell"]
    order = rng.permutation(len(members)) if members else [0]
    for direction_id in range(count):
      if members and direction_id and direction_id % len(members) == 0:
        order = rng.permutation(len(members))
      if rally:
        picked = sample_group(
          point, members, bank, rng, config, _direction, direction_id
        )
        if picked is None:
          point_audit["status"] = "direction_search_exhausted"
          break
        normal, direction, speeds, witness = picked
        point_audit["groups"].append(witness)
      else:
        normal, direction, bounds = _direction(
          members, order[direction_id % len(order)], config, rng
        )
        speeds = (
          _levels(bounds, sweep["speed_levels"]) if sweep else [rng.uniform(*bounds)]
        )
      times = (
        _levels(config["tts_seconds"], sweep["tts_levels"])
        if sweep
        else [rng.uniform(*config["tts_seconds"])]
      )
      for speed_id, speed in enumerate(speeds):
        for tts_id, tts in enumerate(times):
          group.append(
            Trial(
              index,
              y,
              z,
              tuple(point),
              tuple(normal),
              tuple(direction * speed),
              float(tts),
              point_id,
              direction_id,
              speed_id if sweep else None,
              tts_id if sweep else None,
            )
          )
          index += 1
    if rally:
      point_audit.setdefault("status", "complete")
      point_audit["scheduled"] = len(group)
      audit.append(point_audit)
    if config.get("shuffle", True):
      rng.shuffle(group)
    groups.append(group)
  if config.get("shuffle", True):
    rng.shuffle(groups)
  if rally:
    config["rally_sampling_audit"] = audit
    if not any(groups):
      raise ValueError(
        "No normal-rally commands found at the sampled points; inspect physical/training support constraints"
      )
  return [trial for group in groups for trial in group]


def suite_document(config, trials):
  # Keep hashes of previously saved suites without point IDs compatible.
  items = [{k: v for k, v in asdict(t).items() if v is not None} for t in trials]
  digest = hashlib.sha256(
    json.dumps(dict(config=config, trials=items), sort_keys=True).encode()
  ).hexdigest()
  return dict(
    schema_version=1,
    config=config,
    trials=items,
    sha256=digest,
    generation=(
      "Full rectangular grid, including training-support holes; paired random targets across TTS; no selector/outcome filtering"
      if config.get("spatial_coverage") == "full_rectangle"
      else "Grid commands conditioned only on frozen training support and recorded incoming/return physics; no selector or policy feedback"
      if config.get("rally_constraints", {}).get("enabled")
      else "Independent grid over frozen training-support union; no selection or outcome conditioning"
      if config.get("coverage_domain")
      else "Independent fixed grid; no motion or policy inputs"
    ),
    velocity_convention="Nominal reference velocity direction with independent normal perturbation"
    if config.get("coverage_domain")
    else "velocity = speed * normal; normal faces +X",
  )


def load_suite(document):
  validate(document["config"])
  trials = [Trial(**v) for v in document["trials"]]
  if suite_document(document["config"], trials)["sha256"] != document["sha256"]:
    raise ValueError("Suite checksum mismatch")
  if not trials or len({t.trial_id for t in trials}) != len(trials):
    raise ValueError("Empty suite or duplicate trial IDs")
  for t in trials:
    vectors = np.asarray([t.position, t.normal, t.velocity])
    if (
      vectors.shape != (3, 3)
      or not np.isfinite(vectors).all()
      or not np.isclose(np.linalg.norm(t.normal), 1)
      or not np.isfinite(t.tts)
      or t.tts < 0.15
      or not np.isclose(t.position[0], document["config"]["plane_x"])
      or not 0 <= t.cell_y < document["config"]["grid"][0]
      or not 0 <= t.cell_z < document["config"]["grid"][1]
    ):
      raise ValueError("Invalid trial in suite")
    domain = document["config"].get("coverage_domain")
    if (
      domain
      and document["config"].get("spatial_coverage") != "full_rectangle"
      and not any(
        coverage.contains(r, t.position)
        and coverage.supports_vectors(r, np.asarray(t.normal), np.asarray(t.velocity))
        for r in domain["regions"]
      )
    ):
      raise ValueError("Trial lies outside frozen joint training support")
  if document["config"].get("rally_constraints", {}).get("enabled"):
    from deploy.benchmarks.racket_command.rally_constraints import (
      contact_center,
      incoming_history,
      return_outcomes,
    )

    config = document["config"]
    witnesses = {
      (p["point_id"], g["direction_id"]): (p, g)
      for p in config.get("rally_sampling_audit", [])
      for g in p["groups"]
    }
    checked = set()
    for t in trials:
      key = (t.point_id, t.direction_id)
      if key not in witnesses:
        raise ValueError("Rally-constrained trial has no incoming/return witness")
      p, g = witnesses[key]
      speed = np.linalg.norm(t.velocity)
      if (
        not np.allclose(t.position, p["position"], atol=1e-10)
        or not np.allclose(t.normal, g["normal"], atol=1e-10)
        or not np.allclose(
          np.asarray(t.velocity) / speed, g["velocity_direction"], atol=1e-10
        )
        or not any(np.isclose(speed, s) for s in g["speed_levels"])
      ):
        raise ValueError("Rally-constrained command differs from its validity witness")
      if key not in checked:
        physics = config["physical_ball"]
        incoming = g["incoming"]["velocity"]
        normal = np.asarray(g["normal"])
        velocities = np.asarray(g["speed_levels"])[:, None] * g["velocity_direction"]
        ok, _, _ = return_outcomes(
          p["position"],
          normal,
          velocities,
          incoming,
          physics,
          config["rally_constraints"],
          dt=0.001,
        )
        if (
          not ok.all()
          or incoming_history(
            contact_center(p["position"], normal, physics), incoming, physics
          )
          is None
        ):
          raise ValueError("Rally-constrained command has an invalid physical return")
        checked.add(key)
  return trials
