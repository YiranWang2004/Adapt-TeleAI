"""Generic, serialized command-support geometry; no motion/policy imports."""

import numpy as np


def grid_edges(config):
  """The test partition, shared by sampling and visualization."""
  return (
    np.linspace(*config["y_range"], config["grid"][0] + 1),
    np.linspace(*config["z_range"], config["grid"][1] + 1),
  )


def area(polygon):
  p = np.asarray(polygon)
  if len(p) < 3:
    return 0.0
  return float(
    abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))) / 2
  )


def clip(polygon, ymin, ymax, zmin, zmax):
  points = list(np.asarray(polygon, dtype=float))
  for axis, bound, sign in [(0, ymin, 1), (0, ymax, -1), (1, zmin, 1), (1, zmax, -1)]:
    previous = points
    points = []
    if not previous:
      break
    for a, b in zip(previous, previous[1:] + previous[:1], strict=True):
      ina, inb = sign * (a[axis] - bound) >= -1e-12, sign * (b[axis] - bound) >= -1e-12
      if ina:
        points.append(a)
      if ina != inb:
        points.append(a + (b - a) * (bound - a[axis]) / (b[axis] - a[axis]))
  return np.asarray(points)


def cell_regions(domain, bounds):
  result = []
  for region in domain["regions"]:
    polygon = clip(region["polygon_yz"], *bounds)
    a = area(polygon)
    if a > 1e-12:
      result.append((region, polygon, a))
  return result


def contains(region, point):
  offset = np.asarray(region["rotation"]).T @ (
    np.asarray(point) - np.asarray(region["position_center"])
  )
  return bool(np.all(abs(offset) <= np.asarray(region["position_half_extent"]) + 1e-9))


def sample_position(pieces, x, rng):
  """Uniform over the union, correcting overlaps between support regions."""
  weights = np.asarray([item[2] for item in pieces])
  for _ in range(10000):
    _, polygon, _ = pieces[rng.choice(len(pieces), p=weights / weights.sum())]
    triangles = [
      (polygon[0], polygon[i], polygon[i + 1]) for i in range(1, len(polygon) - 1)
    ]
    sizes = np.array([area(t) for t in triangles])
    a, b, c = triangles[rng.choice(len(triangles), p=sizes / sizes.sum())]
    u, v = np.sqrt(rng.uniform()), rng.uniform()
    yz = (1 - u) * a + u * (1 - v) * b + u * v * c
    point = np.r_[x, yz]
    members = [region for region, _, _ in pieces if contains(region, point)]
    if members and rng.uniform() < 1 / len(members):
      return point, members
  raise ValueError("Could not sample support union; inspect domain geometry")


def sample_vectors(regions, rng):
  # Position has already been sampled uniformly over the global grid cell.
  # Sample a valid joint (normal, velocity) distribution at that position.
  region = regions[int(rng.integers(len(regions)))]
  rotation = np.asarray(region["rotation"])
  normal = rotation.T @ np.asarray(region["normal_center"])
  normal += rng.uniform(-1, 1, 3) * region["normal_half_extent"]
  normal = rotation @ (normal / np.linalg.norm(normal))
  speed = rng.uniform(*region["speed_range"])
  velocity = np.asarray(region["velocity_direction"]) * speed
  return normal, velocity


def supports_vectors(region, normal, velocity):
  direction = np.asarray(region["velocity_direction"])
  speed = float(np.asarray(velocity) @ direction)
  if (
    not region["speed_range"][0] - 1e-8 <= speed <= region["speed_range"][1] + 1e-8
    or np.linalg.norm(velocity - speed * direction) > 1e-7
  ):
    return False
  rotation = np.asarray(region["rotation"])
  ray = rotation.T @ normal
  center = rotation.T @ np.asarray(region["normal_center"])
  half = region["normal_half_extent"]
  low, high = 1e-10, np.inf
  for x, c in zip(ray, center, strict=True):
    if abs(x) < 1e-12:
      if abs(c) > half + 1e-8:
        return False
    else:
      a, b = (c - half) / x, (c + half) / x
      low, high = max(low, min(a, b)), min(high, max(a, b))
  return low <= high + 1e-8


def resolve(config):
  """Embed a frozen domain snapshot; standalone suite replay needs no assets."""
  import json
  from copy import deepcopy
  from pathlib import Path

  config = deepcopy(config)
  if config.get("domain_file"):
    config["coverage_domain"] = json.loads(Path(config.pop("domain_file")).read_text())
  domain = config.get("coverage_domain")
  if domain:
    if domain["world_frame"] != config["world_frame"]:
      raise ValueError("Domain world differs from command world")
    config.update(
      plane_x=domain["plane_x"],
      y_range=domain["y_range"],
      z_range=domain["z_range"],
      speed_mps=domain["speed_range"],
      tts_seconds=domain["tts_range"],
    )
  crop = config.pop("crop_bottom_z_rows", 0)
  if (
    not isinstance(crop, int)
    or isinstance(crop, bool)
    or not 0 <= crop < config["grid"][1]
  ):
    raise ValueError("crop_bottom_z_rows must leave at least one Z row")
  if crop:
    _, zs = grid_edges(config)
    config["region_crop"] = dict(
      bottom_z_rows=crop,
      original_grid=list(config["grid"]),
      original_z_range=list(config["z_range"]),
    )
    config["z_range"] = [float(zs[crop]), float(zs[-1])]
    config["grid"] = [config["grid"][0], config["grid"][1] - crop]
  if domain:
    ys, zs = grid_edges(config)
    config["active_cells"] = [
      [y, z]
      for z in range(len(zs) - 1)
      for y in range(len(ys) - 1)
      if cell_regions(domain, (ys[y], ys[y + 1], zs[z], zs[z + 1]))
    ]
  return config
