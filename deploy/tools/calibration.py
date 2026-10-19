"""Fit a right-handed mocap-world -> training-world transform from points."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def quaternion_from_matrix(matrix):
  r = np.asarray(matrix)
  k = np.array(
    [
      [
        r[0, 0] - r[1, 1] - r[2, 2],
        r[0, 1] + r[1, 0],
        r[0, 2] + r[2, 0],
        r[2, 1] - r[1, 2],
      ],
      [
        r[0, 1] + r[1, 0],
        r[1, 1] - r[0, 0] - r[2, 2],
        r[1, 2] + r[2, 1],
        r[0, 2] - r[2, 0],
      ],
      [
        r[0, 2] + r[2, 0],
        r[1, 2] + r[2, 1],
        r[2, 2] - r[0, 0] - r[1, 1],
        r[1, 0] - r[0, 1],
      ],
      [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
    ]
  )
  _, vectors = np.linalg.eigh(k)
  xyzw = vectors[:, -1]
  q = xyzw[[3, 0, 1, 2]]
  return q if q[0] >= 0 else -q


def fit_transform(source, target, *, scale=1.0, fit_scale=False):
  source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
  if (
    source.shape != target.shape
    or source.ndim != 2
    or source.shape[1] != 3
    or len(source) < 3
  ):
    raise ValueError("Provide at least three corresponding 3D points")
  if (
    not np.isfinite(source).all()
    or not np.isfinite(target).all()
    or not np.isfinite(scale)
    or scale <= 0
  ):
    raise ValueError("Nonfinite calibration data or invalid scale")
  x, y = source - source.mean(0), target - target.mean(0)
  if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
    raise ValueError("Calibration points are collinear")
  u, singular, vt = np.linalg.svd(x.T @ y)
  correction = np.eye(3)
  correction[-1, -1] = np.linalg.det(vt.T @ u.T)
  r = vt.T @ correction @ u.T
  if fit_scale:
    scale = float(np.sum(singular * np.diag(correction)) / np.sum(x * x))
  if scale <= 0:
    raise ValueError("Calibration cannot produce a positive scale")
  translation = target.mean(0) - scale * r @ source.mean(0)
  residual = np.linalg.norm(source @ r.T * scale + translation - target, axis=1)
  return {
    "transform": {
      "scale": scale,
      "quaternion_wxyz": quaternion_from_matrix(r).tolist(),
      "translation": translation.tolist(),
    },
    "points": len(source),
    "rms_error_m": float(np.sqrt(np.mean(residual**2))),
    "max_error_m": float(np.max(residual)),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--points",
    type=Path,
    required=True,
    help="CSV: mocap_x,mocap_y,mocap_z,world_x,world_y,world_z",
  )
  parser.add_argument("--output", type=Path)
  parser.add_argument("--scale", type=float, default=1.0)
  parser.add_argument("--fit-scale", action="store_true")
  args = parser.parse_args()
  with args.points.open() as stream:
    rows = list(csv.DictReader(stream))
  source = [[float(row["mocap_" + axis]) for axis in "xyz"] for row in rows]
  target = [[float(row["world_" + axis]) for axis in "xyz"] for row in rows]
  report = fit_transform(source, target, scale=args.scale, fit_scale=args.fit_scale)
  result = json.dumps(report, indent=2)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
      stream.write(result + "\n")
  print(result)


if __name__ == "__main__":
  main()
