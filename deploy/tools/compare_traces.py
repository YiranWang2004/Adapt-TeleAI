"""Compare observation/action arrays captured from identical replayed inputs."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference, candidate, fields, atol=1e-5, rtol=1e-5):
  report = {}
  with (
    np.load(reference, allow_pickle=False) as left,
    np.load(candidate, allow_pickle=False) as right,
  ):
    for key in fields:
      if key not in left or key not in right:
        raise ValueError(f"Missing trace field: {key}")
      a, b = left[key], right[key]
      if a.shape != b.shape or not a.size:
        raise ValueError(
          f"Shape differs or field is empty: {key}: {a.shape}, {b.shape}"
        )
      passed = bool(
        np.isfinite(a).all()
        and np.isfinite(b).all()
        and np.allclose(a, b, atol=atol, rtol=rtol)
      )
      report[key] = {"passed": passed, "max_abs_error": float(np.max(np.abs(a - b)))}
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("reference", type=Path)
  parser.add_argument("candidate", type=Path)
  parser.add_argument(
    "--fields",
    nargs="+",
    default=["tracker_obs", "base_action", "target_q", "reference_time"],
  )
  parser.add_argument("--atol", type=float, default=1e-5)
  parser.add_argument("--rtol", type=float, default=1e-5)
  args = parser.parse_args()
  report = compare(args.reference, args.candidate, args.fields, args.atol, args.rtol)
  print(json.dumps(report, indent=2))
  if not all(row["passed"] for row in report.values()):
    raise SystemExit(1)


if __name__ == "__main__":
  main()
