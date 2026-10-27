"""Audit and summarize repeated continuous play trials for the two deploy runs."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def audit_trial(report, trace):
  """Verify recorded acceptance against physics metrics, independently of flags."""
  limits = np.asarray(report["stability_limits"])
  for i, trial in enumerate(report["trials"]):
    if not trial["passed"]:
      assert trial["events"][-1]["phase"] == "FAIL"
      continue
    events = trial["events"]
    assert [e["phase"] for e in events] == [
      "PREPARE",
      "SWING",
      "RETURN",
      "HOLD",
      "PASS",
    ]
    _, swing, recovery, hold, end = [e["time"] for e in events]
    assert swing >= 3 - 1e-5 and hold - recovery >= 3 - 1e-5
    assert end - hold >= 5 - 1e-5
    for start, stop in ((swing - 0.98, swing), (hold - 0.98, hold), (hold, end)):
      mask = (trace["time"] > start + 1e-6) & (trace["time"] <= stop + 1e-6)
      values = trace["metrics"][mask, i]
      assert len(values) > 0 and np.isfinite(values).all()
      assert np.all(values <= limits + 1e-6), (report["seed"], trial["motion_index"])
    mask = (trace["time"] > hold + 1e-6) & (trace["time"] <= end + 1e-6)
    assert np.allclose(trace["reference_time"][mask, i], 0)


def summarize(root, seeds):
  summary = {
    "seeds": seeds,
    "home": 19,
    "protocol": "Independent trials with randomized training ball feeds; continuous physics within each standing-swing-return-5s-hold trial",
    "total_trials": len(seeds) * 43 * 2,
    "models": {},
  }
  csv_rows = []
  for name in ("tele", "hard"):
    reports = []
    for seed in seeds:
      folder = root / name / f"seed_{seed}"
      report = json.loads((folder / "summary.json").read_text())
      assert report["seed"] == seed and report["home"] == 19
      assert report["total"] == 43
      assert report["ball_difficulty_scale"] == 1
      assert [r["motion_index"] for r in report["trials"]] == list(range(43))
      with np.load(folder / "trace.npz") as trace:
        audit_trial(report, trace)
      reports.append(report)
    for key in (
      "checkpoint_sha256",
      "tracker_sha256",
      "training_iteration",
      "stability_limits",
    ):
      assert all(r[key] == reports[0][key] for r in reports)
    rows = []
    for index in range(43):
      trials = [r["trials"][index] for r in reports]
      failed = [t for t in trials if not t["passed"]]
      failures = Counter(t["events"][-2]["phase"] for t in failed)
      ball_samples = {
        tuple(t["incoming_ball"]["velocity"]) + (t["incoming_ball"]["intercept_time"],)
        for t in trials
        if "incoming_ball" in t
      }
      row = {
        "model": name,
        "motion_index": index,
        "motion": Path(trials[0]["motion"]).name,
        "passed": sum(t["passed"] for t in trials),
        "attempts": len(trials),
        "hit": sum(t["hit"] for t in trials),
        "return_success": sum(t["return_success"] for t in trials),
        "passed_and_return_success": sum(
          t["passed"] and t["return_success"] for t in trials
        ),
        "prepare_failures": failures["PREPARE"],
        "swing_failures": failures["SWING"],
        "return_failures": failures["RETURN"],
        "hold_failures": failures["HOLD"],
        "unique_incoming_samples": len(ball_samples),
        "failed_seeds": [
          seed for seed, t in zip(seeds, trials, strict=True) if not t["passed"]
        ],
      }
      assert row["passed"] + sum(failures.values()) == len(trials)
      rows.append(row)
      csv_rows.append(row)
    summary["models"][name] = {
      "checkpoint": reports[0]["checkpoint"],
      "checkpoint_sha256": reports[0]["checkpoint_sha256"],
      "tracker_sha256": reports[0]["tracker_sha256"],
      "training_iteration": reports[0]["training_iteration"],
      "passed": sum(r["passed"] for r in rows),
      "attempts": len(seeds) * 43,
      "hit": sum(r["hit"] for r in rows),
      "return_success": sum(r["return_success"] for r in rows),
      "passed_and_return_success": sum(r["passed_and_return_success"] for r in rows),
      "failure_phases": {
        phase: sum(r[phase + "_failures"] for r in rows)
        for phase in ("prepare", "swing", "return", "hold")
      },
      "all_passed_motions": [
        r["motion_index"] for r in rows if r["passed"] == len(seeds)
      ],
      "zero_passed_motions": [r["motion_index"] for r in rows if r["passed"] == 0],
      "per_motion": rows,
    }
  summary["common_all_passed"] = sorted(
    set(summary["models"]["tele"]["all_passed_motions"])
    & set(summary["models"]["hard"]["all_passed_motions"])
  )
  (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  with (root / "per_motion.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
  return summary


def plot_summary(root, summary):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  counts = np.array(
    [
      [r["passed"] for r in summary["models"][name]["per_motion"]]
      for name in ("tele", "hard")
    ]
  )
  figure, ax = plt.subplots(figsize=(18, 3.2), layout="constrained")
  ax.imshow(counts, vmin=0, vmax=len(summary["seeds"]), cmap="RdYlGn", aspect="auto")
  for model, row in enumerate(counts):
    for motion, value in enumerate(row):
      ax.text(motion, model, str(value), ha="center", va="center", fontsize=9)
  ax.set_xticks(range(43), labels=range(43))
  ax.set_yticks([0, 1], labels=["Tele-GMR", "Hard"])
  ax.set_xlabel("Motion index (0-based)")
  ax.set_title(
    f"Complete standing-swing-recovery passes / {len(summary['seeds'])} trials per motion\nHome 19; seeds {summary['seeds'][0]}-{summary['seeds'][-1]}; final stable hold 5 s"
  )
  figure.savefig(root / "pass_counts.png", dpi=180)
  plt.close(figure)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("root", type=Path)
  parser.add_argument("--first-seed", type=int, default=42)
  parser.add_argument("--repeats", type=int, default=10)
  args = parser.parse_args()
  result = summarize(
    args.root, list(range(args.first_seed, args.first_seed + args.repeats))
  )
  plot_summary(args.root, result)
  print(
    json.dumps(
      {
        name: {k: v for k, v in value.items() if k != "per_motion"}
        for name, value in result["models"].items()
      },
      indent=2,
    )
  )
  print("Common all-passed motions:", result["common_all_passed"])
