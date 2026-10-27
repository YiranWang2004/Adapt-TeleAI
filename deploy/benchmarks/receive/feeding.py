"""Frozen offline evaluation suites using the shared launch sampling kernel."""

import hashlib
import json

from deploy.g1_pingpong.launch_sampling import sample_launches


def generate_suite(config, physics):
  serves = sample_launches(config, physics)
  payload = dict(config=config, physics=physics, serves=serves)
  return dict(
    schema_version=1,
    task="planner_receive",
    generation="Uniform launch p/v; rejection only by net, receiving bounce and fixed central crossing region; no policy/planner/motion feedback",
    sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
    **payload,
  )


def load_suite(document):
  payload = {k: document[k] for k in ["config", "physics", "serves"]}
  digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
  if digest != document["sha256"] or document.get("task") != "planner_receive":
    raise ValueError("Invalid stage-two ball suite or checksum")
  return document["serves"]
