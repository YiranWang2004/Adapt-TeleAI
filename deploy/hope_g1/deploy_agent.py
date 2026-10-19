"""Independent HOPEPingPongMultiMotionG1 task entry (check / MuJoCo / real)."""

import argparse
import json
import signal
from datetime import datetime
from pathlib import Path

import torch
import yaml

from deploy.hope_g1.configuration import TASK, load_deployment
from deploy.hope_g1.controller import HopeController

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def request_shutdown(*_):
  raise KeyboardInterrupt("HOPE deployment shutdown requested")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--env", choices=["check", "mujoco", "real"], default="check")
  parser.add_argument(
    "--config",
    type=Path,
    help="Default: local task.yaml for check/MuJoCo; task_real.yaml for real",
  )
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--env-config", type=Path)
  parser.add_argument("--agent-config", type=Path)
  parser.add_argument("--motion-dir", type=Path)
  parser.add_argument("--real-config", type=Path, default=CONFIG_DIR / "real.yaml")
  parser.add_argument("--mocap-config", type=Path, default=CONFIG_DIR / "mocap.yaml")
  parser.add_argument("--rally-config", type=Path, default=CONFIG_DIR / "rally.yaml")
  parser.add_argument("--motion-index", type=int, default=0)
  parser.add_argument("--motion-select", choices=["fixed", "auto"], default="fixed")
  parser.add_argument("--motion-allow", default="all")
  parser.add_argument("--duration", type=float, default=120)
  parser.add_argument("--episodes", type=int, default=1)
  parser.add_argument("--network-interface")
  parser.add_argument("--enable-motors", action="store_true")
  parser.add_argument("--render", action="store_true")
  parser.add_argument("--sim-controls", choices=["auto", "manual"], default="auto")
  parser.add_argument(
    "--sim-startup", choices=["ground", "suspended"], default="suspended"
  )
  parser.add_argument("--standing-only", action="store_true")
  parser.add_argument("--ball-source", choices=["sim", "mocap"])
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--min-hit-rate",
    type=float,
    help="Require this physical contact fraction in automatic MuJoCo receiving",
  )
  parser.add_argument("--output", type=Path)
  parser.add_argument("--torch-threads", type=int, default=1)
  args = parser.parse_args()
  if args.duration <= 0 or args.episodes < 1 or args.torch_threads < 1:
    parser.error("duration, episodes and torch-threads must be positive")
  if args.env != "real" and args.enable_motors:
    parser.error("--enable-motors is only valid for --env real")
  if args.env == "real" and args.ball_source == "sim":
    parser.error("Real HOPE deployment requires measured mocap")
  if args.env == "real" and (args.render or args.standing_only):
    parser.error("render/standing-only are simulation options")
  if args.min_hit_rate is not None and (
    args.env != "mujoco"
    or args.sim_controls != "auto"
    or args.standing_only
    or args.ball_source == "mocap"
    or not 0 <= args.min_hit_rate <= 1
  ):
    parser.error(
      "min-hit-rate must be in [0, 1] and requires automatic MuJoCo receiving"
    )
  signal.signal(signal.SIGTERM, request_shutdown)
  torch.set_num_threads(args.torch_threads)
  config_path = args.config or CONFIG_DIR / (
    "task_real.yaml" if args.env == "real" else "task.yaml"
  )
  config = yaml.safe_load(config_path.read_text())
  if config.pop("task", None) != TASK:
    parser.error("This entry only accepts HOPEPingPongMultiMotionG1 task configs")
  for name in ("checkpoint", "motion_dir", "env_config", "agent_config"):
    if getattr(args, name) is not None:
      config[name] = getattr(args, name)
  options = yaml.safe_load(args.rally_config.read_text())
  deployment = load_deployment(**config, ball_config=options["ball"])
  n = len(deployment.motions)
  try:
    allow = (
      list(range(n))
      if args.motion_allow == "all"
      else [int(i) for i in args.motion_allow.split(",")]
    )
    if (
      not 0 <= args.motion_index < n
      or not allow
      or len(set(allow)) != len(allow)
      or any(not 0 <= i < n for i in allow)
    ):
      raise ValueError
  except ValueError:
    parser.error(f"Motion IDs must be unique and within 0..{n - 1}")
  # Construction validates the controller contract before any transport opens.
  runtime = yaml.safe_load(args.real_config.read_text())
  HopeController(
    deployment,
    runtime,
    options,
    home_index=args.motion_index,
    allow=allow,
    selection=args.motion_select,
  )
  print(json.dumps(deployment.report(), indent=2))
  if args.env == "check":
    return
  output = (
    args.output
    or Path("deploy/recordings/hope_g1") / f"{args.env}_{datetime.now():%Y%m%d_%H%M%S}"
  )
  if args.env == "mujoco":
    from deploy.g1_pingpong.simulation_runtime import run_deployment

    result = run_deployment(
      deployment,
      args.real_config,
      args.mocap_config,
      controller_factory=HopeController,
      motion_index=args.motion_index,
      motion_select=args.motion_select,
      motion_allow=allow,
      simulation_config=args.rally_config,
      duration=args.duration,
      episodes=args.episodes,
      seed=args.seed,
      render=args.render,
      controls=args.sim_controls,
      startup=args.sim_startup,
      standing_only=args.standing_only,
      ball_source=args.ball_source or "sim",
      output=output,
      minimum_hit_rate=args.min_hit_rate,
    )
    print(
      json.dumps(
        {
          k: result.get(k)
          for k in (
            "passed",
            "flow_passed",
            "simulated_hits",
            "simulated_balls",
            "hit_rate",
            "error",
            "completed_swings",
            "recovered_swings",
            "physics_validation",
          )
        },
        indent=2,
      )
    )
    if not result["passed"]:
      raise SystemExit(1)
  else:
    from deploy.backends.sdk import network_interfaces, require_sdk
    from deploy.backends.unitree import run_control_loop
    from deploy.tools.preflight import check_configuration

    require_sdk()
    runtime, _ = check_configuration(
      args.real_config, args.mocap_config, args.network_interface
    )
    if runtime["network_interface"] not in network_interfaces():
      parser.error("Configured network interface does not exist")
    run_control_loop(
      deployment,
      runtime,
      args.mocap_config,
      controller_factory=HopeController,
      enable_motors=args.enable_motors,
      motion_index=args.motion_index,
      motion_select=args.motion_select,
      motion_allow=allow,
      duration=args.duration,
      output=output,
      simulation_config=args.rally_config,
    )


if __name__ == "__main__":
  main()
