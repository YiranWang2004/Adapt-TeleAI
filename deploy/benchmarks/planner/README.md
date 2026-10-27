# Real-like planner diagnostic

## Latest A30808 planner replay on recorded real trajectories

```zsh
.venv/bin/python -m deploy.benchmarks.planner.visualize_latest_planner_on_real_log \
  --output deploy/recordings/planner/a30808_real_trajectory_replay
```

Replays the original ball-position packets from the two October 22 real logs,
in timestamp order, through the current A30808 planner core. It does not pass
future ball positions to the planner. The event PNGs compare the predicted
flight and strike point at each old `motion_selected` timestamp with the
previous deploy command and the later measured crossing of X=-0.25 m. The GIFs
show the causal replay frame by frame. A measured crossing is counted only when
its Y/Z lie within the recorded table-height workspace; a ball already below
the table is not reported as a valid crossing.

Outputs are saved under `deploy/recordings/planner/a30808_real_trajectory_replay/`.
This checks planner predictions on those recorded samples; it does not replay
the actor, motion selector decisions, or robot movement.

All interactive MuJoCo windows support `[` / `-` to halve playback speed,
`]` / `+` to double it, and backslash `\` to restore1x (range1/16x..8x).
Click the window before pressing keys. Only wall-clock pacing changes; physical
steps, observation timestamps, policy steps and TTS keep their simulation values.
Existing processes must be restarted to load these controls.

## Full humanoid reproduction using measured ball positions

```zsh
.venv/bin/python -m deploy.benchmarks.planner.recorded_receive \
  --render --manual \
  --sensor-clock recorded \
  --duration 600 \
  --output "deploy/recordings/planner/recorded_receive_$(date +%Y%m%d_%H%M%S)"
```

Use the normal MuJoCo sequence: Enter → POSE_HOLD → L to lower → A → STABLE →
W to enter receiving. Three recorded incoming balls are served5seconds apart;
there is no robot reset between balls. `--repetitions N` repeats the three-ball
sequence N times. Q exits. Omit `--manual` for automatic startup and receiving.

This benchmark runs the unchanged deployment control loop, A3 planner, motion
selector, and actor. It does not replay recorded robot actions. The input file
`recorded_ball_cases_20261022.json` contains405 original position observations
extracted from the two real runs, with original header and receiver timestamps
and source-file hashes. The complete trajectory is private to the environment;
the planner receives only causal position packets through the normal estimator.

The ball follows the measured incoming positions. Camera acquisition stamps
were not recorded, so the physical playback clock is reconstructed on a180Hz
frame grid using100Hz callback groups and inferred missing-frame gaps. Original
callback/arrival times are retained separately. This reconstruction is explicitly
not a measurement of the true camera acquisition clock.

At a simulated racket contact, recorded positions immediately stop controlling
the ball. The standard physical collision response and free-flight dynamics then
take over. If no contact happens before the incoming cutoff, the ball continues
freely from the last incoming state. Real trajectory deflections near the robot
are excluded from playback, so they cannot produce phantom successful hits.

`--sensor-clock capture` is a separate diagnostic that replaces header stamps
with the reconstructed acquisition clock; positions and delivery batches remain
the same. It is not a deployed timestamp fix or calibrated ground truth.

To save a humanoid video without opening a window:

```zsh
MUJOCO_GL=egl .venv/bin/python -m deploy.benchmarks.planner.recorded_receive \
  --sensor-clock recorded --video --duration 45 \
  --output "deploy/recordings/planner/recorded_video_$(date +%Y%m%d_%H%M%S)"
```

All lifecycle, command, joint-feedback and simulated mocap logs are saved, plus
`diagnosis.json` and optionally `humanoid.mp4`. `summary.json:passed` assesses the
deployment flow; it must not be interpreted as successful contact. Check
`simulated_hits` and `simulated_legal_returns` separately.

## Planner-only controlled comparison

Run from the repository root:

```zsh
.venv/bin/python -m deploy.benchmarks.planner.real_like \
  --render \
  --output "deploy/recordings/planner/real_like_$(date +%Y%m%d_%H%M%S)"
```

This is an isolated benchmark, separate from the continuous G1 deployment.
It loads the existing A3 planner and ball estimator, with no actor or motion
selector. The three initial ball positions/velocities are fitted to the incoming
pre-trigger observations in the October 22 real experiment diagnosis. They are
never reconstructed from a motion or racket command. The generated `suite.json`
records the exact initial conditions; `--suite PATH` reproduces those conditions.

The environment uses the same `SimBall` physical flight/table/net kernels as the
production MuJoCo backend at 1 ms, in a MuJoCo scene. This is analytical ball
contact integration, matching the existing deployment, rather than native MuJoCo
sphere/table contact. The ball evolves freely; no paddle or robot changes its
trajectory. The visual display shows the physical ball in orange, the first
time-window command's predicted intersection in red, and the true intersection
in green. Green is diagnostic ground truth and is never passed to the planner.
The display repeats the same independently generated test trajectories used for
the numerical evaluation. It does not exercise policy or physical robot control.

Three input conditions are compared:

- `oracle`: exact plant position/velocity, to isolate prediction-model error.
- `capture_stamp`: 180 Hz observations, delivered by 100 Hz polling, retaining
  their actual capture timestamps.
- `callback_stamp`: identical observations and deliveries, timestamped at their
  batched callbacks, with 30 microseconds between callbacks in one batch.

Default sampling noise is zero; `--noise-mm N` adds the same seeded measurement
noise to both timestamp conditions. Six polling phases (`--phases N`) vary the
relative alignment of capture, polling, and the 50 Hz planning loop.

Two physical conditions are kept separate:

- `matched`: the production A3 physics in both plant and planner.
- `bounce_sensitivity`: plant vertical restitution 0.78 and tangential retention
  0.74, while the unchanged planner uses 1.01 and 0.69. These are explicit
  sensitivity values, **not** newly calibrated field parameters.

Metrics compare the prediction with the physical ball's future X=-0.25 crossing.
The main summary uses the first command inside the production TTS/start-frame
window (TTS <= 0.5 s and start frame <= 29). It does not apply motion-distance,
readiness, or selector gates and therefore does not claim to reproduce the exact
real trigger. Full per-tick predictions and errors remain in `results.json`.
The 5 cm / 20 ms check is a diagnostic accuracy threshold, not a contact rate.

Outputs include `planner_errors.png`, `trajectory_comparison.png`, full numerical
results, initial conditions, and the 1 ms physical trajectories. The later portion
of the real recordings includes unexplained deflections near the robot; those
deflections are deliberately absent from this free-flight planner experiment.

The 1026 input, `recorded_ball_cases_20261026.json`, contains two incoming
trajectories: 154 samples from `real_20261026_201506` and 190 samples from
`real_20261026_202459` (344 total). Use `--recording` to select this file.
The second recording's `world` frame label is normalized to `hope_table` without
changing XYZ: both runs have identity mocap transforms, consistent home anchors,
and a table-height bounce. This is a consistency check, not a fresh calibration;
the evidence and original config are retained in the case metadata.
Reproduce the second extraction with
`.venv/bin/python -m deploy.benchmarks.planner.extract_1026_second` after placing
its original `mocap_packets.jsonl` and `summary.json` in
`deploy/recordings/real_20261026_202459/`. The incoming segment begins at the last
launch X turning point before selection and ends at its first X=-0.25m crossing.
