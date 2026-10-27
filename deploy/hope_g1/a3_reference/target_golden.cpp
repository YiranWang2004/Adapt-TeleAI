// Copyright (c) 2026, AgiBot Inc. All rights reserved.
// Extracted from HOPE-TeleAI A3 helper; Apache-2.0, see LICENSE.
#include <algorithm>
#include <iostream>
#include <iomanip>
#include "hope_math.hpp"
namespace a3_deploy {
struct HopeRacketSpec {
  hope_math::Vec3 pos_b{{0.638, -0.148, -0.061}};
  hope_math::Vec3 vel_b{{-1.071, 2.766, 0.451}};
  hope_math::Vec3 normal_b{{-0.357, 0.922, 0.150}};
  double time_to_strike{0.1};
  // ---- world-latched mode (training semantics) -----------------------------
  // In training the racket target (pos/vel/normal) and the coupled base target
  // are FIXED WORLD quantities per swing: as the robot moves/leans, the
  // *_target_pos_b obs changes, giving a corrective parallax signal. Without
  // latching (pos_b held constant relative to the current root) the target
  // chases the robot and that feedback is silently removed. Latch once at
  // policy start (the reference frame-0 root ≡ current root, same convention
  // as the ref-anchor rebase), then feed these world quantities every tick.
  bool world_latched{false};
  hope_math::Vec3 pos_w{};
  hope_math::Vec3 vel_w{};
  hope_math::Vec3 normal_w{};
  // Base target: fixed world point coupled to the racket target
  // (racket_xy - reach_offset, reach_offset from the reference strike frame).
  bool base_target_latched{false};
  hope_math::Vec3 base_target_w{};
};
struct HopeTargetEnvelope {
  bool valid{false};
  int motion_id{0};
  int strike_step{43};                 // metadata motion_strike_frames, else round(strike_phase * (len-1))
  // Reference strike racket state, TRAINING world (derived from the onnx
  // reference pass at the strike frame — see DeriveHopeTargetEnvelope).
  hope_math::Vec3 ref_pos_train{{0.0, 0.0, 0.0}};
  hope_math::Vec3 ref_normal_train{{1.0, 0.0, 0.0}};
  hope_math::Vec3 ref_vel_train{{3.0, 0.0, 0.0}};   // = hit_sign*speed*normal
  hope_math::Vec2 reach_xy_train{{0.0, 0.0}};       // racket_xy - base_xy @ strike
  // Reference BASE (pelvis) pose at the strike frame, TRAINING world. Only Z is
  // used: training's XYZ base target takes its height straight from the
  // reference pelvis at that frame (hope_commands.py::_resample_command), not
  // from the racket target, so it needs the same reference FK as the XY reach.
  hope_math::Vec3 ref_base_pos_train{{0.0, 0.0, 0.0}};
  double ref_speed{3.0};
  double hit_sign{-1.0};               // -1 backhand, +1 forehand (hit_type)
  // Training perturbation half-extents (HOPEPingPong.yaml), scaled by `scale`
  // (0 pins targets to the pure reference; 1 allows the full envelope).
  hope_math::Vec3 pos_half_extent{{0.08, 0.10, 0.08}};
  double vel_scale_half{0.20};
  double normal_half_extent{0.20};
  double scale{1.0};
};
inline hope_math::Vec3 vadd(const hope_math::Vec3& a, const hope_math::Vec3& b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
inline hope_math::Vec3 RacketCoupledBaseTarget(
    const hope_math::Vec3& racket_pos_w, const hope_math::Vec3& rebase_pos_w,
    const hope_math::Quat& rebase_quat_w, const HopeTargetEnvelope& env) {
  const hope_math::Vec3 p_train = hope_math::quat_apply_inverse(
      rebase_quat_w, hope_math::sub(racket_pos_w, rebase_pos_w));
  const hope_math::Vec3 base_train{p_train[0] - env.reach_xy_train[0],
                                   p_train[1] - env.reach_xy_train[1],
                                   env.ref_base_pos_train[2]};
  return vadd(rebase_pos_w, hope_math::quat_apply(rebase_quat_w, base_train));
}
inline hope_math::Vec3 LatchRacketTargetFromCommand(
    const hope_math::Vec3& cmd_pos_w, const hope_math::Vec3& cmd_vel_w,
    const hope_math::Vec3& cmd_normal_w, const hope_math::Vec3& rebase_pos_w,
    const hope_math::Quat& rebase_quat_w, const HopeTargetEnvelope& env,
    HopeRacketSpec& racket, bool clamp_to_envelope = true,
    double max_velocity_speed = 0.0, bool apply_motion_hit_sign_to_raw = false) {
  // --- position: deploy world -> training world, axis-aligned box clamp ------
  hope_math::Vec3 p_train =
      hope_math::quat_apply_inverse(rebase_quat_w, hope_math::sub(cmd_pos_w, rebase_pos_w));
  if (clamp_to_envelope) {
    for (int i = 0; i < 3; ++i) {
      const double half = env.pos_half_extent[i] * env.scale;
      p_train[i] = std::min(env.ref_pos_train[i] + half,
                            std::max(env.ref_pos_train[i] - half, p_train[i]));
    }
  }  // bypass: p_train stays = inverse-rebased planner point (round-trips to cmd_pos_w)
  racket.pos_w = vadd(rebase_pos_w, hope_math::quat_apply(rebase_quat_w, p_train));

  // --- normal: training semantics normalize(ref + d), d in a +-box -----------
  const hope_math::Vec3& r = env.ref_normal_train;
  hope_math::Vec3 n = hope_math::quat_apply_inverse(rebase_quat_w, cmd_normal_w);
  const double nn = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (nn < 1e-6) {
    n = r;
  } else if (clamp_to_envelope) {
    const double dmax = env.normal_half_extent * env.scale;
    hope_math::Vec3 d{n[0] / nn - r[0], n[1] / nn - r[1], n[2] / nn - r[2]};
    for (auto& c : d) c = std::min(dmax, std::max(-dmax, c));
    n = {r[0] + d[0], r[1] + d[1], r[2] + d[2]};
    const double n2 = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (n2 < 1e-6) n = r;
    else for (auto& c : n) c /= n2;
  } else {
    for (auto& c : n) c /= nn;
    // The command normal is the STRIKING face; the observation normal is the
    // racket's local +Y axis. They agree on a forehand and are opposite on a
    // backhand, so the raw normal takes hit_sign. Keep this separate from
    // clamping: it changes neither the command nor its magnitude.
    if (apply_motion_hit_sign_to_raw)
      for (auto& c : n) c *= env.hit_sign;
  }
  racket.normal_w = n;  // TRAINING world — fed raw into the obs

  // --- velocity: ref_vel * s (direction = unperturbed reference; |s-1|<=h) ---
  if (clamp_to_envelope) {
    const double cmd_speed =
        std::sqrt(cmd_vel_w[0] * cmd_vel_w[0] + cmd_vel_w[1] * cmd_vel_w[1] +
                  cmd_vel_w[2] * cmd_vel_w[2]);
    const double half = env.vel_scale_half * env.scale;
    const double s = env.ref_speed > 1e-9
                         ? std::min(1.0 + half, std::max(1.0 - half, cmd_speed / env.ref_speed))
                         : 1.0;
    racket.vel_w = {env.ref_vel_train[0] * s, env.ref_vel_train[1] * s, env.ref_vel_train[2] * s};
  } else {
    // Raw planner velocity VECTOR (dir + magnitude), deploy -> motion world.
    // NOT sign-flipped: the command velocity already is the physical racket
    // velocity the shot needs, and the observation term is that same physical
    // velocity. hit_sign only describes WHICH RACKET FACE the swing uses, which
    // is carried entirely by the normal above.
    racket.vel_w = hope_math::quat_apply_inverse(rebase_quat_w, cmd_vel_w);
  }
  // This is deliberately after the envelope/raw choice: callers can enforce a
  // hard observation-space speed limit without changing the selected direction.
  if (max_velocity_speed > 0.0) {
    const double speed = std::sqrt(racket.vel_w[0] * racket.vel_w[0] +
                                   racket.vel_w[1] * racket.vel_w[1] +
                                   racket.vel_w[2] * racket.vel_w[2]);
    if (speed > max_velocity_speed) {
      const double scale = max_velocity_speed / speed;
      for (auto& c : racket.vel_w) c *= scale;
    }
  }

  // --- coupled base target: racket_xy - reach (training world axes) ----------
  // Z is NOT coupled to the racket target: training takes it from the reference
  // pelvis height at the strike frame and only jitters it +-5 cm, so the pure
  // reference value is the in-distribution choice. It rides the same rebase as
  // XY, which makes the observed Z the clip's own height change relative to the
  // pose the rebase was anchored on — immune to any absolute-height offset
  // between the mocap frame and the training floor.
  racket.base_target_w =
      RacketCoupledBaseTarget(racket.pos_w, rebase_pos_w, rebase_quat_w, env);

  racket.world_latched = true;
  racket.base_target_latched = true;
  return p_train;
}
}

int main() { using namespace a3_deploy;
  double clamp; while(std::cin>>clamp) {
  HopeTargetEnvelope e; HopeRacketSpec r; hope_math::Vec3 p,v,n,t; hope_math::Quat q;
  for(auto* x : {&p,&v,&n,&t}) for(double& a:*x) std::cin>>a;
  for(double& a:q) std::cin>>a;
  for(auto* x:{&e.ref_pos_train,&e.ref_vel_train,&e.ref_normal_train,&e.ref_base_pos_train,&e.pos_half_extent}) for(double& a:*x) std::cin>>a;
  for(double& a:e.reach_xy_train) std::cin>>a;
  std::cin>>e.ref_speed>>e.hit_sign>>e.vel_scale_half>>e.normal_half_extent>>e.scale;
  LatchRacketTargetFromCommand(p,v,n,t,q,e,r,clamp!=0,0,true);
  std::cout<<std::setprecision(17);
  for(auto* x:{&r.pos_w,&r.vel_w,&r.normal_w,&r.base_target_w}) for(double a:*x) std::cout<<a<<' ';
  std::cout<<'\n'; }
}