// Copyright (c) 2026, AgiBot Inc. All rights reserved.
//
// hope_math.hpp — minimal quaternion / frame-transform math for the HOPE
// ping-pong policy observation builder. Ported bit-faithfully from
// IsaacLab/source/isaaclab/isaaclab/utils/math.py so the C++ deploy obs matches
// the training-time obs term-by-term. Quaternions are (w, x, y, z), matching
// IsaacLab. All operations are on single vectors (num_envs == 1 at deploy).
//
// Validated against the numpy port in
//   agibot/code_deployment/hope_integration/m1_offline_selfcheck.py
// which is in turn validated against the IsaacLab source (self-tests + golden).
#pragma once

#include <array>
#include <cmath>

namespace a3_deploy {
namespace hope_math {

using Vec3 = std::array<double, 3>;
using Vec2 = std::array<double, 2>;
using Quat = std::array<double, 4>;   // (w, x, y, z)
using Mat3 = std::array<double, 9>;   // row-major: [m00 m01 m02 m10 m11 m12 m20 m21 m22]

inline Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
  return {a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

inline Vec3 sub(const Vec3& a, const Vec3& b) noexcept {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// quat_apply: rotate vec by quat.  vec + w*t + xyz×t,  t = (xyz×vec)*2
inline Vec3 quat_apply(const Quat& q, const Vec3& v) noexcept {
  const Vec3 xyz{q[1], q[2], q[3]};
  Vec3 t = cross(xyz, v);
  t[0] *= 2.0; t[1] *= 2.0; t[2] *= 2.0;
  const Vec3 ct = cross(xyz, t);
  return {v[0] + q[0] * t[0] + ct[0],
          v[1] + q[0] * t[1] + ct[1],
          v[2] + q[0] * t[2] + ct[2]};
}

// quat_apply_inverse: rotate vec by inverse quat.  vec - w*t + xyz×t
inline Vec3 quat_apply_inverse(const Quat& q, const Vec3& v) noexcept {
  const Vec3 xyz{q[1], q[2], q[3]};
  Vec3 t = cross(xyz, v);
  t[0] *= 2.0; t[1] *= 2.0; t[2] *= 2.0;
  const Vec3 ct = cross(xyz, t);
  return {v[0] - q[0] * t[0] + ct[0],
          v[1] - q[0] * t[1] + ct[1],
          v[2] - q[0] * t[2] + ct[2]};
}

inline Quat quat_conjugate(const Quat& q) noexcept {
  return {q[0], -q[1], -q[2], -q[3]};
}

inline Quat quat_inv(const Quat& q, double eps = 1e-9) noexcept {
  double n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  if (n2 < eps) n2 = eps;
  const Quat c = quat_conjugate(q);
  return {c[0] / n2, c[1] / n2, c[2] / n2, c[3] / n2};
}

// quat_mul — IsaacLab's optimized Hamilton product (wxyz).
inline Quat quat_mul(const Quat& q1, const Quat& q2) noexcept {
  const double w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
  const double w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];
  const double ww = (z1 + x1) * (x2 + y2);
  const double yy = (w1 - y1) * (w2 + z2);
  const double zz = (w1 + y1) * (w2 - z2);
  const double xx = ww + yy + zz;
  const double qq = 0.5 * (xx + (z1 - x1) * (x2 - y2));
  const double w = qq - ww + (z1 - y1) * (y2 - z2);
  const double x = qq - xx + (x1 + w1) * (x2 + w2);
  const double y = qq - yy + (w1 - x1) * (y2 + z2);
  const double z = qq - zz + (z1 + y1) * (w2 - x2);
  return {w, x, y, z};
}

// matrix_from_quat — row-major 3x3, IsaacLab's two_s form.
inline Mat3 matrix_from_quat(const Quat& q) noexcept {
  const double r = q[0], i = q[1], j = q[2], k = q[3];
  const double two_s = 2.0 / (r * r + i * i + j * j + k * k);
  return {
      1.0 - two_s * (j * j + k * k), two_s * (i * j - k * r),       two_s * (i * k + j * r),
      two_s * (i * j + k * r),       1.0 - two_s * (i * i + k * k), two_s * (j * k - i * r),
      two_s * (i * k - j * r),       two_s * (j * k + i * r),       1.0 - two_s * (i * i + j * j),
  };
}

// First two columns of the rotation matrix, row-major flattened -> 6D.
// order: [m00, m01, m10, m11, m20, m21]  (matches torch mat[...,:2].reshape(-1))
inline std::array<double, 6> mat6d_from_quat(const Quat& q) noexcept {
  const Mat3 m = matrix_from_quat(q);
  return {m[0], m[1], m[3], m[4], m[6], m[7]};
}

// yaw_quat — quaternion with only the yaw component.
inline Quat yaw_quat(const Quat& q) noexcept {
  const double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
  const double yaw =
      std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
  Quat out{std::cos(yaw / 2.0), 0.0, 0.0, std::sin(yaw / 2.0)};
  const double n = std::sqrt(out[0] * out[0] + out[3] * out[3]);
  const double inv = (n > 1e-9) ? 1.0 / n : 1.0;
  out[0] *= inv; out[3] *= inv;
  return out;
}

// subtract_frame_transforms: T12 = inv(T01) * T02.  Returns (t12, q12).
struct FrameTf {
  Vec3 t;
  Quat q;
};
inline FrameTf subtract_frame_transforms(const Vec3& t01, const Quat& q01,
                                         const Vec3& t02, const Quat& q02) noexcept {
  const Quat q10 = quat_inv(q01);
  const Quat q12 = quat_mul(q10, q02);
  const Vec3 t12 = quat_apply(q10, sub(t02, t01));
  return {t12, q12};
}

// --- interpolation helpers (reference blending) ------------------------------
inline double lerp(double a, double b, double t) noexcept { return a + (b - a) * t; }

inline Vec3 lerp(const Vec3& a, const Vec3& b, double t) noexcept {
  return {a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t};
}

inline Quat normalize_quat(const Quat& q) noexcept {
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n < 1e-12) return Quat{{1.0, 0.0, 0.0, 0.0}};
  return {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

// Normalized LERP between quaternions along the shortest arc. Adequate for the
// small per-tick steps of reference blending (no full slerp needed). t in [0,1].
inline Quat quat_nlerp(const Quat& a, const Quat& b_in, double t) noexcept {
  const double dot = a[0] * b_in[0] + a[1] * b_in[1] + a[2] * b_in[2] + a[3] * b_in[3];
  const Quat b = (dot < 0.0) ? Quat{{-b_in[0], -b_in[1], -b_in[2], -b_in[3]}} : b_in;
  return normalize_quat(Quat{{a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
                              a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t}});
}

}  // namespace hope_math
}  // namespace a3_deploy
