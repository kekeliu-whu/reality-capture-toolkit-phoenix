#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <emmintrin.h>
#include <xmmintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

Eigen::Quaterniond binaryQuaternionInverse(
    const Eigen::Quaterniond& quaternion) {
  const double x = quaternion.x();
  const double y = quaternion.y();
  const double z = quaternion.z();
  const double w = quaternion.w();

  // Match the two packed lanes and final horizontal add used by the
  // installed Eigen build: (z*z + x*x) + (w*w + y*y).
  const double norm_lane_0 = z * z + x * x;
  const double norm_lane_1 = w * w + y * y;
  const double squared_norm = norm_lane_0 + norm_lane_1;
  return Eigen::Quaterniond(w / squared_norm, -x / squared_norm,
                            -y / squared_norm, -z / squared_norm);
}

Eigen::Quaterniond binaryQuaternionProduct(
    const Eigen::Quaterniond& lhs, const Eigen::Quaterniond& rhs) {
  const double ax = lhs.x();
  const double ay = lhs.y();
  const double az = lhs.z();
  const double aw = lhs.w();
  const double bx = rhs.x();
  const double by = rhs.y();
  const double bz = rhs.z();
  const double bw = rhs.w();

  // Reproduce Eigen's packed-double quaternion kernel as emitted in the
  // installed binary. The temporary grouping is observable when roll and
  // pitch cancel down to the last few bits.
  const double xy_t1_0 = aw * bx + ay * bz;
  const double xy_t1_1 = aw * by + ay * bw;
  const double xy_t2_0 = az * bx - ax * bz;
  const double xy_t2_1 = az * by - ax * bw;
  const double x = xy_t1_0 - xy_t2_1;
  const double y = xy_t1_1 + xy_t2_0;

  const double zw_t1_0 = aw * bz - ay * bx;
  const double zw_t1_1 = aw * bw - ay * by;
  const double zw_t2_0 = az * bz + ax * bx;
  const double zw_t2_1 = az * bw + ax * by;
  const double z = zw_t1_0 + zw_t2_1;
  const double w = zw_t1_1 - zw_t2_0;
  return Eigen::Quaterniond(w, x, y, z);
}

Eigen::Vector3d binaryQuaternionRotate(
    const Eigen::Quaterniond& quaternion, const Eigen::Vector3d& vector) {
  const double qx = quaternion.x();
  const double qy = quaternion.y();
  const double qz = quaternion.z();
  const double qw = quaternion.w();
  const double vx = vector.x();
  const double vy = vector.y();
  const double vz = vector.z();

  // Eigen's quaternion-vector kernel evaluates v + w*t + q.vec().cross(t),
  // where t = 2*q.vec().cross(v). Keep each observable operation separate.
  const double tx = 2.0 * (qy * vz - qz * vy);
  const double ty = 2.0 * (qz * vx - qx * vz);
  const double tz = 2.0 * (qx * vy - qy * vx);
  const double cx = qy * tz - qz * ty;
  const double cy = qz * tx - qx * tz;
  const double cz = qx * ty - qy * tx;
  return Eigen::Vector3d((vx + qw * tx) + cx,
                         (vy + qw * ty) + cy,
                         (vz + qw * tz) + cz);
}

Eigen::Quaterniond binaryQuaternionNormalized(
    const Eigen::Quaterniond& quaternion) {
  const double x = quaternion.x();
  const double y = quaternion.y();
  const double z = quaternion.z();
  const double w = quaternion.w();
  const double norm_lane_0 = z * z + x * x;
  const double norm_lane_1 = w * w + y * y;
  const double norm = std::sqrt(norm_lane_0 + norm_lane_1);
  return Eigen::Quaterniond(w / norm, x / norm, y / norm, z / norm);
}

Eigen::Vector3d binaryQuaternionTransformVector(
    const Eigen::Quaterniond& quaternion, const Eigen::Vector3d& vector) {
  return quaternion * vector;
}

Eigen::Vector3d binaryQuaternionRotationMatrixTransformVector(
    const Eigen::Quaterniond& quaternion, const Eigen::Vector3d& vector) {
  const double x = quaternion.x();
  const double y = quaternion.y();
  const double z = quaternion.z();
  const double w = quaternion.w();
  const double doubled_x = x + x;
  const double doubled_y = y + y;
  const double doubled_z = z + z;
  const double doubled_wx = doubled_x * w;
  const double doubled_wy = doubled_y * w;
  const double doubled_wz = doubled_z * w;
  const double doubled_xx = doubled_x * x;
  const double doubled_xy = doubled_y * x;
  const double doubled_xz = doubled_z * x;
  const double doubled_yy = doubled_y * y;
  const double doubled_yz = doubled_z * y;
  const double doubled_zz = doubled_z * z;
  const double row_00 = 1.0 - (doubled_yy + doubled_zz);
  const double row_01 = doubled_xy - doubled_wz;
  const double row_02 = doubled_xz + doubled_wy;
  const double row_10 = doubled_xy + doubled_wz;
  const double row_11 = 1.0 - (doubled_xx + doubled_zz);
  const double row_12 = doubled_yz - doubled_wx;
  const double row_20 = doubled_xz - doubled_wy;
  const double row_21 = doubled_yz + doubled_wx;
  const double row_22 = 1.0 - (doubled_xx + doubled_yy);
  const double output_x =
      (row_00 * vector.x() + row_01 * vector.y()) + row_02 * vector.z();
  const double output_y =
      (row_10 * vector.x() + row_11 * vector.y()) + row_12 * vector.z();
  const double output_z =
      row_20 * vector.x() +
      (row_21 * vector.y() + row_22 * vector.z());
  return Eigen::Vector3d(output_x, output_y, output_z);
}

Eigen::Quaterniond binaryRotationVectorQuaternion(
    const Eigen::Vector3d& rotation_vector) {
  const double x = rotation_vector.x();
  const double y = rotation_vector.y();
  const double z = rotation_vector.z();
  // The installed kernel evaluates z*z + (x*x + y*y), then forms one
  // sin(theta/2)/theta factor before multiplying all three coefficients.
  // This differs in the last bits from Eigen's usual (v/theta)*sin(theta/2)
  // expression even though the two formulas are algebraically identical.
  const double squared_angle = z * z + (x * x + y * y);
  if (!(squared_angle > 0.0)) {
    return Eigen::Quaterniond::Identity();
  }
  const double angle = std::sqrt(squared_angle);
  const double half_angle = 0.5 * angle;
  const double coefficient = std::sin(half_angle) / angle;
  return Eigen::Quaterniond(std::cos(half_angle), coefficient * x,
                            coefficient * y, coefficient * z);
}

struct FastImuIntegral {
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
};

[[gnu::always_inline]] inline Eigen::Vector3d clippedLinearIntegral(
    const Eigen::Vector3d& first, const Eigen::Vector3d& second,
    const double full_duration, const double left_clip,
    const double right_clip) {
  const double left_ratio = left_clip / full_duration;
  const double right_ratio = right_clip / full_duration;
  const Eigen::Vector3d first_boundary =
      left_ratio * second + (1.0 - left_ratio) * first;
  const Eigen::Vector3d second_boundary =
      right_ratio * first + (1.0 - right_ratio) * second;
  const double active_duration = full_duration - left_clip - right_clip;
  return (first_boundary + second_boundary) * 0.5 * active_duration;
}

Eigen::Vector3d clippedLinearIntegralVariant(
    const Eigen::Vector3d& first, const Eigen::Vector3d& second,
    const double full_duration, const double left_clip,
    const double right_clip, const std::uint32_t variant) {
  const double active_duration = full_duration - left_clip - right_clip;
  switch (variant) {
    case 0:
      return clippedLinearIntegral(first, second, full_duration, left_clip,
                                   right_clip);
    case 1: {
      const Eigen::Vector3d slope = (second - first) / full_duration;
      const Eigen::Vector3d first_boundary = first + left_clip * slope;
      const Eigen::Vector3d second_boundary = second - right_clip * slope;
      return (first_boundary + second_boundary) * 0.5 * active_duration;
    }
    case 2: {
      const Eigen::Vector3d slope = (second - first) / full_duration;
      const Eigen::Vector3d first_boundary = first + slope * left_clip;
      const Eigen::Vector3d second_boundary = second - slope * right_clip;
      return (first_boundary + second_boundary) * (0.5 * active_duration);
    }
    case 3: {
      const double left_ratio = left_clip / full_duration;
      const double right_ratio = right_clip / full_duration;
      const double first_weight = (1.0 - left_ratio) + right_ratio;
      const double second_weight = left_ratio + (1.0 - right_ratio);
      return (first_weight * first + second_weight * second) *
             (0.5 * active_duration);
    }
    case 4: {
      const double midpoint =
          (left_clip + (full_duration - right_clip)) * 0.5;
      const Eigen::Vector3d midpoint_value =
          first + (midpoint / full_duration) * (second - first);
      return midpoint_value * active_duration;
    }
    case 5: {
      const Eigen::Vector3d slope = (second - first) / full_duration;
      return active_duration *
             (first + slope * (left_clip + 0.5 * active_duration));
    }
    case 6: {
      const double right_time = full_duration - right_clip;
      const double slope_integral =
          0.5 * (right_time * right_time - left_clip * left_clip) /
          full_duration;
      return active_duration * first + slope_integral * (second - first);
    }
    case 7: {
      const double left_ratio = left_clip / full_duration;
      const double right_ratio = right_clip / full_duration;
      const Eigen::Vector3d first_boundary =
          first + left_ratio * (second - first);
      const Eigen::Vector3d second_boundary =
          second + right_ratio * (first - second);
      return (first_boundary + second_boundary) *
             (active_duration * 0.5);
    }
    case 8: {
      const double left_ratio = left_clip / full_duration;
      const double right_ratio = right_clip / full_duration;
      const Eigen::Vector3d first_boundary =
          first * (1.0 - left_ratio) + second * left_ratio;
      const Eigen::Vector3d second_boundary =
          second * (1.0 - right_ratio) + first * right_ratio;
      return active_duration * 0.5 *
             (first_boundary + second_boundary);
    }
    case 9: {
      const double left_ratio = left_clip / full_duration;
      const double right_ratio = right_clip / full_duration;
      const Eigen::Vector3d first_boundary =
          left_ratio * second + (1.0 - left_ratio) * first;
      const Eigen::Vector3d second_boundary =
          right_ratio * first + (1.0 - right_ratio) * second;
      const double scale = 0.5 * active_duration;
      return first_boundary * scale + second_boundary * scale;
    }
    case 10: {
      const double left_ratio = left_clip / full_duration;
      const double right_ratio = right_clip / full_duration;
      const Eigen::Vector3d first_boundary =
          left_ratio * second + (1.0 - left_ratio) * first;
      const Eigen::Vector3d second_boundary =
          right_ratio * first + (1.0 - right_ratio) * second;
      return first_boundary * 0.5 * active_duration +
             second_boundary * 0.5 * active_duration;
    }
    case 11: {
      const double left_ratio = left_clip / full_duration;
      const double right_ratio = right_clip / full_duration;
      const Eigen::Vector3d first_boundary =
          left_ratio * second + (1.0 - left_ratio) * first;
      const Eigen::Vector3d second_boundary =
          right_ratio * first + (1.0 - right_ratio) * second;
      return (first_boundary * active_duration +
              second_boundary * active_duration) *
             0.5;
    }
    case 12: {
      const double left_ratio = left_clip / full_duration;
      const double right_ratio = right_clip / full_duration;
      const Eigen::Vector3d first_boundary =
          left_ratio * second + (1.0 - left_ratio) * first;
      const Eigen::Vector3d second_boundary =
          right_ratio * first + (1.0 - right_ratio) * second;
      return first_boundary * (active_duration * 0.5) +
             second_boundary * (active_duration * 0.5);
    }
    default:
      return Eigen::Vector3d::Constant(
          std::numeric_limits<double>::quiet_NaN());
  }
}

FastImuIntegral integrateFastImu(
    const std::uint64_t sample_count, const std::int64_t* const timestamps_ns,
    const double* const linear_accelerations,
    const double* const angular_velocities, const std::int64_t begin_ns,
    const std::int64_t end_ns, double* const trace = nullptr,
    const std::uint64_t trace_capacity = 0,
    std::uint64_t* const trace_count = nullptr) {
  FastImuIntegral result;
  std::uint64_t step = 0;
  std::uint64_t index = 0;
  while (index + 1 < sample_count && timestamps_ns[index + 1] <= begin_ns) {
    ++index;
  }
  while (index + 1 < sample_count && timestamps_ns[index] < end_ns) {
    const std::int64_t first_ns = timestamps_ns[index];
    const std::int64_t second_ns = timestamps_ns[index + 1];
    const double full_duration =
        static_cast<double>(second_ns - first_ns) / 1.0e9;
    const double left_clip =
        static_cast<double>(std::max(begin_ns, first_ns) - first_ns) / 1.0e9;
    const double right_clip =
        static_cast<double>(second_ns - std::min(end_ns, second_ns)) / 1.0e9;
    const Eigen::Map<const Eigen::Vector3d> first_acceleration(
        linear_accelerations + 3 * index);
    const Eigen::Map<const Eigen::Vector3d> second_acceleration(
        linear_accelerations + 3 * (index + 1));
    const Eigen::Map<const Eigen::Vector3d> first_angular_velocity(
        angular_velocities + 3 * index);
    const Eigen::Map<const Eigen::Vector3d> second_angular_velocity(
        angular_velocities + 3 * (index + 1));
    const Eigen::Vector3d delta_angle = clippedLinearIntegral(
        first_angular_velocity, second_angular_velocity, full_duration,
        left_clip, right_clip);
    const Eigen::Quaterniond next_rotation = binaryQuaternionProduct(
        result.rotation, binaryRotationVectorQuaternion(delta_angle));
    const Eigen::Vector3d first_rotated_acceleration =
        binaryQuaternionRotationMatrixTransformVector(result.rotation,
                                                       first_acceleration);
    const Eigen::Vector3d second_rotated_acceleration =
        binaryQuaternionRotationMatrixTransformVector(next_rotation,
                                                       second_acceleration);
    result.velocity += clippedLinearIntegral(
        first_rotated_acceleration, second_rotated_acceleration, full_duration,
        left_clip, right_clip);
    result.rotation = next_rotation;
    if (trace != nullptr && step < trace_capacity) {
      double* const output = trace + 7 * step;
      output[0] = result.velocity.x();
      output[1] = result.velocity.y();
      output[2] = result.velocity.z();
      output[3] = result.rotation.x();
      output[4] = result.rotation.y();
      output[5] = result.rotation.z();
      output[6] = result.rotation.w();
    }
    ++step;
    ++index;
  }
  if (trace_count != nullptr) {
    *trace_count = step;
  }
  return result;
}

FastImuIntegral integrateFastImuVariant(
    const std::uint64_t sample_count, const std::int64_t* const timestamps_ns,
    const double* const linear_accelerations,
    const double* const angular_velocities, const std::int64_t begin_ns,
    const std::int64_t end_ns, const std::uint32_t variant) {
  FastImuIntegral result;
  std::uint64_t index = 0;
  while (index + 1 < sample_count && timestamps_ns[index + 1] <= begin_ns) {
    ++index;
  }
  while (index + 1 < sample_count && timestamps_ns[index] < end_ns) {
    const std::int64_t first_ns = timestamps_ns[index];
    const std::int64_t second_ns = timestamps_ns[index + 1];
    const double full_duration =
        static_cast<double>(second_ns - first_ns) / 1.0e9;
    const double left_clip =
        static_cast<double>(std::max(begin_ns, first_ns) - first_ns) / 1.0e9;
    const double right_clip =
        static_cast<double>(second_ns - std::min(end_ns, second_ns)) / 1.0e9;
    const Eigen::Map<const Eigen::Vector3d> first_acceleration(
        linear_accelerations + 3 * index);
    const Eigen::Map<const Eigen::Vector3d> second_acceleration(
        linear_accelerations + 3 * (index + 1));
    const Eigen::Map<const Eigen::Vector3d> first_angular_velocity(
        angular_velocities + 3 * index);
    const Eigen::Map<const Eigen::Vector3d> second_angular_velocity(
        angular_velocities + 3 * (index + 1));
    const Eigen::Vector3d delta_angle = clippedLinearIntegral(
        first_angular_velocity, second_angular_velocity, full_duration,
        left_clip, right_clip);
    const Eigen::Quaterniond next_rotation = binaryQuaternionProduct(
        result.rotation, binaryRotationVectorQuaternion(delta_angle));
    const Eigen::Vector3d first_rotated_acceleration =
        binaryQuaternionRotationMatrixTransformVector(result.rotation,
                                                       first_acceleration);
    const Eigen::Vector3d second_rotated_acceleration =
        binaryQuaternionRotationMatrixTransformVector(next_rotation,
                                                       second_acceleration);
    if (variant < 13) {
      result.velocity += clippedLinearIntegralVariant(
          first_rotated_acceleration, second_rotated_acceleration,
          full_duration, left_clip, right_clip, variant);
    } else {
      const double left_ratio = left_clip / full_duration;
      const double right_ratio = right_clip / full_duration;
      const Eigen::Vector3d first_boundary =
          left_ratio * second_rotated_acceleration +
          (1.0 - left_ratio) * first_rotated_acceleration;
      const Eigen::Vector3d second_boundary =
          right_ratio * first_rotated_acceleration +
          (1.0 - right_ratio) * second_rotated_acceleration;
      const double scale = 0.5 *
                           (full_duration - left_clip - right_clip);
      switch (variant) {
        case 13:
          result.velocity =
              (first_boundary + second_boundary) * scale + result.velocity;
          break;
        case 14:
          result.velocity = result.velocity +
                            first_boundary * scale +
                            second_boundary * scale;
          break;
        case 15:
          result.velocity = (result.velocity + first_boundary * scale) +
                            second_boundary * scale;
          break;
        case 16:
          result.velocity += first_boundary * scale;
          result.velocity += second_boundary * scale;
          break;
        case 17:
          result.velocity = result.velocity +
                            (first_boundary * scale +
                             second_boundary * scale);
          break;
        default:
          result.velocity.setConstant(
              std::numeric_limits<double>::quiet_NaN());
          break;
      }
    }
    result.rotation = next_rotation;
    ++index;
  }
  return result;
}

Eigen::Quaterniond binaryQuaternionFromTwoVectors(
    const Eigen::Vector3d& first, const Eigen::Vector3d& second) {
  const double first_norm = std::sqrt(
      (first.x() * first.x() + first.y() * first.y()) +
      first.z() * first.z());
  const double second_norm = std::sqrt(
      (second.x() * second.x() + second.y() * second.y()) +
      second.z() * second.z());
  if (!(first_norm > 0.0) || !(second_norm > 0.0)) {
    return Eigen::Quaterniond::Identity();
  }
  const double first_x = first.x() / first_norm;
  const double first_y = first.y() / first_norm;
  const double first_z = first.z() / first_norm;
  const double second_x = second.x() / second_norm;
  const double second_y = second.y() / second_norm;
  const double second_z = second.z() / second_norm;
  const double dot =
      (first_x * second_x + first_y * second_y) + first_z * second_z;

  // The regular Eigen::Quaternion::FromTwoVectors branch emitted in the
  // installed image.  The SVD antiparallel fallback is irrelevant for the
  // gravity tracker, whose vectors remain nearly parallel, but is retained
  // through Eigen for completeness.
  if (dot < -1.0 + std::numeric_limits<double>::epsilon()) {
    return Eigen::Quaterniond::FromTwoVectors(first, second);
  }
  const double doubled_one_plus_dot = (dot + 1.0) + (dot + 1.0);
  const double scale = std::sqrt(doubled_one_plus_dot);
  // The installed Eigen kernel computes one reciprocal and multiplies all
  // three cross-product lanes by it.  Three independent divisions are
  // algebraically equivalent, but can move the small gravity correction by
  // one ULP and that difference accumulates in the IMU predictor.
  const double inverse_scale = 1.0 / scale;
  const double x =
      (first_y * second_z - first_z * second_y) * inverse_scale;
  const double y =
      (first_z * second_x - first_x * second_z) * inverse_scale;
  const double z =
      (first_x * second_y - first_y * second_x) * inverse_scale;
  const double w = scale * 0.5;
  return Eigen::Quaterniond(w, x, y, z);
}

inline bool valid_label(std::uint64_t label, std::uint64_t state_count) {
  return label < state_count;
}

bool surfel_geometry(const float* const values, Eigen::Vector3f* const normal,
                     Eigen::Vector3f* const eigenvalues) {
  Eigen::Matrix3f covariance;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      // BaseSurfel stores Matrix3f in Eigen's default column-major layout.
      covariance(row, column) = values[3 * column + row];
    }
  }
  covariance = covariance.selfadjointView<Eigen::Lower>();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(
      covariance, Eigen::ComputeEigenvectors);
  if (solver.info() != Eigen::Success) {
    return false;
  }
  *eigenvalues = solver.eigenvalues();
  *normal = solver.eigenvectors().col(0);
  return true;
}

bool valid_merge_surfel(const float weight,
                        const Eigen::Vector3f& eigenvalues) {
  const float smallest = eigenvalues[0];
  const float middle = eigenvalues[1];
  const float largest = eigenvalues[2];
  const float total = (middle + largest) + smallest;
  const float planarity = static_cast<float>(
      2.0 * static_cast<double>(middle - smallest) /
      static_cast<double>(total));
  const float curvature = static_cast<float>(
      3.0 * static_cast<double>(smallest) / static_cast<double>(total));
  const float minor_spread = static_cast<float>(
      2.0 * std::sqrt(3.0 * static_cast<double>(middle)));
  const float normal_spread = static_cast<float>(
      2.0 * std::sqrt(3.0 * static_cast<double>(smallest)));
  return weight >= 3.5F && planarity >= 0.5F && curvature <= 0.5F &&
         minor_spread >= 0.075F && normal_spread <= 0.05F;
}

bool valid_split_surfel(const float weight,
                        const Eigen::Vector3f& eigenvalues) {
  const float smallest = eigenvalues[0];
  const float middle = eigenvalues[1];
  const float largest = eigenvalues[2];
  const float total = (middle + largest) + smallest;
  const float planarity = static_cast<float>(
      2.0 * static_cast<double>(middle - smallest) /
      static_cast<double>(total));
  const float curvature = static_cast<float>(
      3.0 * static_cast<double>(smallest) / static_cast<double>(total));
  const float major_spread = static_cast<float>(
      2.0 * std::sqrt(3.0 * static_cast<double>(largest)));
  const float minor_spread = static_cast<float>(
      2.0 * std::sqrt(3.0 * static_cast<double>(middle)));
  const float normal_spread = static_cast<float>(
      2.0 * std::sqrt(3.0 * static_cast<double>(smallest)));
  return weight >= 10.0F && planarity >= 0.5F && curvature <= 0.5F &&
         major_spread >= 0.0F && minor_spread >= 0.075F &&
         normal_spread <= 0.05F;
}

void add_unit_weight_surfel(float* const weight, std::uint32_t* const count,
                            float* const mean, float* const covariance,
                            float* const viewpoint_mean,
                            const float* const origin, const float* const point) {
  const float old_weight = *weight;
  const float new_weight = old_weight + 1.0F;
  const float reciprocal_weight = 1.0F / new_weight;

  const float delta_x = point[0] - mean[0];
  const float delta_y = point[1] - mean[1];
  const float delta_z = point[2] - mean[2];
  const float new_mean_x = mean[0] + reciprocal_weight * delta_x;
  const float new_mean_y = mean[1] + reciprocal_weight * delta_y;
  const float new_mean_z = mean[2] + reciprocal_weight * delta_z;

  const float remainder_x = point[0] - new_mean_x;
  const float remainder_y = point[1] - new_mean_y;
  const float remainder_z = point[2] - new_mean_z;
  const float delta[3] = {delta_x, delta_y, delta_z};
  const float remainder[3] = {remainder_x, remainder_y, remainder_z};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      const int offset = 3 * row + column;
      covariance[offset] =
          (covariance[offset] * old_weight +
           remainder[row] * delta[column]) *
          reciprocal_weight;
    }
  }

  mean[0] = new_mean_x;
  mean[1] = new_mean_y;
  mean[2] = new_mean_z;
  for (int axis = 0; axis < 3; ++axis) {
    viewpoint_mean[axis] =
        (viewpoint_mean[axis] * old_weight + origin[axis]) *
        reciprocal_weight;
  }
  *weight = new_weight;
  ++*count;
}

void merge_base_surfel(float* const first_weight,
                       std::uint32_t* const first_count,
                       float* const first_mean,
                       float* const first_covariance,
                       float* const first_viewpoint,
                       const float second_weight,
                       const std::uint32_t second_count,
                       const float* const second_mean,
                       const float* const second_covariance,
                       const float* const second_viewpoint) {
  if (second_weight == 0.0F) {
    return;
  }
  const float old_weight = *first_weight;
  // BaseSurfel::add does not special-case an empty destination.  Even with
  // zero old weight it evaluates the weighted mean/covariance expressions,
  // so multiplying and dividing the live source by its own weight can move a
  // few float fields by one ulp.  This is observable when a frozen directed
  // merge later canonicalizes its surviving neighbour.
  const float combined_weight = old_weight + second_weight;
  const float reciprocal_weight = 1.0F / combined_weight;
  float combined_mean[3];
  float combined_viewpoint[3];
  for (int axis = 0; axis < 3; ++axis) {
    combined_mean[axis] =
        (first_mean[axis] * old_weight +
         second_mean[axis] * second_weight) *
        reciprocal_weight;
    combined_viewpoint[axis] =
        (first_viewpoint[axis] * old_weight +
         second_viewpoint[axis] * second_weight) *
        reciprocal_weight;
  }
  for (int row = 0; row < 3; ++row) {
    const float first_delta = first_mean[row] - combined_mean[row];
    const float second_delta = second_mean[row] - combined_mean[row];
    for (int column = 0; column < 3; ++column) {
      const float first_remainder =
          first_mean[column] - combined_mean[column];
      const float second_remainder =
          second_mean[column] - combined_mean[column];
      const int offset = 3 * row + column;
      // BaseSurfel::merge normalizes the covariance and between-centroid
      // terms separately, then adds them.  Keeping this instruction grouping
      // matches the installed SSE kernel down to float32 rounding.
      const float weighted_covariance =
          (first_covariance[offset] * old_weight +
           second_covariance[offset] * second_weight) *
          reciprocal_weight;
      const float first_weighted_remainder =
          first_remainder * old_weight;
      const float second_weighted_remainder =
          second_remainder * second_weight;
      const float centroid_correction =
          (first_delta * first_weighted_remainder +
           second_delta * second_weighted_remainder) *
          reciprocal_weight;
      first_covariance[offset] =
          weighted_covariance + centroid_correction;
    }
  }
  std::copy_n(combined_mean, 3, first_mean);
  std::copy_n(combined_viewpoint, 3, first_viewpoint);
  *first_weight = combined_weight;
  *first_count += second_count;
}

void orient_surfel_normal(const float* const mean,
                          const float* const viewpoint_mean,
                          Eigen::Vector3f* const normal) {
  float dot = (mean[2] - viewpoint_mean[2]) * (*normal)[2];
  dot += (mean[1] - viewpoint_mean[1]) * (*normal)[1];
  dot += (mean[0] - viewpoint_mean[0]) * (*normal)[0];
  if (dot > 0.0F) {
    *normal = -*normal;
  }
}

constexpr float kMinProbability = 0.1F;
constexpr float kMaxProbability = 0.9F;
constexpr std::uint16_t kUpdateMarker = 1U << 15;
constexpr std::uint64_t kCellMask = (1ULL << 21) - 1ULL;
constexpr std::int32_t kMinimumCell = -(1 << 20);
constexpr std::int32_t kMaximumCell = (1 << 20) - 1;

inline bool pack_cell(const std::int32_t x, const std::int32_t y,
                      const std::int32_t z, std::uint64_t* const key) {
  if (x < kMinimumCell || x > kMaximumCell || y < kMinimumCell ||
      y > kMaximumCell || z < kMinimumCell || z > kMaximumCell) {
    return false;
  }
  *key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) &
          kCellMask) << 42;
  *key |= (static_cast<std::uint64_t>(static_cast<std::uint32_t>(y)) &
           kCellMask) << 21;
  *key |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(z)) &
          kCellMask;
  return true;
}

inline float value_to_probability(const std::uint16_t value) {
  if (value == 0) {
    return kMinProbability;
  }
  constexpr float kScale =
      (kMaxProbability - kMinProbability) / (32768.F - 2.F);
  return static_cast<float>(value) * kScale +
         (kMinProbability - kScale);
}

inline std::uint16_t probability_to_value(const float probability) {
  const float bounded =
      std::min(kMaxProbability, std::max(kMinProbability, probability));
  return static_cast<std::uint16_t>(
      std::lround((bounded - kMinProbability) *
                  (32766.F / (kMaxProbability - kMinProbability))) +
      1);
}

inline float odds(const float probability) {
  return probability / (1.F - probability);
}

inline float probability_from_odds(const float value) {
  return value / (value + 1.F);
}

std::vector<std::uint16_t> make_update_table(const float probability) {
  std::vector<std::uint16_t> table(32768);
  const float update_odds = odds(probability);
  table[0] = probability_to_value(probability_from_odds(update_odds)) +
             kUpdateMarker;
  for (std::uint32_t value = 1; value < 32768; ++value) {
    table[value] = probability_to_value(probability_from_odds(
                       update_odds * odds(value_to_probability(value)))) +
                   kUpdateMarker;
  }
  return table;
}

struct ProbabilityGrid {
  explicit ProbabilityGrid(const float resolution_value)
      : resolution(resolution_value),
        hit_table(make_update_table(0.9F)),
        miss_table(make_update_table(0.1F)) {}

  float resolution;
  std::unordered_map<std::uint64_t, std::uint16_t> cells;
  std::vector<std::uint16_t> hit_table;
  std::vector<std::uint16_t> miss_table;
};

struct RangeCentroid {
  std::uint64_t key = 0;
  float origin[3] = {0.F, 0.F, 0.F};
  float point[3] = {0.F, 0.F, 0.F};
  float count = 0.F;
  std::uint8_t octant = 0;
};

inline bool point_cell(const float* const point, const float resolution,
                       std::int32_t* const output) {
  for (int axis = 0; axis < 3; ++axis) {
    const long value = std::lround(point[axis] / resolution);
    if (value < kMinimumCell || value > kMaximumCell) {
      return false;
    }
    output[axis] = static_cast<std::int32_t>(value);
  }
  return true;
}

inline void apply_table(ProbabilityGrid* const grid, const std::uint64_t key,
                        const std::vector<std::uint16_t>& table) {
  const auto found = grid->cells.find(key);
  const std::uint16_t previous =
      found == grid->cells.end() ? 0 : found->second;
  grid->cells[key] = table[previous] - kUpdateMarker;
}

struct FloatPoseMatrix {
  __m128 columns[4];
};

FloatPoseMatrix make_float_pose_matrix(const float* const translation,
                                       const float* const quaternion_xyzw) {
  const float x = quaternion_xyzw[0];
  const float y = quaternion_xyzw[1];
  const float z = quaternion_xyzw[2];
  const float w = quaternion_xyzw[3];
  const float x2 = x + x;
  const float y2 = y + y;
  const float z2 = z + z;

  const float xx2 = x * x2;
  const float xy2 = x * y2;
  const float xz2 = x * z2;
  const float xw2 = w * x2;
  const float yy2 = y * y2;
  const float yz2 = y * z2;
  const float yw2 = w * y2;
  const float zz2 = z * z2;
  const float zw2 = w * z2;

  FloatPoseMatrix result;
  result.columns[0] =
      _mm_set_ps(0.F, xz2 - yw2, xy2 + zw2, 1.F - (yy2 + zz2));
  result.columns[1] =
      _mm_set_ps(0.F, yz2 + xw2, 1.F - (xx2 + zz2), xy2 - zw2);
  result.columns[2] =
      _mm_set_ps(0.F, 1.F - (xx2 + yy2), yz2 - xw2, xz2 + yw2);
  result.columns[3] =
      _mm_set_ps(1.F, translation[2], translation[1], translation[0]);
  return result;
}

void transform_point_exact(const FloatPoseMatrix& pose,
                           const float* const input, float* const output) {
  const __m128 transformed = _mm_add_ps(
      _mm_add_ps(
          _mm_add_ps(_mm_mul_ps(_mm_set1_ps(input[0]), pose.columns[0]),
                     _mm_mul_ps(_mm_set1_ps(input[1]), pose.columns[1])),
          _mm_mul_ps(_mm_set1_ps(input[2]), pose.columns[2])),
      pose.columns[3]);
  alignas(16) float values[4];
  _mm_store_ps(values, transformed);
  output[0] = values[0];
  output[1] = values[1];
  output[2] = values[2];
}

// The installed ICP correspondence finder uses Jens Behley's UniBN octree
// with a 32-point leaf.  Its nearest-neighbour result is observably different
// from a KD tree at float32 pruning boundaries, so binary-compatible ICP must
// reproduce the tree construction and depth-first traversal as well as the
// point-distance arithmetic.  This is a compact, read-only implementation of
// that MIT-licensed index specialized for contiguous xyz float points.
class BinaryOctree {
 public:
  BinaryOctree(const std::uint64_t point_count, const float* const points)
      : points_(3 * point_count), successors_(point_count) {
    std::copy(points, points + 3 * point_count, points_.begin());
    if (point_count == 0) {
      return;
    }

    float minimum[3] = {points_[0], points_[1], points_[2]};
    float maximum[3] = {points_[0], points_[1], points_[2]};
    for (std::uint32_t index = 0; index < point_count; ++index) {
      successors_[index] = index + 1;
      const float* const point = points_.data() + 3 * index;
      for (int axis = 0; axis < 3; ++axis) {
        if (point[axis] < minimum[axis]) {
          minimum[axis] = point[axis];
        }
        if (point[axis] > maximum[axis]) {
          maximum[axis] = point[axis];
        }
      }
    }

    float center[3] = {minimum[0], minimum[1], minimum[2]};
    float maximum_extent = 0.5F * (maximum[0] - minimum[0]);
    center[0] += maximum_extent;
    for (int axis = 1; axis < 3; ++axis) {
      const float extent = 0.5F * (maximum[axis] - minimum[axis]);
      center[axis] += extent;
      if (extent > maximum_extent) {
        maximum_extent = extent;
      }
    }
    root_ = create_node(center[0], center[1], center[2], maximum_extent,
                        0, static_cast<std::uint32_t>(point_count - 1),
                        static_cast<std::uint32_t>(point_count));
  }

  BinaryOctree(const BinaryOctree&) = delete;
  BinaryOctree& operator=(const BinaryOctree&) = delete;

  bool nearest(const float* const query, const float search_radius,
               std::uint64_t* const result_index,
               float* const result_distance_squared) const {
    if (!root_) {
      return false;
    }
    float maximum_distance = search_radius;
    std::int32_t index = -1;
    find_nearest(root_.get(), query, -1.F, &maximum_distance, &index);
    if (index < 0) {
      return false;
    }
    *result_index = static_cast<std::uint64_t>(index);

    // OctreeUniBN recomputes the public squared distance after the search in
    // z,y,x order.  The octree itself ranks leaf points in x,y,z order.
    const float* const point = points_.data() + 3 * index;
    const float dx = query[0] - point[0];
    const float dy = query[1] - point[1];
    const float dz = query[2] - point[2];
    *result_distance_squared = (dz * dz + dy * dy) + dx * dx;
    return true;
  }

 private:
  static constexpr std::uint32_t kBucketSize = 32;

  struct Node {
    bool is_leaf = true;
    float center[3] = {0.F, 0.F, 0.F};
    float extent = 0.F;
    std::uint32_t start = 0;
    std::uint32_t end = 0;
    std::uint32_t size = 0;
    std::unique_ptr<Node> children[8];
  };

  std::unique_ptr<Node> create_node(const float x, const float y,
                                    const float z, const float extent,
                                    const std::uint32_t start,
                                    const std::uint32_t end,
                                    const std::uint32_t size) {
    auto node = std::make_unique<Node>();
    node->center[0] = x;
    node->center[1] = y;
    node->center[2] = z;
    node->extent = extent;
    node->start = start;
    node->end = end;
    node->size = size;
    if (size <= kBucketSize) {
      return node;
    }

    node->is_leaf = false;
    std::uint32_t child_starts[8] = {};
    std::uint32_t child_ends[8] = {};
    std::uint32_t child_sizes[8] = {};
    std::uint32_t index = start;
    for (std::uint32_t offset = 0; offset < size; ++offset) {
      const float* const point = points_.data() + 3 * index;
      std::uint32_t code = 0;
      if (point[0] > x) {
        code |= 1;
      }
      if (point[1] > y) {
        code |= 2;
      }
      if (point[2] > z) {
        code |= 4;
      }
      if (child_sizes[code] == 0) {
        child_starts[code] = index;
      } else {
        successors_[child_ends[code]] = index;
      }
      ++child_sizes[code];
      child_ends[code] = index;
      index = successors_[index];
    }

    constexpr float kOffset[2] = {-0.5F, 0.5F};
    const float child_extent = 0.5F * extent;
    bool first_child = true;
    std::uint32_t previous_child = 0;
    for (std::uint32_t code = 0; code < 8; ++code) {
      if (child_sizes[code] == 0) {
        continue;
      }
      node->children[code] = create_node(
          x + kOffset[(code & 1) != 0] * extent,
          y + kOffset[(code & 2) != 0] * extent,
          z + kOffset[(code & 4) != 0] * extent, child_extent,
          child_starts[code], child_ends[code], child_sizes[code]);
      if (first_child) {
        node->start = node->children[code]->start;
      } else {
        successors_[node->children[previous_child]->end] =
            node->children[code]->start;
      }
      previous_child = code;
      node->end = node->children[code]->end;
      first_child = false;
    }
    return node;
  }

  static float squared_norm(const float x, const float y, const float z) {
    return (x * x + y * y) + z * z;
  }

  static bool inside(const float* const query, const float radius,
                     const Node* const node) {
    const float x = std::abs(query[0] - node->center[0]) + radius;
    const float y = std::abs(query[1] - node->center[1]) + radius;
    const float z = std::abs(query[2] - node->center[2]) + radius;
    return x <= node->extent && y <= node->extent && z <= node->extent;
  }

  static bool overlaps(const float* const query, const float radius,
                       const float squared_radius, const Node* const node) {
    float x = std::abs(query[0] - node->center[0]);
    float y = std::abs(query[1] - node->center[1]);
    float z = std::abs(query[2] - node->center[2]);
    const float maximum_distance = radius + node->extent;
    if (x > maximum_distance || y > maximum_distance ||
        z > maximum_distance) {
      return false;
    }
    const std::int32_t inside_axes = (x < node->extent) +
                                     (y < node->extent) +
                                     (z < node->extent);
    if (inside_axes > 1) {
      return true;
    }
    x = std::max(x - node->extent, 0.F);
    y = std::max(y - node->extent, 0.F);
    z = std::max(z - node->extent, 0.F);
    return squared_norm(x, y, z) < squared_radius;
  }

  bool find_nearest(const Node* const node, const float* const query,
                    const float minimum_distance, float* maximum_distance,
                    std::int32_t* const result_index) const {
    if (node->is_leaf) {
      float squared_maximum = *maximum_distance * *maximum_distance;
      const float squared_minimum = minimum_distance < 0.F
                                        ? minimum_distance
                                        : minimum_distance * minimum_distance;
      std::uint32_t index = node->start;
      for (std::uint32_t offset = 0; offset < node->size; ++offset) {
        const float* const point = points_.data() + 3 * index;
        const float dx = query[0] - point[0];
        const float dy = query[1] - point[1];
        const float dz = query[2] - point[2];
        const float distance = squared_norm(dx, dy, dz);
        if (distance > squared_minimum && distance < squared_maximum) {
          *result_index = static_cast<std::int32_t>(index);
          squared_maximum = distance;
        }
        index = successors_[index];
      }
      *maximum_distance = std::sqrt(squared_maximum);
      return inside(query, *maximum_distance, node);
    }

    std::uint32_t query_code = 0;
    if (query[0] > node->center[0]) {
      query_code |= 1;
    }
    if (query[1] > node->center[1]) {
      query_code |= 2;
    }
    if (query[2] > node->center[2]) {
      query_code |= 4;
    }
    if (node->children[query_code] &&
        find_nearest(node->children[query_code].get(), query,
                     minimum_distance, maximum_distance, result_index)) {
      return true;
    }

    const float squared_maximum = *maximum_distance * *maximum_distance;
    for (std::uint32_t code = 0; code < 8; ++code) {
      if (code == query_code || !node->children[code] ||
          !overlaps(query, *maximum_distance, squared_maximum,
                    node->children[code].get())) {
        continue;
      }
      if (find_nearest(node->children[code].get(), query, minimum_distance,
                       maximum_distance, result_index)) {
        return true;
      }
    }
    return inside(query, *maximum_distance, node);
  }

  std::vector<float> points_;
  std::vector<std::uint32_t> successors_;
  std::unique_ptr<Node> root_;
};

}  // namespace

extern "C" {

int navvis_recon_slam_submap_rotation(
    const double* const node_quaternion_xyzw,
    const double* const gravity_observation,
    double* const output_quaternion_xyzw) {
  if (node_quaternion_xyzw == nullptr || gravity_observation == nullptr ||
      output_quaternion_xyzw == nullptr) {
    return 1;
  }
  const Eigen::Quaterniond node_rotation(
      node_quaternion_xyzw[3], node_quaternion_xyzw[0],
      node_quaternion_xyzw[1], node_quaternion_xyzw[2]);
  const Eigen::Vector3d gravity(gravity_observation[0],
                                gravity_observation[1],
                                gravity_observation[2]);
  const Eigen::Quaterniond inverse_node_rotation =
      binaryQuaternionInverse(node_rotation);
  const Eigen::Vector3d implied_gravity = binaryQuaternionTransformVector(
      inverse_node_rotation, Eigen::Vector3d::UnitZ());
  const Eigen::Quaterniond remove_tilt =
      binaryQuaternionFromTwoVectors(gravity, implied_gravity);
  // Keep the complete conjugation chain observed at 0x48c4c0. Collapsing
  // node_rotation with its inverse removes the tiny z coefficient retained by
  // the installed submap pose.
  const Eigen::Quaterniond corrected_rotation =
      binaryQuaternionProduct(node_rotation, remove_tilt);
  const Eigen::Quaterniond normalization_rotation = binaryQuaternionProduct(
      corrected_rotation, inverse_node_rotation);
  const Eigen::Quaterniond submap_rotation =
      binaryQuaternionInverse(normalization_rotation);
  output_quaternion_xyzw[0] = submap_rotation.x();
  output_quaternion_xyzw[1] = submap_rotation.y();
  output_quaternion_xyzw[2] = submap_rotation.z();
  output_quaternion_xyzw[3] = submap_rotation.w();
  return 0;
}

int navvis_recon_slam_raw_imu_initialize_gravity(
    const double gravity_magnitude, const double* const quaternion_xyzw,
    double* const output_gravity) {
  if (quaternion_xyzw == nullptr || output_gravity == nullptr) {
    return 1;
  }
  const Eigen::Quaterniond orientation(
      quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1],
      quaternion_xyzw[2]);
  const Eigen::Quaterniond inverse = binaryQuaternionInverse(orientation);
  const double x = inverse.x();
  const double y = inverse.y();
  const double z = inverse.z();
  const double w = inverse.w();
  // This is the packed Eigen transform emitted in the installed
  // initialization branch.  Keep its doubled terms and zero-subtractions
  // explicit; 2*(x*z + w*y) rounds differently for this firmware quaternion.
  const double two_y = y + y;
  const double two_negative_x = (-x) + (-x);
  const double transformed_x =
      ((0.0 - z * two_negative_x) + w * two_y) * gravity_magnitude;
  const double transformed_y =
      ((two_y * z - x * 0.0) + w * two_negative_x) * gravity_magnitude;
  const double transformed_z =
      ((two_negative_x * x - two_y * y) + 1.0) * gravity_magnitude;
  output_gravity[0] = transformed_x;
  output_gravity[1] = transformed_y;
  output_gravity[2] = transformed_z;
  return 0;
}

int navvis_recon_slam_raw_imu_angular_update(
    const double dt_s, const double* const quaternion_xyzw,
    const double* const gravity, const double* const angular_start,
    const double* const angular_end, double* const output_quaternion_xyzw,
    double* const output_gravity) {
  if (quaternion_xyzw == nullptr || gravity == nullptr ||
      angular_start == nullptr || angular_end == nullptr ||
      output_quaternion_xyzw == nullptr || output_gravity == nullptr) {
    return 1;
  }
  const Eigen::Quaterniond orientation(
      quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1],
      quaternion_xyzw[2]);
  const Eigen::Vector3d angular_sum(
      angular_start[0] + angular_end[0],
      angular_start[1] + angular_end[1],
      angular_start[2] + angular_end[2]);
  // The installed tracker first scales the interpolated endpoint sum by 0.5,
  // then applies the interval duration.  Preserve that grouping: the
  // algebraically equivalent (0.5 * dt) * sum drifts in the last bits.
  const Eigen::Vector3d rotation_vector = (0.5 * angular_sum) * dt_s;
  const Eigen::Quaterniond delta =
      binaryRotationVectorQuaternion(rotation_vector);
  const Eigen::Quaterniond updated_orientation =
      binaryQuaternionProduct(orientation, delta);
  const Eigen::Quaterniond inverse_delta = binaryQuaternionInverse(delta);
  const Eigen::Vector3d updated_gravity = binaryQuaternionTransformVector(
      inverse_delta, Eigen::Vector3d(gravity[0], gravity[1], gravity[2]));
  output_quaternion_xyzw[0] = updated_orientation.x();
  output_quaternion_xyzw[1] = updated_orientation.y();
  output_quaternion_xyzw[2] = updated_orientation.z();
  output_quaternion_xyzw[3] = updated_orientation.w();
  output_gravity[0] = updated_gravity.x();
  output_gravity[1] = updated_gravity.y();
  output_gravity[2] = updated_gravity.z();
  return 0;
}

int navvis_recon_slam_rotation_from_two_vectors(
    const double* const first, const double* const second,
    double* const output_quaternion_xyzw) {
  if (first == nullptr || second == nullptr ||
      output_quaternion_xyzw == nullptr) {
    return 1;
  }
  const Eigen::Quaterniond rotation = binaryQuaternionFromTwoVectors(
      Eigen::Vector3d(first[0], first[1], first[2]),
      Eigen::Vector3d(second[0], second[1], second[2]));
  output_quaternion_xyzw[0] = rotation.x();
  output_quaternion_xyzw[1] = rotation.y();
  output_quaternion_xyzw[2] = rotation.z();
  output_quaternion_xyzw[3] = rotation.w();
  return 0;
}

int navvis_recon_slam_raw_imu_gravity_diagnostics(
    const double alpha, const double gravity_magnitude,
    const double* const quaternion_xyzw, const double* const gravity,
    const double* const acceleration, double* const output_expected_gravity,
    double* const output_correction_xyzw) {
  if (quaternion_xyzw == nullptr || gravity == nullptr ||
      acceleration == nullptr || output_expected_gravity == nullptr ||
      output_correction_xyzw == nullptr) {
    return 1;
  }
  static_cast<void>(gravity_magnitude);
  const Eigen::Quaterniond orientation(
      quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1],
      quaternion_xyzw[2]);
  const Eigen::Vector3d previous_gravity(gravity[0], gravity[1], gravity[2]);
  const Eigen::Vector3d measured_acceleration(
      acceleration[0], acceleration[1], acceleration[2]);
  const Eigen::Vector3d blended_world_gravity =
      (1.0 - alpha) * (orientation * previous_gravity) +
      alpha * (orientation * measured_acceleration);
  const Eigen::Quaterniond inverse_orientation = orientation.inverse();
  const Eigen::Vector3d updated_gravity =
      inverse_orientation * blended_world_gravity;
  const Eigen::Vector3d expected_gravity =
      inverse_orientation * Eigen::Vector3d::UnitZ();
  const Eigen::Quaterniond correction =
      binaryQuaternionFromTwoVectors(updated_gravity, expected_gravity);
  output_expected_gravity[0] = expected_gravity.x();
  output_expected_gravity[1] = expected_gravity.y();
  output_expected_gravity[2] = expected_gravity.z();
  output_correction_xyzw[0] = correction.x();
  output_correction_xyzw[1] = correction.y();
  output_correction_xyzw[2] = correction.z();
  output_correction_xyzw[3] = correction.w();
  return 0;
}

int navvis_recon_slam_raw_imu_gravity_update(
    const double alpha, const double gravity_magnitude,
    const double* const quaternion_xyzw, const double* const gravity,
    const double* const acceleration, double* const output_quaternion_xyzw,
    double* const output_gravity) {
  if (quaternion_xyzw == nullptr || gravity == nullptr ||
      acceleration == nullptr || output_quaternion_xyzw == nullptr ||
      output_gravity == nullptr) {
    return 1;
  }
  static_cast<void>(gravity_magnitude);

  Eigen::Quaterniond orientation(
      quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1],
      quaternion_xyzw[2]);
  const Eigen::Vector3d previous_gravity(gravity[0], gravity[1], gravity[2]);
  const Eigen::Vector3d measured_acceleration(
      acceleration[0], acceleration[1], acceleration[2]);

  const Eigen::Vector3d world_gravity = orientation * previous_gravity;
  const Eigen::Vector3d world_acceleration =
      orientation * measured_acceleration;
  const Eigen::Vector3d blended_world_gravity =
      (1.0 - alpha) * world_gravity + alpha * world_acceleration;
  const Eigen::Quaterniond inverse_orientation = orientation.inverse();
  const Eigen::Vector3d updated_gravity =
      inverse_orientation * blended_world_gravity;
  const Eigen::Vector3d expected_gravity =
      inverse_orientation * Eigen::Vector3d::UnitZ();
  const Eigen::Quaterniond gravity_correction =
      binaryQuaternionFromTwoVectors(updated_gravity, expected_gravity);
  orientation = binaryQuaternionNormalized(
      binaryQuaternionProduct(orientation, gravity_correction));

  output_quaternion_xyzw[0] = orientation.x();
  output_quaternion_xyzw[1] = orientation.y();
  output_quaternion_xyzw[2] = orientation.z();
  output_quaternion_xyzw[3] = orientation.w();
  output_gravity[0] = updated_gravity.x();
  output_gravity[1] = updated_gravity.y();
  output_gravity[2] = updated_gravity.z();
  return 0;
}

void* navvis_recon_slam_octree_create(const std::uint64_t point_count,
                                      const float* const points) {
  if (point_count == 0 || points == nullptr ||
      point_count > std::numeric_limits<std::uint32_t>::max()) {
    return nullptr;
  }
  try {
    return new BinaryOctree(point_count, points);
  } catch (...) {
    return nullptr;
  }
}

void navvis_recon_slam_octree_destroy(void* const opaque_octree) {
  delete static_cast<BinaryOctree*>(opaque_octree);
}

int navvis_recon_slam_octree_nearest(
    void* const opaque_octree, const std::uint64_t query_count,
    const float* const queries, const float search_radius,
    const std::int32_t num_threads, std::uint8_t* const found,
    std::uint64_t* const indices, float* const distances_squared) {
  if (opaque_octree == nullptr || !(search_radius > 0.F) ||
      !std::isfinite(search_radius) ||
      (query_count != 0 && (queries == nullptr || found == nullptr ||
                            indices == nullptr || distances_squared == nullptr))) {
    return 1;
  }
  const auto* const octree = static_cast<const BinaryOctree*>(opaque_octree);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads > 0 ? num_threads : 1)
#endif
  for (std::int64_t query_index = 0;
       query_index < static_cast<std::int64_t>(query_count); ++query_index) {
    found[query_index] = octree->nearest(
        queries + 3 * query_index, search_radius, indices + query_index,
        distances_squared + query_index);
    if (found[query_index] == 0) {
      indices[query_index] = std::numeric_limits<std::uint64_t>::max();
      distances_squared[query_index] =
          std::numeric_limits<float>::infinity();
    }
  }
  return 0;
}

// Cartographer-compatible horizontal slice-direction histogram.  This is
// evaluated for every tenth retained node, so keeping the point walk in C++
// avoids a large Python scalar-loop cost without changing the downstream
// histogram representation.
int navvis_recon_slam_rotational_histogram(
    const std::uint64_t point_count, const float* const points,
    const std::int32_t histogram_size, float* const histogram) {
  if (histogram_size < 1 || histogram == nullptr ||
      (point_count != 0 && points == nullptr)) {
    return 1;
  }
  std::fill(histogram, histogram + histogram_size, 0.F);
  if (point_count == 0) {
    return 0;
  }

  const auto round_away_from_zero = [](const float value) -> std::int32_t {
    return static_cast<std::int32_t>(
        value >= 0.F ? std::floor(value + 0.5F)
                     : std::ceil(value - 0.5F));
  };
  std::unordered_map<std::int32_t, std::vector<std::uint64_t>> slices;
  slices.reserve(point_count / 32 + 1);
  std::vector<std::int32_t> slice_order;
  for (std::uint64_t index = 0; index < point_count; ++index) {
    const std::int32_t slice =
        round_away_from_zero(points[3 * index + 2] / 0.2F);
    const auto [found, inserted] = slices.try_emplace(slice);
    if (inserted) {
      slice_order.push_back(slice);
    }
    found->second.push_back(index);
  }
  std::sort(slice_order.begin(), slice_order.end());

  struct AngularPoint {
    float angle;
    std::uint64_t index;
  };
  constexpr float kMinimumRadius = 0.2F;
  constexpr float kMaximumStep = 0.9F;
  const float pi = static_cast<float>(std::acos(-1.0));
  for (const std::int32_t slice : slice_order) {
    const std::vector<std::uint64_t>& indices = slices.at(slice);
    float centroid[3] = {0.F, 0.F, 0.F};
    for (const std::uint64_t index : indices) {
      centroid[0] += points[3 * index];
      centroid[1] += points[3 * index + 1];
      centroid[2] += points[3 * index + 2];
    }
    const float reciprocal = 1.F / static_cast<float>(indices.size());
    centroid[0] *= reciprocal;
    centroid[1] *= reciprocal;
    centroid[2] *= reciprocal;

    std::vector<AngularPoint> ordered;
    ordered.reserve(indices.size());
    for (const std::uint64_t index : indices) {
      const float dx = points[3 * index] - centroid[0];
      const float dy = points[3 * index + 1] - centroid[1];
      const float radius = std::sqrt(dy * dy + dx * dx);
      if (radius >= kMinimumRadius) {
        ordered.push_back({std::atan2(dy, dx), index});
      }
    }
    if (ordered.empty()) {
      continue;
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const AngularPoint& lhs, const AngularPoint& rhs) {
                return lhs.angle < rhs.angle;
              });

    std::uint64_t last_index = ordered.front().index;
    for (const AngularPoint& angular_point : ordered) {
      const std::uint64_t index = angular_point.index;
      const float delta_x = points[3 * index] - points[3 * last_index];
      const float delta_y =
          points[3 * index + 1] - points[3 * last_index + 1];
      const float direction_x = points[3 * index] - centroid[0];
      const float direction_y = points[3 * index + 1] - centroid[1];
      const float distance =
          std::sqrt(delta_y * delta_y + delta_x * delta_x);
      const float direction_norm = std::sqrt(
          direction_y * direction_y + direction_x * direction_x);
      if (distance < kMinimumRadius || direction_norm < kMinimumRadius) {
        continue;
      }
      if (distance > kMaximumStep) {
        last_index = index;
        continue;
      }
      const float normalized_delta_x = delta_x / distance;
      const float normalized_delta_y = delta_y / distance;
      const float normalized_direction_x = direction_x / direction_norm;
      const float normalized_direction_y = direction_y / direction_norm;
      const float alignment =
          normalized_delta_x * normalized_direction_x +
          normalized_delta_y * normalized_direction_y;
      const float weight = std::max(0.F, 1.F - std::fabs(alignment));
      float angle = std::atan2(delta_y, delta_x);
      while (angle > pi) {
        angle -= pi;
      }
      while (angle < 0.F) {
        angle += pi;
      }
      const float bucket_value =
          static_cast<float>(histogram_size) * (angle / pi) - 0.5F;
      const std::int32_t bucket = std::max(
          0, std::min(histogram_size - 1,
                      round_away_from_zero(bucket_value)));
      histogram[bucket] += weight;
    }
  }
  return 0;
}

// HASH_MAP_CENTROID range filter used immediately before scan matching and
// the high-resolution node-cloud filter. The installed implementation keeps
// the first-seen voxel order within each sign octant and performs the running
// centroid in float arithmetic over the full origin/endpoint measurement.
int navvis_recon_slam_range_centroid_filter(
    const std::uint64_t input_count, const float* const origins,
    const float* const points, const float resolution,
    const std::uint64_t output_capacity, std::uint64_t* const output_count,
    float* const output_origins, float* const output_points) {
  if (!(resolution > 0.F) || !std::isfinite(resolution) ||
      output_count == nullptr ||
      (input_count != 0 && (origins == nullptr || points == nullptr)) ||
      (output_capacity != 0 &&
       (output_origins == nullptr || output_points == nullptr))) {
    return 1;
  }

  std::unordered_map<std::uint64_t, std::size_t> labels;
  labels.reserve(input_count);
  std::vector<RangeCentroid> centroids;
  centroids.reserve(input_count);
  const double resolution_double = static_cast<double>(resolution);

  for (std::uint64_t input_index = 0; input_index < input_count;
       ++input_index) {
    const float* const origin = origins + 3 * input_index;
    const float* const point = points + 3 * input_index;
    std::int32_t cell[3];
    for (int axis = 0; axis < 3; ++axis) {
      const double coordinate = static_cast<double>(point[axis]);
      const double cell_double = std::floor(coordinate / resolution_double);
      if (cell_double < static_cast<double>(kMinimumCell) ||
          cell_double > static_cast<double>(kMaximumCell)) {
        return 2;
      }
      cell[axis] = static_cast<std::int32_t>(cell_double);
    }
    std::uint64_t key = 0;
    if (!pack_cell(cell[0], cell[1], cell[2], &key)) {
      return 2;
    }

    const auto [found, inserted] = labels.emplace(key, centroids.size());
    if (inserted) {
      RangeCentroid centroid;
      centroid.key = key;
      centroid.count = 1.F;
      centroid.octant =
          static_cast<std::uint8_t>((point[0] >= 0.F ? 1 : 0) |
                                    (point[1] >= 0.F ? 2 : 0) |
                                    (point[2] >= 0.F ? 4 : 0));
      for (int axis = 0; axis < 3; ++axis) {
        centroid.origin[axis] = origin[axis];
        centroid.point[axis] = point[axis];
      }
      centroids.push_back(centroid);
      continue;
    }

    RangeCentroid& centroid = centroids[found->second];
    const float old_count = centroid.count;
    const float new_count = old_count + 1.F;
    const float reciprocal = 1.F / new_count;
    for (int axis = 0; axis < 3; ++axis) {
      centroid.origin[axis] =
          (centroid.origin[axis] * old_count + origin[axis]) * reciprocal;
      centroid.point[axis] =
          (centroid.point[axis] * old_count + point[axis]) * reciprocal;
    }
    centroid.count = new_count;
  }

  if (centroids.size() > output_capacity) {
    *output_count = centroids.size();
    return 3;
  }
  std::uint64_t output_index = 0;
  for (std::uint8_t octant = 0; octant < 8; ++octant) {
    for (const RangeCentroid& centroid : centroids) {
      if (centroid.octant != octant) {
        continue;
      }
      for (int axis = 0; axis < 3; ++axis) {
        output_origins[3 * output_index + axis] = centroid.origin[axis];
        output_points[3 * output_index + axis] = centroid.point[axis];
      }
      ++output_index;
    }
  }
  *output_count = output_index;
  return 0;
}

// Assign split-surfel cells while preserving the dense vector's first
// insertion order.  Previous labels remain stable and unseen cells are
// appended in raw point order.  This replaces repeated whole-state sorting in
// the Python coordinator without changing any downstream accumulation order.
int navvis_recon_slam_label_surfel_cells(
    const std::uint64_t previous_count, const std::int64_t* const previous_keys,
    const std::uint64_t point_count, const float* const points,
    const double grid_origin_x, const double grid_origin_y,
    const double grid_origin_z, const double inverse_resolution,
    const std::uint64_t output_capacity, std::uint64_t* const output_count,
    std::int64_t* const output_keys, std::uint64_t* const point_labels) {
  if (!(inverse_resolution > 0.0) || !std::isfinite(inverse_resolution) ||
      output_count == nullptr || output_capacity < previous_count ||
      (previous_count != 0 && previous_keys == nullptr) ||
      (output_capacity != 0 && output_keys == nullptr) ||
      (point_count != 0 && (points == nullptr || point_labels == nullptr))) {
    return 1;
  }
  const double origins[3] = {grid_origin_x, grid_origin_y, grid_origin_z};
  std::unordered_map<std::uint64_t, std::uint64_t> labels;
  labels.reserve(previous_count + point_count);
  for (std::uint64_t index = 0; index < previous_count; ++index) {
    const std::int64_t x = previous_keys[3 * index];
    const std::int64_t y = previous_keys[3 * index + 1];
    const std::int64_t z = previous_keys[3 * index + 2];
    if (x < kMinimumCell || x > kMaximumCell || y < kMinimumCell ||
        y > kMaximumCell || z < kMinimumCell || z > kMaximumCell) {
      return 2;
    }
    std::uint64_t packed = 0;
    if (!pack_cell(static_cast<std::int32_t>(x), static_cast<std::int32_t>(y),
                   static_cast<std::int32_t>(z), &packed) ||
        !labels.emplace(packed, index).second) {
      return 2;
    }
    output_keys[3 * index] = x;
    output_keys[3 * index + 1] = y;
    output_keys[3 * index + 2] = z;
  }

  std::uint64_t size = previous_count;
  for (std::uint64_t point_index = 0; point_index < point_count;
       ++point_index) {
    std::int64_t cell[3];
    for (int axis = 0; axis < 3; ++axis) {
      const double coordinate =
          (static_cast<double>(points[3 * point_index + axis]) + origins[axis]) *
          inverse_resolution;
      const double floored = std::floor(coordinate);
      if (floored < static_cast<double>(kMinimumCell) ||
          floored > static_cast<double>(kMaximumCell)) {
        return 3;
      }
      cell[axis] = static_cast<std::int64_t>(floored);
    }
    std::uint64_t packed = 0;
    if (!pack_cell(static_cast<std::int32_t>(cell[0]),
                   static_cast<std::int32_t>(cell[1]),
                   static_cast<std::int32_t>(cell[2]), &packed)) {
      return 3;
    }
    const auto found = labels.find(packed);
    if (found != labels.end()) {
      point_labels[point_index] = found->second;
      continue;
    }
    if (size >= output_capacity) {
      *output_count = size + 1;
      return 4;
    }
    labels.emplace(packed, size);
    output_keys[3 * size] = cell[0];
    output_keys[3 * size + 1] = cell[1];
    output_keys[3 * size + 2] = cell[2];
    point_labels[point_index] = size;
    ++size;
  }
  *output_count = size;
  return 0;
}

void* navvis_recon_slam_probability_grid_create(const float resolution) {
  if (!(resolution > 0.F) || !std::isfinite(resolution)) {
    return nullptr;
  }
  try {
    return new ProbabilityGrid(resolution);
  } catch (...) {
    return nullptr;
  }
}

void navvis_recon_slam_probability_grid_destroy(void* const opaque_grid) {
  delete static_cast<ProbabilityGrid*>(opaque_grid);
}

std::uint64_t navvis_recon_slam_probability_grid_size(
    const void* const opaque_grid) {
  const auto* const grid = static_cast<const ProbabilityGrid*>(opaque_grid);
  return grid == nullptr ? 0 : grid->cells.size();
}

int navvis_recon_slam_probability_grid_export(
    const void* const opaque_grid, const std::uint64_t capacity,
    std::uint64_t* const output_count, std::int32_t* const indices,
    std::uint16_t* const values) {
  const auto* const grid = static_cast<const ProbabilityGrid*>(opaque_grid);
  if (grid == nullptr || output_count == nullptr) {
    return 1;
  }
  *output_count = grid->cells.size();
  if (capacity < grid->cells.size()) {
    return 2;
  }
  if (!grid->cells.empty() && (indices == nullptr || values == nullptr)) {
    return 3;
  }

  const auto unpack_axis = [](const std::uint64_t key,
                              const unsigned shift) -> std::int32_t {
    std::uint32_t value =
        static_cast<std::uint32_t>((key >> shift) & kCellMask);
    if ((value & (1U << 20)) != 0U) {
      value |= ~static_cast<std::uint32_t>(kCellMask);
    }
    return static_cast<std::int32_t>(value);
  };
  std::uint64_t index = 0;
  for (const auto& [key, value] : grid->cells) {
    indices[3 * index] = unpack_axis(key, 42);
    indices[3 * index + 1] = unpack_axis(key, 21);
    indices[3 * index + 2] = unpack_axis(key, 0);
    values[index] = value;
    ++index;
  }
  return 0;
}

int navvis_recon_slam_probability_grid_load(
    void* const opaque_grid, const std::uint64_t cell_count,
    const std::int32_t* const indices, const std::uint16_t* const values) {
  auto* const grid = static_cast<ProbabilityGrid*>(opaque_grid);
  if (grid == nullptr ||
      (cell_count != 0 && (indices == nullptr || values == nullptr))) {
    return 1;
  }
  grid->cells.clear();
  grid->cells.reserve(cell_count);
  for (std::uint64_t index = 0; index < cell_count; ++index) {
    if (values[index] >= kUpdateMarker) {
      return 2;
    }
    std::uint64_t key = 0;
    if (!pack_cell(indices[3 * index], indices[3 * index + 1],
                   indices[3 * index + 2], &key)) {
      return 3;
    }
    grid->cells[key] = values[index];
  }
  return 0;
}

// RangeDataInserter3D-compatible update. Hits are applied before misses and
// every cell is updated at most once in this batch. Miss rays contain their
// origin and exclude the endpoint; only the final two samples are retained.
int navvis_recon_slam_probability_grid_insert(
    void* const opaque_grid, const std::uint64_t point_count,
    const float* const points, const float* const origins) {
  auto* const grid = static_cast<ProbabilityGrid*>(opaque_grid);
  if (grid == nullptr ||
      (point_count != 0 && (points == nullptr || origins == nullptr))) {
    return 1;
  }
  std::unordered_set<std::uint64_t> hit_cells;
  hit_cells.reserve(point_count);
  for (std::uint64_t index = 0; index < point_count; ++index) {
    std::int32_t cell[3];
    std::uint64_t key = 0;
    if (!point_cell(points + 3 * index, grid->resolution, cell) ||
        !pack_cell(cell[0], cell[1], cell[2], &key)) {
      return 2;
    }
    hit_cells.insert(key);
  }
  for (const std::uint64_t key : hit_cells) {
    apply_table(grid, key, grid->hit_table);
  }

  std::unordered_set<std::uint64_t> miss_cells;
  miss_cells.reserve(std::min<std::uint64_t>(2 * point_count, 1ULL << 22));
  for (std::uint64_t index = 0; index < point_count; ++index) {
    std::int32_t origin_cell[3];
    std::int32_t hit_cell[3];
    if (!point_cell(origins + 3 * index, grid->resolution, origin_cell) ||
        !point_cell(points + 3 * index, grid->resolution, hit_cell)) {
      return 3;
    }
    const std::int32_t delta[3] = {
        hit_cell[0] - origin_cell[0], hit_cell[1] - origin_cell[1],
        hit_cell[2] - origin_cell[2]};
    const std::int32_t num_samples =
        std::max({std::abs(delta[0]), std::abs(delta[1]),
                  std::abs(delta[2])});
    if (num_samples == 0 || num_samples >= (1 << 15)) {
      continue;
    }
    const std::int32_t start = std::max(0, num_samples - 2);
    for (std::int32_t position = start; position < num_samples; ++position) {
      const std::int32_t cell[3] = {
          origin_cell[0] + delta[0] * position / num_samples,
          origin_cell[1] + delta[1] * position / num_samples,
          origin_cell[2] + delta[2] * position / num_samples};
      std::uint64_t key = 0;
      if (!pack_cell(cell[0], cell[1], cell[2], &key)) {
        return 4;
      }
      if (hit_cells.find(key) == hit_cells.end()) {
        miss_cells.insert(key);
      }
    }
  }
  for (const std::uint64_t key : miss_cells) {
    apply_table(grid, key, grid->miss_table);
  }
  return 0;
}

float navvis_recon_slam_probability_grid_score(
    const void* const opaque_grid, const std::uint64_t point_count,
    const float* const points) {
  const auto* const grid = static_cast<const ProbabilityGrid*>(opaque_grid);
  if (grid == nullptr || point_count == 0 || points == nullptr) {
    return 0.F;
  }
  std::uint64_t sum = 0;
  for (std::uint64_t index = 0; index < point_count; ++index) {
    std::int32_t cell[3];
    std::uint64_t key = 0;
    if (!point_cell(points + 3 * index, grid->resolution, cell) ||
        !pack_cell(cell[0], cell[1], cell[2], &key)) {
      continue;
    }
    const auto found = grid->cells.find(key);
    const std::uint16_t value =
        found == grid->cells.end() ? 0 : found->second;
    const int quantized = static_cast<int>(std::lround(
        (value_to_probability(value) - kMinProbability) *
        (255.F / (kMaxProbability - kMinProbability))));
    sum += static_cast<std::uint64_t>(std::max(0, std::min(255, quantized)));
  }
  return kMinProbability +
         (static_cast<float>(sum) / static_cast<float>(point_count)) *
             ((kMaxProbability - kMinProbability) / 255.F);
}

// Update unit-weight surfels in original point order.  This mirrors the
// BaseSurfel<float>::add path recovered from the installed G11 binary:
// weighted mean followed by the population-covariance Welford update.
int navvis_recon_slam_update_surfels(
    std::uint64_t state_count, float* weights, std::uint32_t* counts,
    float* means, float* covariances, std::uint64_t point_count,
    const std::uint64_t* labels, const float* points) {
  if ((state_count != 0 &&
       (weights == nullptr || counts == nullptr || means == nullptr ||
        covariances == nullptr)) ||
      (point_count != 0 && (labels == nullptr || points == nullptr))) {
    return 1;
  }

  for (std::uint64_t point_index = 0; point_index < point_count;
       ++point_index) {
    const std::uint64_t label = labels[point_index];
    if (!valid_label(label, state_count)) {
      return 2;
    }
    float* const mean = means + 3 * label;
    float* const covariance = covariances + 9 * label;
    const float* const point = points + 3 * point_index;
    float unused_viewpoint_mean[3] = {0.0F, 0.0F, 0.0F};
    add_unit_weight_surfel(weights + label, counts + label, mean, covariance,
                           unused_viewpoint_mean, point, point);
  }
  return 0;
}

// Update the two BaseSurfel<float> instances stored in each 232-byte
// SurfelGrid voxel.  A voxel starts with only its primary side.  Maintenance
// freezes the primary PCA normal once the split predicate first succeeds;
// later rays are routed by the sign of (origin - endpoint) dot split_normal.
int navvis_recon_slam_update_split_surfels(
    const std::uint64_t state_count, float* const primary_weights,
    std::uint32_t* const primary_counts, float* const primary_means,
    float* const primary_covariances, float* const primary_viewpoints,
    float* const secondary_weights, std::uint32_t* const secondary_counts,
    float* const secondary_means, float* const secondary_covariances,
    float* const secondary_viewpoints, std::uint8_t* const is_split,
    float* const split_normals, std::uint8_t* const primary_dirty,
    std::uint8_t* const secondary_dirty, const std::uint64_t point_count,
    const std::uint64_t* const labels, const float* const origins,
    const float* const points, const std::uint8_t maintain_surfels) {
  if ((state_count != 0 &&
       (primary_weights == nullptr || primary_counts == nullptr ||
        primary_means == nullptr || primary_covariances == nullptr ||
        primary_viewpoints == nullptr ||
        secondary_weights == nullptr || secondary_counts == nullptr ||
        secondary_means == nullptr || secondary_covariances == nullptr ||
        secondary_viewpoints == nullptr || is_split == nullptr ||
        split_normals == nullptr || primary_dirty == nullptr ||
        secondary_dirty == nullptr)) ||
      (point_count != 0 &&
       (labels == nullptr || origins == nullptr || points == nullptr))) {
    return 1;
  }

  std::vector<std::uint8_t> touched(state_count, 0);
  for (std::uint64_t point_index = 0; point_index < point_count;
       ++point_index) {
    const std::uint64_t label = labels[point_index];
    if (!valid_label(label, state_count)) {
      return 2;
    }
    const float* const point = points + 3 * point_index;
    bool use_secondary = false;
    if (is_split[label] != 0) {
      const float* const origin = origins + 3 * point_index;
      const float* const normal = split_normals + 3 * label;
      // Preserve the binary's scalar instruction order: z, then y, then x.
      float side = (origin[2] - point[2]) * normal[2];
      side += (origin[1] - point[1]) * normal[1];
      side += (origin[0] - point[0]) * normal[0];
      use_secondary = side < 0.0F;
    }
    float* const weights =
        use_secondary ? secondary_weights : primary_weights;
    std::uint32_t* const counts =
        use_secondary ? secondary_counts : primary_counts;
    float* const means = use_secondary ? secondary_means : primary_means;
    float* const covariances =
        use_secondary ? secondary_covariances : primary_covariances;
    float* const viewpoints =
        use_secondary ? secondary_viewpoints : primary_viewpoints;
    const float* const origin = origins + 3 * point_index;
    add_unit_weight_surfel(weights + label, counts + label, means + 3 * label,
                           covariances + 9 * label, viewpoints + 3 * label,
                           origin, point);
    (use_secondary ? secondary_dirty : primary_dirty)[label] = 1;
    touched[label] = 1;
  }

  if (maintain_surfels == 0) {
    return 0;
  }

  // The installed worker recomputes dirty PCA fields before testing whether
  // an unsplit voxel should become split.  It visits only cells touched by
  // this ray batch; dirty state created by a merge remains pending until that
  // cell is touched again.  Existing split normals are frozen.
  for (std::uint64_t index = 0; index < state_count; ++index) {
    if (touched[index] == 0) {
      continue;
    }
    Eigen::Vector3f primary_normal;
    Eigen::Vector3f primary_eigenvalues;
    bool have_primary_geometry = false;
    if (primary_dirty[index] != 0 && primary_counts[index] > 2) {
      if (!surfel_geometry(primary_covariances + 9 * index, &primary_normal,
                           &primary_eigenvalues)) {
        return 3;
      }
      orient_surfel_normal(primary_means + 3 * index,
                           primary_viewpoints + 3 * index, &primary_normal);
      have_primary_geometry = true;
      primary_dirty[index] = 0;
    }
    if (secondary_dirty[index] != 0 && secondary_counts[index] > 2) {
      Eigen::Vector3f secondary_normal;
      Eigen::Vector3f secondary_eigenvalues;
      if (!surfel_geometry(secondary_covariances + 9 * index,
                           &secondary_normal, &secondary_eigenvalues)) {
        return 4;
      }
      orient_surfel_normal(secondary_means + 3 * index,
                           secondary_viewpoints + 3 * index,
                           &secondary_normal);
      secondary_dirty[index] = 0;
    }
    if (is_split[index] == 0 && have_primary_geometry &&
        valid_split_surfel(primary_weights[index], primary_eigenvalues)) {
      is_split[index] = 1;
      for (int axis = 0; axis < 3; ++axis) {
        split_normals[3 * index + axis] = primary_normal[axis];
      }
    }
  }
  return 0;
}

// Perform the deferred worker maintenance used when an overlapping insertion
// submap becomes the matching submap.  All non-empty cells have accumulated
// raw BaseSurfel moments already; the binary visits that complete insertion-
// ordered state once before its first cross-cell merge pass.
int navvis_recon_slam_maintain_split_surfels(
    const std::uint64_t state_count, const float* const primary_weights,
    const std::uint32_t* const primary_counts, const float* const primary_means,
    const float* const primary_covariances,
    const float* const primary_viewpoints, const std::uint32_t* const secondary_counts,
    const float* const secondary_means, const float* const secondary_covariances,
    const float* const secondary_viewpoints, std::uint8_t* const is_split,
    float* const split_normals, std::uint8_t* const primary_dirty,
    std::uint8_t* const secondary_dirty) {
  if (state_count != 0 &&
      (primary_weights == nullptr || primary_counts == nullptr ||
       primary_means == nullptr || primary_covariances == nullptr ||
       primary_viewpoints == nullptr || secondary_counts == nullptr ||
       secondary_means == nullptr || secondary_covariances == nullptr ||
       secondary_viewpoints == nullptr || is_split == nullptr ||
       split_normals == nullptr || primary_dirty == nullptr ||
       secondary_dirty == nullptr)) {
    return 1;
  }
  for (std::uint64_t index = 0; index < state_count; ++index) {
    Eigen::Vector3f primary_normal;
    Eigen::Vector3f primary_eigenvalues;
    bool have_primary_geometry = false;
    if (primary_dirty[index] != 0 && primary_counts[index] > 2) {
      if (!surfel_geometry(primary_covariances + 9 * index, &primary_normal,
                           &primary_eigenvalues)) {
        return 2;
      }
      orient_surfel_normal(primary_means + 3 * index,
                           primary_viewpoints + 3 * index, &primary_normal);
      have_primary_geometry = true;
      primary_dirty[index] = 0;
    }
    if (secondary_dirty[index] != 0 && secondary_counts[index] > 2) {
      Eigen::Vector3f secondary_normal;
      Eigen::Vector3f secondary_eigenvalues;
      if (!surfel_geometry(secondary_covariances + 9 * index,
                           &secondary_normal, &secondary_eigenvalues)) {
        return 3;
      }
      orient_surfel_normal(secondary_means + 3 * index,
                           secondary_viewpoints + 3 * index,
                           &secondary_normal);
      secondary_dirty[index] = 0;
    }
    if (is_split[index] == 0 && have_primary_geometry &&
        valid_split_surfel(primary_weights[index], primary_eigenvalues)) {
      is_split[index] = 1;
      for (int axis = 0; axis < 3; ++axis) {
        split_normals[3 * index + axis] = primary_normal[axis];
      }
    }
  }
  return 0;
}

// Merge compatible surfels across the face selected by their dominant normal.
// The installed 0.1 m grid freezes directed candidates by signed normal score,
// then applies them in caller-provided source order.  Initial construction
// visits every cell in insertion order; incremental updates visit only cells
// touched by the current ray batch, in first-occurrence order.  A merged
// unsplit primary is immediately eligible to freeze its PCA normal and become
// split.  Empty losers retain their old geometry; their live weights/counts
// and split flag are cleared.
int navvis_recon_slam_merge_surfels(
    const std::uint64_t state_count, const std::int64_t* const keys,
    float* const primary_weights, std::uint32_t* const primary_counts,
    float* const primary_means, float* const primary_covariances,
    float* const primary_viewpoints, float* const secondary_weights,
    std::uint32_t* const secondary_counts, float* const secondary_means,
    float* const secondary_covariances, float* const secondary_viewpoints,
    std::uint8_t* const is_split, float* const split_normals,
    std::uint8_t* const primary_dirty,
    std::uint8_t* const secondary_dirty,
    const double grid_origin_x, const double grid_origin_y,
    const double grid_origin_z, const double inverse_resolution,
    const std::uint64_t source_count,
    const std::uint64_t* const source_indices,
    std::uint64_t* const applied_merge_count) {
  if (applied_merge_count != nullptr) {
    *applied_merge_count = 0;
  }
  if (state_count == 0) {
    return 0;
  }
  if (keys == nullptr || primary_weights == nullptr ||
      primary_counts == nullptr || primary_means == nullptr ||
      primary_covariances == nullptr || primary_viewpoints == nullptr ||
      secondary_weights == nullptr || secondary_counts == nullptr ||
      secondary_means == nullptr || secondary_covariances == nullptr ||
      secondary_viewpoints == nullptr || is_split == nullptr ||
      split_normals == nullptr || primary_dirty == nullptr ||
      secondary_dirty == nullptr ||
      (source_count != 0 && source_indices == nullptr)) {
    return 1;
  }
  if (!std::isfinite(grid_origin_x) || !std::isfinite(grid_origin_y) ||
      !std::isfinite(grid_origin_z) || !std::isfinite(inverse_resolution) ||
      inverse_resolution <= 0.0) {
    return 2;
  }

  std::vector<Eigen::Vector3f> primary_normals(state_count);
  std::vector<Eigen::Vector3f> primary_eigenvalues(state_count);
  // Eigen's fixed-size default constructor leaves coefficients untouched.
  // A split pair may legitimately have an empty secondary side; the binary
  // stores a zero normal for that side, so initialize it explicitly before
  // evaluating all four side-pair dot products.
  std::vector<Eigen::Vector3f> secondary_normals(
      state_count, Eigen::Vector3f::Zero());
  std::vector<Eigen::Vector3f> secondary_eigenvalues(
      state_count, Eigen::Vector3f::Zero());
  std::vector<bool> primary_valid(state_count, false);
  std::vector<bool> secondary_valid(state_count, false);
  std::unordered_map<std::uint64_t, std::uint64_t> indices;
  indices.reserve(state_count * 2);
  for (std::uint64_t index = 0; index < state_count; ++index) {
    if (!surfel_geometry(primary_covariances + 9 * index,
                         &primary_normals[index],
                         &primary_eigenvalues[index])) {
      return 3;
    }
    orient_surfel_normal(primary_means + 3 * index,
                         primary_viewpoints + 3 * index,
                         &primary_normals[index]);
    primary_valid[index] = valid_merge_surfel(
        primary_weights[index], primary_eigenvalues[index]);
    // Clearing one side of a SurfelPair resets only its live weight/count and
    // dirty flag.  The installed pair keeps the old PCA fields and continues
    // to use that normal when deciding same-side versus cross-side ordering.
    // Our compact state does not store the cached normal separately, but the
    // retained covariance reproduces it bit-for-bit.  A secondary side that
    // has never held geometry has an all-zero covariance and keeps the zero
    // normal initialized above.
    bool secondary_has_retained_geometry = secondary_counts[index] > 2;
    if (!secondary_has_retained_geometry) {
      for (int coefficient = 0; coefficient < 9; ++coefficient) {
        if (secondary_covariances[9 * index + coefficient] != 0.0F) {
          secondary_has_retained_geometry = true;
          break;
        }
      }
    }
    if (is_split[index] != 0 && secondary_has_retained_geometry) {
      if (!surfel_geometry(secondary_covariances + 9 * index,
                           &secondary_normals[index],
                           &secondary_eigenvalues[index])) {
        return 4;
      }
      orient_surfel_normal(secondary_means + 3 * index,
                           secondary_viewpoints + 3 * index,
                           &secondary_normals[index]);
      secondary_valid[index] = valid_merge_surfel(
          secondary_weights[index], secondary_eigenvalues[index]);
    }
    std::uint64_t packed = 0;
    if (!pack_cell(static_cast<std::int32_t>(keys[3 * index]),
                   static_cast<std::int32_t>(keys[3 * index + 1]),
                   static_cast<std::int32_t>(keys[3 * index + 2]), &packed)) {
      return 5;
    }
    indices.emplace(packed, index);
  }

  struct Candidate {
    std::uint64_t first;
    std::uint64_t second;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(source_count / 32);
  constexpr float kMaximumDistanceSquared = 0.004900000041723251F;
  constexpr float kMinimumNormalCosine = 0.9396926164627075F;
  constexpr float kMinimumPrimaryWeight = 2.5F;
  constexpr float kMissingPairScore = -1.0F;
  constexpr double kMissingNormalDot =
      -std::numeric_limits<double>::max();

  const auto normal_dot = [](const Eigen::Vector3f& first,
                             const Eigen::Vector3f& second) {
    // Preserve the installed scalar reduction order: z, then y, then x.
    return static_cast<double>(
        (first[2] * second[2] + first[1] * second[1]) +
        first[0] * second[0]);
  };

  const auto directed_pair_score = [&](const std::uint64_t source,
                                       const std::uint64_t neighbor) {
    const bool source_split = is_split[source] != 0;
    const bool neighbor_split = is_split[neighbor] != 0;
    const bool primary_primary_valid =
        primary_valid[source] && primary_valid[neighbor];
    const bool secondary_primary_valid =
        source_split && secondary_valid[source] && primary_valid[neighbor];
    const bool primary_secondary_valid =
        neighbor_split && primary_valid[source] && secondary_valid[neighbor];
    const bool secondary_secondary_valid =
        source_split && neighbor_split && secondary_valid[source] &&
        secondary_valid[neighbor];

    const double primary_primary =
        normal_dot(primary_normals[source], primary_normals[neighbor]);
    const double secondary_primary =
        source_split
            ? normal_dot(secondary_normals[source], primary_normals[neighbor])
            : kMissingNormalDot;
    const double primary_secondary =
        neighbor_split
            ? normal_dot(primary_normals[source], secondary_normals[neighbor])
            : kMissingNormalDot;
    const double secondary_secondary =
        source_split && neighbor_split
            ? normal_dot(secondary_normals[source],
                         secondary_normals[neighbor])
            : kMissingNormalDot;

    // SurfelPair::isMergeable first decides whether the two split pairs have
    // the same or opposite side ordering.  If both pairings in that ordering
    // are valid, the installed implementation selects the pairing with the
    // larger minimum side weight.  This is observable when two opposing wall
    // layers share adjacent cells: a 0.96 secondary-secondary match with
    // support 48 loses to a -0.93 primary-primary match with support 49, so
    // the complete pair is rejected.
    const bool same_side_order =
        (primary_primary > primary_secondary &&
         primary_primary > secondary_primary) ||
        (secondary_secondary > primary_secondary &&
         secondary_secondary > secondary_primary);
    if (same_side_order) {
      if (primary_primary_valid && secondary_secondary_valid) {
        const float primary_support =
            std::min(primary_weights[source], primary_weights[neighbor]);
        const float secondary_support =
            std::min(secondary_weights[source], secondary_weights[neighbor]);
        return secondary_support >= primary_support
                   ? static_cast<float>(secondary_secondary)
                   : static_cast<float>(primary_primary);
      }
      if (primary_primary_valid) {
        return static_cast<float>(primary_primary);
      }
      if (secondary_secondary_valid) {
        return static_cast<float>(secondary_secondary);
      }
      return kMissingPairScore;
    }
    if (primary_secondary_valid && secondary_primary_valid) {
      const float primary_secondary_support =
          std::min(primary_weights[source], secondary_weights[neighbor]);
      const float secondary_primary_support =
          std::min(secondary_weights[source], primary_weights[neighbor]);
      return primary_secondary_support >= secondary_primary_support
                 ? static_cast<float>(primary_secondary)
                 : static_cast<float>(secondary_primary);
    }
    if (primary_secondary_valid) {
      return static_cast<float>(primary_secondary);
    }
    if (secondary_primary_valid) {
      return static_cast<float>(secondary_primary);
    }
    return kMissingPairScore;
  };

  for (std::uint64_t source_position = 0; source_position < source_count;
       ++source_position) {
    const std::uint64_t index = source_indices[source_position];
    if (index >= state_count) {
      return 6;
    }
    if (primary_dirty[index] != 0 ||
        (is_split[index] != 0 && secondary_dirty[index] != 0)) {
      continue;
    }
    const bool source_uses_secondary =
        is_split[index] != 0 &&
        secondary_weights[index] > primary_weights[index];
    const Eigen::Vector3f& source_normal =
        source_uses_secondary ? secondary_normals[index]
                              : primary_normals[index];
    const float* const source_mean =
        (source_uses_secondary ? secondary_means : primary_means) + 3 * index;
    int axis = 0;
    float largest_component = std::fabs(source_normal[0]);
    for (int candidate_axis = 1; candidate_axis < 3; ++candidate_axis) {
      const float component = std::fabs(source_normal[candidate_axis]);
      if (component > largest_component) {
        axis = candidate_axis;
        largest_component = component;
      }
    }

    bool found = false;
    std::uint64_t best_neighbor = 0;
    float best_score = -std::numeric_limits<float>::max();
    // Candidate faces are built around the current dominant surfel center,
    // not around the cell key that originally owns the SurfelPair.  A center
    // can drift across a cell boundary after accumulation.  In that case one
    // of the two face lookups can resolve to the source pair itself; the
    // installed merge pass deliberately preserves that candidate and its
    // normal write-then-clear aliasing semantics.  Positive direction is
    // visited first in the frozen candidate search.
    std::int32_t center_key[3];
    for (int center_axis = 0; center_axis < 3; ++center_axis) {
      const double origin = center_axis == 0 ? grid_origin_x
                            : center_axis == 1 ? grid_origin_y
                                               : grid_origin_z;
      center_key[center_axis] = static_cast<std::int32_t>(std::floor(
          (static_cast<double>(source_mean[center_axis]) + origin) *
          inverse_resolution));
    }
    for (const std::int64_t direction : {1, -1}) {
      std::int32_t neighbor_key[3] = {
          center_key[0], center_key[1], center_key[2]};
      neighbor_key[axis] += static_cast<std::int32_t>(direction);
      std::uint64_t packed = 0;
      if (!pack_cell(neighbor_key[0], neighbor_key[1], neighbor_key[2],
                     &packed)) {
        continue;
      }
      const auto entry = indices.find(packed);
      if (entry == indices.end()) {
        continue;
      }
      const std::uint64_t neighbor = entry->second;
      if (primary_dirty[neighbor] != 0 ||
          (is_split[neighbor] != 0 && secondary_dirty[neighbor] != 0)) {
        continue;
      }
      if (primary_weights[index] < kMinimumPrimaryWeight ||
          primary_weights[neighbor] < kMinimumPrimaryWeight) {
        continue;
      }
      const bool neighbor_uses_secondary =
          is_split[neighbor] != 0 &&
          secondary_weights[neighbor] > primary_weights[neighbor];
      const float* const neighbor_mean =
          (neighbor_uses_secondary ? secondary_means : primary_means) +
          3 * neighbor;
      const float delta_x = neighbor_mean[0] - source_mean[0];
      const float delta_y = neighbor_mean[1] - source_mean[1];
      const float delta_z = neighbor_mean[2] - source_mean[2];
      const float distance_squared =
          (delta_z * delta_z + delta_y * delta_y) + delta_x * delta_x;
      const float score = directed_pair_score(index, neighbor);
      if (distance_squared < kMaximumDistanceSquared &&
          score > best_score && score > kMinimumNormalCosine) {
        found = true;
        best_neighbor = neighbor;
        best_score = score;
      }
    }
    if (found) {
      candidates.push_back({index, best_neighbor});
    }
  }

  const double grid_origins[3] = {
      grid_origin_x, grid_origin_y, grid_origin_z};
  const auto pair_center_key = [&](const std::uint64_t index,
                                   std::int64_t* const output) {
    const bool use_secondary =
        is_split[index] != 0 &&
        secondary_weights[index] > primary_weights[index];
    const float* const center =
        (use_secondary ? secondary_means : primary_means) + 3 * index;
    for (int axis = 0; axis < 3; ++axis) {
      output[axis] = static_cast<std::int64_t>(std::floor(
          (static_cast<double>(center[axis]) + grid_origins[axis]) *
          inverse_resolution));
    }
  };

  std::uint64_t merges = 0;
  for (const Candidate& candidate : candidates) {
    const std::uint64_t first = candidate.first;
    const std::uint64_t second = candidate.second;
    if (primary_weights[first] == 0.0F) {
      // Candidate discovery is frozen before any merge is applied. If an
      // earlier candidate has emptied the directed first cell, the installed
      // SurfelPair::add constructs a new pair from that cell's retained
      // zero-weight primary and adds only the live second primary. A secondary
      // side that belonged to the live second pair is not copied into this
      // newly constructed pair. The canonicalized primary then chooses which
      // adjacent cell survives.
      if (primary_weights[second] != 0.0F) {
        float canonical_weight = primary_weights[first];
        std::uint32_t canonical_count = primary_counts[first];
        float canonical_mean[3];
        float canonical_covariance[9];
        float canonical_viewpoint[3];
        std::copy_n(primary_means + 3 * first, 3, canonical_mean);
        std::copy_n(primary_covariances + 9 * first, 9,
                    canonical_covariance);
        std::copy_n(primary_viewpoints + 3 * first, 3,
                    canonical_viewpoint);
        merge_base_surfel(
            &canonical_weight, &canonical_count, canonical_mean,
            canonical_covariance, canonical_viewpoint,
            primary_weights[second], primary_counts[second],
            primary_means + 3 * second,
            primary_covariances + 9 * second,
            primary_viewpoints + 3 * second);
        Eigen::Vector3f canonical_normal;
        Eigen::Vector3f canonical_eigenvalues;
        if (!surfel_geometry(canonical_covariance,
                             &canonical_normal, &canonical_eigenvalues)) {
          return 7;
        }
        orient_surfel_normal(canonical_mean, canonical_viewpoint,
                             &canonical_normal);
        const bool canonical_is_split = valid_split_surfel(
            canonical_weight, canonical_eigenvalues);

        std::int64_t canonical_key[3];
        for (int axis = 0; axis < 3; ++axis) {
          const double origin = axis == 0 ? grid_origin_x
                                : axis == 1 ? grid_origin_y
                                            : grid_origin_z;
          canonical_key[axis] = static_cast<std::int64_t>(std::floor(
              (static_cast<double>(canonical_mean[axis]) + origin) *
              inverse_resolution));
        }
        const auto key_matches = [&](const std::uint64_t index) {
          std::int64_t current_key[3];
          pair_center_key(index, current_key);
          return current_key[0] == canonical_key[0] &&
                 current_key[1] == canonical_key[1] &&
                 current_key[2] == canonical_key[2];
        };
        std::uint64_t winner = first;
        std::uint64_t loser = second;
        if (!key_matches(first)) {
          if (!key_matches(second)) {
            continue;
          }
          winner = second;
          loser = first;
        }

        primary_weights[winner] = canonical_weight;
        primary_counts[winner] = canonical_count;
        std::copy_n(canonical_mean, 3, primary_means + 3 * winner);
        std::copy_n(canonical_covariance, 9,
                    primary_covariances + 9 * winner);
        std::copy_n(canonical_viewpoint, 3,
                    primary_viewpoints + 3 * winner);
        secondary_weights[winner] = secondary_weights[first];
        secondary_counts[winner] = secondary_counts[first];
        std::copy_n(secondary_means + 3 * first, 3,
                    secondary_means + 3 * winner);
        std::copy_n(secondary_covariances + 9 * first, 9,
                    secondary_covariances + 9 * winner);
        std::copy_n(secondary_viewpoints + 3 * first, 3,
                    secondary_viewpoints + 3 * winner);
        is_split[winner] = canonical_is_split ? 1 : 0;
        if (canonical_is_split) {
          for (int axis = 0; axis < 3; ++axis) {
            split_normals[3 * winner + axis] = canonical_normal[axis];
          }
        }
        primary_dirty[winner] = 0;
        secondary_dirty[winner] = 0;

        primary_weights[loser] = 0.0F;
        primary_counts[loser] = 0;
        secondary_weights[loser] = 0.0F;
        secondary_counts[loser] = 0;
        is_split[loser] = 0;
        primary_dirty[loser] = 0;
        secondary_dirty[loser] = 0;
        ++merges;
      }
      continue;
    }
    if (primary_weights[second] == 0.0F) {
      continue;
    }

    // The merge routine constructs its temporary from the directed first
    // voxel.  Consequently an existing first split normal stays frozen,
    // while an unsplit first voxel can acquire the merged PCA normal.
    float merged_weight = primary_weights[first];
    std::uint32_t merged_count = primary_counts[first];
    float merged_mean[3];
    float merged_covariance[9];
    float merged_viewpoint[3];
    std::copy_n(primary_means + 3 * first, 3, merged_mean);
    std::copy_n(primary_covariances + 9 * first, 9, merged_covariance);
    std::copy_n(primary_viewpoints + 3 * first, 3, merged_viewpoint);
    float merged_secondary_weight = secondary_weights[first];
    std::uint32_t merged_secondary_count = secondary_counts[first];
    float merged_secondary_mean[3];
    float merged_secondary_covariance[9];
    float merged_secondary_viewpoint[3];
    std::copy_n(secondary_means + 3 * first, 3, merged_secondary_mean);
    std::copy_n(secondary_covariances + 9 * first, 9,
                merged_secondary_covariance);
    std::copy_n(secondary_viewpoints + 3 * first, 3,
                merged_secondary_viewpoint);
    const bool second_was_split = is_split[second] != 0;
    const float first_secondary_weight = merged_secondary_weight;
    const float second_secondary_weight = secondary_weights[second];

    Eigen::Vector3f first_primary_normal;
    Eigen::Vector3f first_primary_eigenvalues;
    Eigen::Vector3f first_secondary_normal = Eigen::Vector3f::Zero();
    Eigen::Vector3f first_secondary_eigenvalues;
    Eigen::Vector3f second_primary_normal;
    Eigen::Vector3f second_primary_eigenvalues;
    Eigen::Vector3f second_secondary_normal = Eigen::Vector3f::Zero();
    Eigen::Vector3f second_secondary_eigenvalues;
    const auto has_retained_secondary_geometry = [=](
        const std::uint64_t index) {
      if (secondary_counts[index] > 2) {
        return true;
      }
      for (int coefficient = 0; coefficient < 9; ++coefficient) {
        if (secondary_covariances[9 * index + coefficient] != 0.0F) {
          return true;
        }
      }
      return false;
    };
    if (!surfel_geometry(primary_covariances + 9 * first,
                         &first_primary_normal,
                         &first_primary_eigenvalues) ||
        !surfel_geometry(primary_covariances + 9 * second,
                         &second_primary_normal,
                         &second_primary_eigenvalues)) {
      return 8;
    }
    orient_surfel_normal(primary_means + 3 * first,
                         primary_viewpoints + 3 * first,
                         &first_primary_normal);
    orient_surfel_normal(primary_means + 3 * second,
                         primary_viewpoints + 3 * second,
                         &second_primary_normal);
    // Clearing one side resets its live count but does not clear BaseSurfel's
    // cached PCA fields. SurfelPair::add still uses that retained normal to
    // decide same-side versus cross-side ordering. Reconstruct it from the
    // preserved covariance exactly as the frozen-candidate pass above does.
    if (is_split[first] != 0 && has_retained_secondary_geometry(first)) {
      if (!surfel_geometry(secondary_covariances + 9 * first,
                           &first_secondary_normal,
                           &first_secondary_eigenvalues)) {
        return 8;
      }
      orient_surfel_normal(secondary_means + 3 * first,
                           secondary_viewpoints + 3 * first,
                           &first_secondary_normal);
    }
    if (second_was_split && has_retained_secondary_geometry(second)) {
      if (!surfel_geometry(secondary_covariances + 9 * second,
                           &second_secondary_normal,
                           &second_secondary_eigenvalues)) {
        return 8;
      }
      orient_surfel_normal(secondary_means + 3 * second,
                           secondary_viewpoints + 3 * second,
                           &second_secondary_normal);
    }

    const double primary_primary =
        normal_dot(first_primary_normal, second_primary_normal);
    const double secondary_primary =
        is_split[first] != 0
            ? normal_dot(first_secondary_normal, second_primary_normal)
            : kMissingNormalDot;
    const double primary_secondary =
        second_was_split
            ? normal_dot(first_primary_normal, second_secondary_normal)
            : kMissingNormalDot;
    const double secondary_secondary =
        is_split[first] != 0 && second_was_split
            ? normal_dot(first_secondary_normal, second_secondary_normal)
            : kMissingNormalDot;
    const bool same_side_order =
        (primary_primary > primary_secondary &&
         primary_primary > secondary_primary) ||
        (secondary_secondary > primary_secondary &&
         secondary_secondary > secondary_primary);

    if (same_side_order) {
      merge_base_surfel(
          &merged_weight, &merged_count, merged_mean, merged_covariance,
          merged_viewpoint, primary_weights[second], primary_counts[second],
          primary_means + 3 * second, primary_covariances + 9 * second,
          primary_viewpoints + 3 * second);
      if (second_was_split) {
        merge_base_surfel(
            &merged_secondary_weight, &merged_secondary_count,
            merged_secondary_mean, merged_secondary_covariance,
            merged_secondary_viewpoint, second_secondary_weight,
            secondary_counts[second], secondary_means + 3 * second,
            secondary_covariances + 9 * second,
            secondary_viewpoints + 3 * second);
      }
    } else {
      merge_base_surfel(
          &merged_secondary_weight, &merged_secondary_count,
          merged_secondary_mean, merged_secondary_covariance,
          merged_secondary_viewpoint, primary_weights[second],
          primary_counts[second], primary_means + 3 * second,
          primary_covariances + 9 * second,
          primary_viewpoints + 3 * second);
      if (second_was_split) {
        merge_base_surfel(
            &merged_weight, &merged_count, merged_mean, merged_covariance,
            merged_viewpoint, second_secondary_weight,
            secondary_counts[second], secondary_means + 3 * second,
            secondary_covariances + 9 * second,
            secondary_viewpoints + 3 * second);
      }
    }

    Eigen::Vector3f merged_normal;
    Eigen::Vector3f merged_eigenvalues;
    if (!surfel_geometry(merged_covariance, &merged_normal,
                         &merged_eigenvalues)) {
      return 8;
    }
    orient_surfel_normal(merged_mean, merged_viewpoint, &merged_normal);
    const bool merged_is_split =
        is_split[first] != 0 ||
        valid_split_surfel(merged_weight, merged_eigenvalues);
    float merged_split_normal[3] = {merged_normal[0], merged_normal[1],
                                    merged_normal[2]};
    if (is_split[first] != 0) {
      std::copy_n(split_normals + 3 * first, 3, merged_split_normal);
    }

    // SurfelPair::getCenter returns the side with the larger accumulated
    // weight.  SurfelGrid uses that dominant center to choose which of the
    // two adjacent cells survives.
    const float* const dominant_merged_mean =
        merged_is_split && merged_secondary_weight > merged_weight
            ? merged_secondary_mean
            : merged_mean;
    std::int64_t merged_key[3];
    for (int axis = 0; axis < 3; ++axis) {
      merged_key[axis] = static_cast<std::int64_t>(
          std::floor((static_cast<double>(dominant_merged_mean[axis]) +
                      grid_origins[axis]) *
                     inverse_resolution));
    }
    auto key_matches = [&](const std::uint64_t index) {
      std::int64_t current_key[3];
      pair_center_key(index, current_key);
      return current_key[0] == merged_key[0] &&
             current_key[1] == merged_key[1] &&
             current_key[2] == merged_key[2];
    };
    std::uint64_t winner = first;
    std::uint64_t loser = second;
    if (!key_matches(first)) {
      if (!key_matches(second)) {
        continue;
      }
      winner = second;
      loser = first;
    }

    primary_weights[winner] = merged_weight;
    primary_counts[winner] = merged_count;
    std::copy_n(merged_mean, 3, primary_means + 3 * winner);
    std::copy_n(merged_covariance, 9, primary_covariances + 9 * winner);
    std::copy_n(merged_viewpoint, 3, primary_viewpoints + 3 * winner);
    secondary_weights[winner] = merged_secondary_weight;
    secondary_counts[winner] = merged_secondary_count;
    std::copy_n(merged_secondary_mean, 3, secondary_means + 3 * winner);
    std::copy_n(merged_secondary_covariance, 9,
                secondary_covariances + 9 * winner);
    std::copy_n(merged_secondary_viewpoint, 3,
                secondary_viewpoints + 3 * winner);
    is_split[winner] = merged_is_split ? 1 : 0;
    std::copy_n(merged_split_normal, 3, split_normals + 3 * winner);
    primary_dirty[winner] = 0;
    // SurfelPair::add marks the destination secondary dirty exactly when the
    // directed second voxel contributes a split pair, even if that secondary
    // has zero weight.  The maintenance pass intentionally leaves an empty
    // dirty secondary pending.
    if (!second_was_split) {
      secondary_dirty[winner] = secondary_dirty[first];
    } else if (first_secondary_weight == 0.0F &&
               second_secondary_weight == 0.0F) {
      secondary_dirty[winner] = 1;
    } else if (first_secondary_weight == 0.0F) {
      secondary_dirty[winner] = secondary_dirty[second];
    } else if (second_secondary_weight == 0.0F) {
      secondary_dirty[winner] = secondary_dirty[first];
    } else {
      secondary_dirty[winner] = 0;
    }

    primary_weights[loser] = 0.0F;
    primary_counts[loser] = 0;
    secondary_weights[loser] = 0.0F;
    secondary_counts[loser] = 0;
    is_split[loser] = 0;
    ++merges;
  }
  if (applied_merge_count != nullptr) {
    *applied_merge_count = merges;
  }
  return 0;
}

// Recompute the dirty PCA fields using Eigen's float self-adjoint solver,
// matching the scalar type and eigenvalue order of BaseSurfel<float>.
int navvis_recon_slam_surfel_geometry(
    std::uint64_t state_count, const float* covariances, float* normals,
    float* eigenvalues) {
  if (state_count != 0 &&
      (covariances == nullptr || normals == nullptr || eigenvalues == nullptr)) {
    return 1;
  }
  for (std::uint64_t index = 0; index < state_count; ++index) {
    const float* const values = covariances + 9 * index;
    Eigen::Vector3f normal;
    Eigen::Vector3f values_ascending;
    if (!surfel_geometry(values, &normal, &values_ascending)) {
      return 2;
    }
    for (int axis = 0; axis < 3; ++axis) {
      eigenvalues[3 * index + axis] = values_ascending[axis];
      normals[3 * index + axis] = normal[axis];
    }
  }
  return 0;
}

int navvis_recon_slam_oriented_surfel_geometry(
    const std::uint64_t state_count, const float* const covariances,
    const float* const means, const float* const viewpoint_means,
    float* const normals, float* const eigenvalues) {
  if (state_count != 0 &&
      (covariances == nullptr || means == nullptr || viewpoint_means == nullptr ||
       normals == nullptr || eigenvalues == nullptr)) {
    return 1;
  }
  int failure = 0;
#pragma omp parallel for schedule(static) if (state_count >= 1024) \
    reduction(max : failure)
  for (std::int64_t signed_index = 0;
       signed_index < static_cast<std::int64_t>(state_count);
       ++signed_index) {
    const std::uint64_t index = static_cast<std::uint64_t>(signed_index);
    Eigen::Vector3f normal;
    Eigen::Vector3f values_ascending;
    if (!surfel_geometry(covariances + 9 * index, &normal,
                         &values_ascending)) {
      failure = 2;
      continue;
    }
    orient_surfel_normal(means + 3 * index, viewpoint_means + 3 * index,
                         &normal);
    for (int axis = 0; axis < 3; ++axis) {
      eigenvalues[3 * index + axis] = values_ascending[axis];
      normals[3 * index + axis] = normal[axis];
    }
  }
  return failure;
}

// Perform the single Gauss-Newton point-to-plane step used by the local
// matcher.  The installed solver removes yaw and scales coordinates by the
// mean target radius before forming its 6x6 system.  Geometry remains float,
// while the Jacobian and normal-equation accumulation are double.
int navvis_recon_slam_point_plane_step(
    const std::uint64_t correspondence_count,
    const float* const source_points, const float* const target_points,
    const float* const target_normals,
    const double* const normalization_translation,
    const double* const normalization_quaternion_xyzw,
    double* const output_translation,
    double* const output_quaternion_xyzw, double* const output_delta,
    double* const output_scale, double* const output_normal_matrix,
    double* const output_right_hand_side) {
  if (correspondence_count == 0 || source_points == nullptr ||
      target_points == nullptr || target_normals == nullptr ||
      normalization_translation == nullptr ||
      normalization_quaternion_xyzw == nullptr ||
      output_translation == nullptr || output_quaternion_xyzw == nullptr) {
    return 1;
  }

  const Eigen::Vector3d normalization_translation_double(
      normalization_translation[0], normalization_translation[1],
      normalization_translation[2]);
  const Eigen::Quaterniond normalization_rotation_raw(
      normalization_quaternion_xyzw[3],
      normalization_quaternion_xyzw[0],
      normalization_quaternion_xyzw[1],
      normalization_quaternion_xyzw[2]);
  Eigen::Quaterniond normalization_rotation_double =
      normalization_rotation_raw;
  normalization_rotation_double.normalize();

  // The binary narrows the normalization transform to float before applying
  // it to its float point vectors.
  const Eigen::Vector3f normalization_translation_float =
      normalization_translation_double.cast<float>();
  const Eigen::Quaternionf normalization_rotation_float =
      normalization_rotation_double.cast<float>();

  double radius_sum = 0.0;
  for (std::uint64_t index = 0; index < correspondence_count; ++index) {
    const Eigen::Vector3f target(target_points[3 * index],
                                 target_points[3 * index + 1],
                                 target_points[3 * index + 2]);
    const Eigen::Vector3f normalized =
        normalization_rotation_float * target +
        normalization_translation_float;
    const float radius_squared =
        (normalized.z() * normalized.z() +
         normalized.y() * normalized.y()) +
        normalized.x() * normalized.x();
    radius_sum += static_cast<double>(std::sqrt(radius_squared));
  }
  const double mean_radius =
      radius_sum / static_cast<double>(correspondence_count);
  double scale_double = 1.0 / mean_radius;
  if (!std::isfinite(scale_double) ||
      scale_double > std::numeric_limits<double>::max()) {
    scale_double = 1.0;
  }
  const float scale = static_cast<float>(scale_double);
  if (output_scale != nullptr) {
    *output_scale = static_cast<double>(scale);
  }

  Eigen::Matrix<double, 6, 6> normal_matrix =
      Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> right_hand_side =
      Eigen::Matrix<double, 6, 1>::Zero();
  for (std::uint64_t index = 0; index < correspondence_count; ++index) {
    const Eigen::Vector3f source(source_points[3 * index],
                                 source_points[3 * index + 1],
                                 source_points[3 * index + 2]);
    const Eigen::Vector3f target(target_points[3 * index],
                                 target_points[3 * index + 1],
                                 target_points[3 * index + 2]);
    const Eigen::Vector3f target_normal(target_normals[3 * index],
                                        target_normals[3 * index + 1],
                                        target_normals[3 * index + 2]);
    const Eigen::Vector3f normalized_source =
        scale * (normalization_rotation_float * source +
                 normalization_translation_float);
    const Eigen::Vector3f normalized_target =
        scale * (normalization_rotation_float * target +
                 normalization_translation_float);
    const Eigen::Vector3f normalized_normal =
        normalization_rotation_float * target_normal;

    Eigen::Matrix<double, 6, 1> jacobian;
    const double point_x = static_cast<double>(normalized_target.x());
    const double point_y = static_cast<double>(normalized_target.y());
    const double point_z = static_cast<double>(normalized_target.z());
    const double normal_x = static_cast<double>(normalized_normal.x());
    const double normal_y = static_cast<double>(normalized_normal.y());
    const double normal_z = static_cast<double>(normalized_normal.z());
    jacobian << point_y * normal_z - point_z * normal_y,
        point_z * normal_x - point_x * normal_z,
        point_x * normal_y - point_y * normal_x, normal_x, normal_y,
        normal_z;

    // The installed SSE kernel widens the already-normalized float geometry
    // before subtracting and evaluating the dot product, then narrows only
    // the completed residual.  Its rotational Jacobian has the opposite sign
    // to Eigen's usual point-cross-normal form, so the equivalent system in
    // this coefficient order accumulates +J*r.
    const double delta_x = static_cast<double>(normalized_source.x()) -
                           static_cast<double>(normalized_target.x());
    const double delta_y = static_cast<double>(normalized_source.y()) -
                           static_cast<double>(normalized_target.y());
    const double delta_z = static_cast<double>(normalized_source.z()) -
                           static_cast<double>(normalized_target.z());
    const float residual = static_cast<float>(
        static_cast<double>(normalized_normal.z()) * delta_z +
        (static_cast<double>(normalized_normal.x()) * delta_x +
         static_cast<double>(normalized_normal.y()) * delta_y));
    for (int row = 0; row < 6; ++row) {
      right_hand_side[row] +=
          jacobian[row] * static_cast<double>(residual);
      for (int column = 0; column < 6; ++column) {
        normal_matrix(row, column) += jacobian[row] * jacobian[column];
      }
    }
  }

  if (output_normal_matrix != nullptr) {
    for (int column = 0; column < 6; ++column) {
      for (int row = 0; row < 6; ++row) {
        output_normal_matrix[6 * column + row] = normal_matrix(row, column);
      }
    }
  }
  if (output_right_hand_side != nullptr) {
    for (int row = 0; row < 6; ++row) {
      output_right_hand_side[row] = right_hand_side[row];
    }
  }

  const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(normal_matrix);
  if (decomposition.info() != Eigen::Success) {
    return 2;
  }
  const Eigen::Matrix<double, 6, 1> delta =
      decomposition.solve(right_hand_side);
  if (decomposition.info() != Eigen::Success || !delta.allFinite()) {
    return 3;
  }
  if (output_delta != nullptr) {
    for (int index = 0; index < 6; ++index) {
      output_delta[index] = delta[index];
    }
  }

  const Eigen::Vector3d rotation_vector = delta.head<3>();
  const Eigen::Quaterniond normalized_increment =
      binaryRotationVectorQuaternion(rotation_vector);
  // Undo normalization through the same three Rigid3 operations as the
  // installed solver: inverse(N) * delta_pose * N.  Keeping the intermediate
  // translations and the final quaternion normalization is observable at the
  // last few double bits and avoids rotating delta_pose.translation by its own
  // incremental rotation.
  const Eigen::Quaterniond inverse_normalization_rotation =
      binaryQuaternionInverse(normalization_rotation_raw);
  const Eigen::Vector3d inverse_normalization_translation =
      inverse_normalization_rotation * -normalization_translation_double;
  const Eigen::Vector3d normalized_delta_translation =
      delta.tail<3>() / scale_double;
  const Eigen::Vector3d first_translation =
      inverse_normalization_translation +
      inverse_normalization_rotation * normalized_delta_translation;
  const Eigen::Quaterniond first_rotation = binaryQuaternionProduct(
      inverse_normalization_rotation, normalized_increment);
  const Eigen::Vector3d increment_translation =
      first_translation +
      first_rotation * normalization_translation_double;
  const Eigen::Quaterniond inner_increment_rotation = binaryQuaternionNormalized(
      binaryQuaternionProduct(first_rotation, normalization_rotation_raw));
  // The Gauss-Newton kernel returns a normalized Rigid3, then its outer
  // matcher copies that pose into the iteration state and normalizes the
  // quaternion once more.  The second pass is usually a no-op, but it moves
  // a few coefficients by one ulp when the first norm rounded across 1.0.
  const Eigen::Quaterniond increment_rotation =
      binaryQuaternionNormalized(inner_increment_rotation);

  output_translation[0] = increment_translation.x();
  output_translation[1] = increment_translation.y();
  output_translation[2] = increment_translation.z();
  output_quaternion_xyzw[0] = increment_rotation.x();
  output_quaternion_xyzw[1] = increment_rotation.y();
  output_quaternion_xyzw[2] = increment_rotation.z();
  output_quaternion_xyzw[3] = increment_rotation.w();
  return 0;
}

// Diagnostic primitive for validating the exact Eigen AngleAxis conversion
// used by the point-to-plane kernel.  Keeping this separate from the solver
// lets frozen binary captures identify coefficient-order differences without
// changing the production step ABI.
int navvis_recon_slam_rotation_vector_quaternion(
    const double* const rotation_vector, double* const output_quaternion_xyzw) {
  if (rotation_vector == nullptr || output_quaternion_xyzw == nullptr) {
    return 1;
  }
  const Eigen::Vector3d vector(rotation_vector[0], rotation_vector[1],
                               rotation_vector[2]);
  const Eigen::Quaterniond quaternion =
      binaryRotationVectorQuaternion(vector);
  output_quaternion_xyzw[0] = quaternion.x();
  output_quaternion_xyzw[1] = quaternion.y();
  output_quaternion_xyzw[2] = quaternion.z();
  output_quaternion_xyzw[3] = quaternion.w();
  return 0;
}

int navvis_recon_slam_quaternion_transform_vector(
    const double* const quaternion_xyzw, const double* const vector,
    double* const output_vector) {
  if (quaternion_xyzw == nullptr || vector == nullptr ||
      output_vector == nullptr) {
    return 1;
  }
  const Eigen::Quaterniond quaternion(
      quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1],
      quaternion_xyzw[2]);
  const Eigen::Vector3d transformed = binaryQuaternionTransformVector(
      quaternion, Eigen::Vector3d(vector[0], vector[1], vector[2]));
  output_vector[0] = transformed.x();
  output_vector[1] = transformed.y();
  output_vector[2] = transformed.z();
  return 0;
}

int navvis_recon_slam_fast_imu_integrate(
    const std::uint64_t sample_count, const std::int64_t* const timestamps_ns,
    const double* const linear_accelerations,
    const double* const angular_velocities, const std::int64_t begin_ns,
    const std::int64_t end_ns, double* const output_quaternion_xyzw,
    double* const output_velocity) {
  if (sample_count < 2 || timestamps_ns == nullptr ||
      linear_accelerations == nullptr || angular_velocities == nullptr ||
      output_quaternion_xyzw == nullptr || output_velocity == nullptr ||
      begin_ns >= end_ns || begin_ns < timestamps_ns[0] ||
      end_ns > timestamps_ns[sample_count - 1]) {
    return 1;
  }
  const FastImuIntegral result = integrateFastImu(
      sample_count, timestamps_ns, linear_accelerations, angular_velocities,
      begin_ns, end_ns);
  output_quaternion_xyzw[0] = result.rotation.x();
  output_quaternion_xyzw[1] = result.rotation.y();
  output_quaternion_xyzw[2] = result.rotation.z();
  output_quaternion_xyzw[3] = result.rotation.w();
  output_velocity[0] = result.velocity.x();
  output_velocity[1] = result.velocity.y();
  output_velocity[2] = result.velocity.z();
  return 0;
}

int navvis_recon_slam_fast_imu_integrate_trace(
    const std::uint64_t sample_count, const std::int64_t* const timestamps_ns,
    const double* const linear_accelerations,
    const double* const angular_velocities, const std::int64_t begin_ns,
    const std::int64_t end_ns, const std::uint64_t trace_capacity,
    double* const output_trace, std::uint64_t* const output_trace_count) {
  if (sample_count < 2 || timestamps_ns == nullptr ||
      linear_accelerations == nullptr || angular_velocities == nullptr ||
      output_trace == nullptr || output_trace_count == nullptr ||
      begin_ns >= end_ns || begin_ns < timestamps_ns[0] ||
      end_ns > timestamps_ns[sample_count - 1]) {
    return 1;
  }
  integrateFastImu(sample_count, timestamps_ns, linear_accelerations,
                   angular_velocities, begin_ns, end_ns, output_trace,
                   trace_capacity, output_trace_count);
  return *output_trace_count <= trace_capacity ? 0 : 2;
}

int navvis_recon_slam_fast_imu_acceleration_measurement(
    const std::uint64_t sample_count, const std::int64_t* const timestamps_ns,
    const double* const linear_accelerations,
    const double* const angular_velocities, const std::int64_t first_ns,
    const std::int64_t second_ns, const std::int64_t third_ns,
    double* const output_measurement) {
  if (sample_count < 2 || timestamps_ns == nullptr ||
      linear_accelerations == nullptr || angular_velocities == nullptr ||
      output_measurement == nullptr || first_ns >= second_ns ||
      second_ns >= third_ns || first_ns < timestamps_ns[0] ||
      third_ns > timestamps_ns[sample_count - 1]) {
    return 1;
  }
  const std::int64_t first_duration_ns = second_ns - first_ns;
  const std::int64_t second_duration_ns = third_ns - second_ns;
  const std::int64_t first_center_ns = first_ns + first_duration_ns / 2;
  const std::int64_t second_center_ns = second_ns + second_duration_ns / 2;
  const FastImuIntegral interval = integrateFastImu(
      sample_count, timestamps_ns, linear_accelerations, angular_velocities,
      first_ns, second_ns);
  const FastImuIntegral to_first_center = integrateFastImu(
      sample_count, timestamps_ns, linear_accelerations, angular_velocities,
      first_ns, first_center_ns);
  const FastImuIntegral center_to_center = integrateFastImu(
      sample_count, timestamps_ns, linear_accelerations, angular_velocities,
      first_center_ns, second_center_ns);
  const Eigen::Quaterniond center_from_first_center =
      binaryQuaternionProduct(binaryQuaternionInverse(interval.rotation),
                              to_first_center.rotation);
  const Eigen::Vector3d measurement = binaryQuaternionTransformVector(
      center_from_first_center, center_to_center.velocity);
  output_measurement[0] = measurement.x();
  output_measurement[1] = measurement.y();
  output_measurement[2] = measurement.z();
  return 0;
}

int navvis_recon_slam_fast_imu_acceleration_measurement_variant(
    const std::uint64_t sample_count, const std::int64_t* const timestamps_ns,
    const double* const linear_accelerations,
    const double* const angular_velocities, const std::int64_t first_ns,
    const std::int64_t second_ns, const std::int64_t third_ns,
    const std::uint32_t variant, double* const output_measurement) {
  if (sample_count < 2 || timestamps_ns == nullptr ||
      linear_accelerations == nullptr || angular_velocities == nullptr ||
      output_measurement == nullptr || first_ns >= second_ns ||
      second_ns >= third_ns || first_ns < timestamps_ns[0] ||
      third_ns > timestamps_ns[sample_count - 1] || variant > 17) {
    return 1;
  }
  const std::int64_t first_duration_ns = second_ns - first_ns;
  const std::int64_t second_duration_ns = third_ns - second_ns;
  const std::int64_t first_center_ns = first_ns + first_duration_ns / 2;
  const std::int64_t second_center_ns = second_ns + second_duration_ns / 2;
  const FastImuIntegral interval = integrateFastImu(
      sample_count, timestamps_ns, linear_accelerations, angular_velocities,
      first_ns, second_ns);
  const FastImuIntegral to_first_center = integrateFastImu(
      sample_count, timestamps_ns, linear_accelerations, angular_velocities,
      first_ns, first_center_ns);
  const FastImuIntegral center_to_center = integrateFastImuVariant(
      sample_count, timestamps_ns, linear_accelerations, angular_velocities,
      first_center_ns, second_center_ns, variant);
  const Eigen::Quaterniond center_from_first_center =
      binaryQuaternionProduct(binaryQuaternionInverse(interval.rotation),
                              to_first_center.rotation);
  const Eigen::Vector3d measurement = binaryQuaternionTransformVector(
      center_from_first_center, center_to_center.velocity);
  output_measurement[0] = measurement.x();
  output_measurement[1] = measurement.y();
  output_measurement[2] = measurement.z();
  return 0;
}

// Expose the scalar paths used while resolving the installed point-cloud
// transform. Variant 3 is the production path recovered at image 0x4b85b0:
// both the double pose and input vector are narrowed before Eigen's float
// Quaternion::_transformVector is evaluated.
int navvis_recon_slam_deskew_points_variant(
    const std::uint64_t point_count, const double* const input_points,
    const double* const quaternion_xyzw, const double* const translation,
    const std::uint32_t variant, float* const output_points) {
  if ((point_count != 0 &&
       (input_points == nullptr || quaternion_xyzw == nullptr ||
        translation == nullptr || output_points == nullptr))) {
    return 1;
  }

  for (std::uint64_t index = 0; index < point_count; ++index) {
    const Eigen::Quaterniond rotation(
        quaternion_xyzw[4 * index + 3], quaternion_xyzw[4 * index],
        quaternion_xyzw[4 * index + 1], quaternion_xyzw[4 * index + 2]);
    const Eigen::Vector3d point(input_points[3 * index],
                                input_points[3 * index + 1],
                                input_points[3 * index + 2]);
    const Eigen::Vector3d offset(translation[3 * index],
                                 translation[3 * index + 1],
                                 translation[3 * index + 2]);
    Eigen::Vector3f transformed;
    switch (variant) {
      case 0:
        transformed = (rotation * point + offset).cast<float>();
        break;
      case 1:
        transformed =
            (rotation.toRotationMatrix() * point + offset).cast<float>();
        break;
      case 2: {
        const Eigen::Quaterniond normalized_rotation = rotation.normalized();
        transformed =
            (normalized_rotation * point + offset).cast<float>();
        break;
      }
      case 3:
        transformed = rotation.cast<float>() * point.cast<float>() +
                      offset.cast<float>();
        break;
      case 4:
        transformed = rotation.cast<float>().toRotationMatrix() *
                          point.cast<float>() +
                      offset.cast<float>();
        break;
      case 5: {
        const Eigen::Matrix3d matrix = rotation.toRotationMatrix();
        for (int axis = 0; axis < 3; ++axis) {
          transformed[axis] = static_cast<float>(
              (matrix(axis, 0) * point.x() +
               matrix(axis, 1) * point.y()) +
              matrix(axis, 2) * point.z() + offset[axis]);
        }
        break;
      }
      case 6: {
        const Eigen::Matrix3d matrix = rotation.toRotationMatrix();
        for (int axis = 0; axis < 3; ++axis) {
          transformed[axis] = static_cast<float>(
              matrix(axis, 0) * point.x() +
              (matrix(axis, 1) * point.y() +
               matrix(axis, 2) * point.z()) +
              offset[axis]);
        }
        break;
      }
      default:
        return 2;
    }
    output_points[3 * index] = transformed.x();
    output_points[3 * index + 1] = transformed.y();
    output_points[3 * index + 2] = transformed.z();
  }
  return 0;
}

int navvis_recon_slam_deskew_points(
    const std::uint64_t point_count, const double* const input_points,
    const double* const quaternion_xyzw, const double* const translation,
    float* const output_points) {
  return navvis_recon_slam_deskew_points_variant(
      point_count, input_points, quaternion_xyzw, translation, 3,
      output_points);
}

// The range-data accumulator stores world_from_start as an inverse pose, then
// reconstructs it with a second inverse when it builds end_from_start.  The
// double inverse is not algebraically removable: the first raw-IMU quaternion
// is slightly non-unit, so Eigen's Quaternion::inverse() leaves a measurable
// translation round-trip at startup.  Keep the same Rigid3 operation order as
// the installed binary (inverse, inverse, inverse, compose).
int navvis_recon_slam_end_from_start_pose(
    const double* const start_translation,
    const double* const start_quaternion_xyzw,
    const double* const end_translation,
    const double* const end_quaternion_xyzw,
    double* const output_pose) {
  if (start_translation == nullptr || start_quaternion_xyzw == nullptr ||
      end_translation == nullptr || end_quaternion_xyzw == nullptr ||
      output_pose == nullptr) {
    return 1;
  }

  const Eigen::Vector3d start_position(
      start_translation[0], start_translation[1], start_translation[2]);
  const Eigen::Quaterniond start_rotation(
      start_quaternion_xyzw[3], start_quaternion_xyzw[0],
      start_quaternion_xyzw[1], start_quaternion_xyzw[2]);
  const Eigen::Vector3d end_position(
      end_translation[0], end_translation[1], end_translation[2]);
  const Eigen::Quaterniond end_rotation(
      end_quaternion_xyzw[3], end_quaternion_xyzw[0],
      end_quaternion_xyzw[1], end_quaternion_xyzw[2]);

  const Eigen::Quaterniond start_from_world_rotation =
      start_rotation.inverse();
  const Eigen::Vector3d start_from_world_translation =
      start_from_world_rotation * -start_position;
  const Eigen::Quaterniond world_from_start_rotation =
      start_from_world_rotation.inverse();
  const Eigen::Vector3d world_from_start_translation =
      world_from_start_rotation * -start_from_world_translation;

  const Eigen::Quaterniond end_from_world_rotation = end_rotation.inverse();
  const Eigen::Vector3d end_from_world_translation =
      end_from_world_rotation * -end_position;
  const Eigen::Vector3d end_from_start_translation =
      end_from_world_translation +
      end_from_world_rotation * world_from_start_translation;
  const Eigen::Quaterniond end_from_start_rotation =
      end_from_world_rotation * world_from_start_rotation;

  output_pose[0] = end_from_start_translation.x();
  output_pose[1] = end_from_start_translation.y();
  output_pose[2] = end_from_start_translation.z();
  output_pose[3] = end_from_start_rotation.x();
  output_pose[4] = end_from_start_rotation.y();
  output_pose[5] = end_from_start_rotation.z();
  output_pose[6] = end_from_start_rotation.w();
  return 0;
}

// Rigid3 primitives used by the local matcher.  These intentionally retain
// non-unit quaternion coefficients and mirror the installed inverse/compose
// helpers instead of passing through scipy.spatial.transform.Rotation.
int navvis_recon_slam_inverse_pose(
    const double* const translation,
    const double* const quaternion_xyzw,
    double* const output_pose) {
  if (translation == nullptr || quaternion_xyzw == nullptr ||
      output_pose == nullptr) {
    return 1;
  }
  const Eigen::Vector3d position(translation[0], translation[1],
                                 translation[2]);
  const Eigen::Quaterniond rotation(
      quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1],
      quaternion_xyzw[2]);
  const Eigen::Quaterniond inverse_rotation =
      binaryQuaternionInverse(rotation);
  const Eigen::Vector3d inverse_translation =
      -(inverse_rotation * position);
  output_pose[0] = inverse_translation.x();
  output_pose[1] = inverse_translation.y();
  output_pose[2] = inverse_translation.z();
  output_pose[3] = inverse_rotation.x();
  output_pose[4] = inverse_rotation.y();
  output_pose[5] = inverse_rotation.z();
  output_pose[6] = inverse_rotation.w();
  return 0;
}

int navvis_recon_slam_compose_pose(
    const double* const lhs_translation,
    const double* const lhs_quaternion_xyzw,
    const double* const rhs_translation,
    const double* const rhs_quaternion_xyzw,
    double* const output_pose) {
  if (lhs_translation == nullptr || lhs_quaternion_xyzw == nullptr ||
      rhs_translation == nullptr || rhs_quaternion_xyzw == nullptr ||
      output_pose == nullptr) {
    return 1;
  }
  const Eigen::Vector3d lhs_position(lhs_translation[0], lhs_translation[1],
                                     lhs_translation[2]);
  const Eigen::Quaterniond lhs_rotation(
      lhs_quaternion_xyzw[3], lhs_quaternion_xyzw[0],
      lhs_quaternion_xyzw[1], lhs_quaternion_xyzw[2]);
  const Eigen::Vector3d rhs_position(rhs_translation[0], rhs_translation[1],
                                     rhs_translation[2]);
  const Eigen::Quaterniond rhs_rotation(
      rhs_quaternion_xyzw[3], rhs_quaternion_xyzw[0],
      rhs_quaternion_xyzw[1], rhs_quaternion_xyzw[2]);
  const Eigen::Vector3d output_translation =
      lhs_position + lhs_rotation * rhs_position;
  const Eigen::Quaterniond output_rotation =
      binaryQuaternionProduct(lhs_rotation, rhs_rotation);
  output_pose[0] = output_translation.x();
  output_pose[1] = output_translation.y();
  output_pose[2] = output_translation.z();
  output_pose[3] = output_rotation.x();
  output_pose[4] = output_rotation.y();
  output_pose[5] = output_rotation.z();
  output_pose[6] = output_rotation.w();
  return 0;
}

int navvis_recon_slam_compose_pose_normalized(
    const double* const lhs_translation,
    const double* const lhs_quaternion_xyzw,
    const double* const rhs_translation,
    const double* const rhs_quaternion_xyzw,
    double* const output_pose) {
  if (lhs_translation == nullptr || lhs_quaternion_xyzw == nullptr ||
      rhs_translation == nullptr || rhs_quaternion_xyzw == nullptr ||
      output_pose == nullptr) {
    return 1;
  }
  const Eigen::Vector3d lhs_position(lhs_translation[0], lhs_translation[1],
                                     lhs_translation[2]);
  const Eigen::Quaterniond lhs_rotation(
      lhs_quaternion_xyzw[3], lhs_quaternion_xyzw[0],
      lhs_quaternion_xyzw[1], lhs_quaternion_xyzw[2]);
  const Eigen::Vector3d rhs_position(rhs_translation[0], rhs_translation[1],
                                     rhs_translation[2]);
  const Eigen::Quaterniond rhs_rotation(
      rhs_quaternion_xyzw[3], rhs_quaternion_xyzw[0],
      rhs_quaternion_xyzw[1], rhs_quaternion_xyzw[2]);
  const Eigen::Vector3d output_translation =
      lhs_position + lhs_rotation * rhs_position;
  const Eigen::Quaterniond output_rotation = binaryQuaternionNormalized(
      binaryQuaternionProduct(lhs_rotation, rhs_rotation));
  output_pose[0] = output_translation.x();
  output_pose[1] = output_translation.y();
  output_pose[2] = output_translation.z();
  output_pose[3] = output_rotation.x();
  output_pose[4] = output_rotation.y();
  output_pose[5] = output_rotation.z();
  output_pose[6] = output_rotation.w();
  return 0;
}

int navvis_recon_slam_predict_translation(
    const std::int64_t previous_timestamp_ns,
    const double* const previous_translation,
    const std::int64_t anchor_timestamp_ns,
    const double* const anchor_translation,
    const double* const anchor_quaternion_xyzw,
    const std::int64_t query_timestamp_ns,
    double* const output_translation) {
  if (previous_translation == nullptr || anchor_translation == nullptr ||
      anchor_quaternion_xyzw == nullptr || output_translation == nullptr ||
      anchor_timestamp_ns <= previous_timestamp_ns ||
      query_timestamp_ns < anchor_timestamp_ns) {
    return 1;
  }
  const Eigen::Vector3d previous(previous_translation[0],
                                 previous_translation[1],
                                 previous_translation[2]);
  const Eigen::Vector3d anchor(anchor_translation[0], anchor_translation[1],
                               anchor_translation[2]);
  const Eigen::Quaterniond anchor_rotation(
      anchor_quaternion_xyzw[3], anchor_quaternion_xyzw[0],
      anchor_quaternion_xyzw[1], anchor_quaternion_xyzw[2]);
  const double pose_delta_seconds =
      static_cast<double>(anchor_timestamp_ns - previous_timestamp_ns) /
      1.0e9;
  const double inverse_pose_delta_seconds = 1.0 / pose_delta_seconds;
  const Eigen::Vector3d linear_velocity =
      (anchor - previous) * inverse_pose_delta_seconds;
  const double extrapolation_seconds =
      static_cast<double>(query_timestamp_ns - anchor_timestamp_ns) / 1.0e9;
  const Eigen::Vector3d displacement =
      extrapolation_seconds * linear_velocity;
  // The installed predictor represents a world-frame linear displacement as
  // a local rigid increment and then composes that increment with the anchor
  // pose. Do not simplify the inverse/forward rotation pair: its packed Eigen
  // rounding is observable in autonomous prediction.
  const Eigen::Vector3d local_displacement =
      binaryQuaternionRotate(binaryQuaternionInverse(anchor_rotation),
                             displacement);
  const Eigen::Vector3d prediction =
      anchor + binaryQuaternionRotate(anchor_rotation, local_displacement);
  output_translation[0] = prediction.x();
  output_translation[1] = prediction.y();
  output_translation[2] = prediction.z();
  return 0;
}

int navvis_recon_slam_icp_normalization_pose(
    const double* const translation,
    const double* const quaternion_xyzw,
    double* const output_pose) {
  if (translation == nullptr || quaternion_xyzw == nullptr ||
      output_pose == nullptr) {
    return 1;
  }
  const Eigen::Vector3d position(translation[0], translation[1],
                                 translation[2]);
  const Eigen::Quaterniond rotation(
      quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1],
      quaternion_xyzw[2]);
  const Eigen::Quaterniond inverse_rotation = rotation.inverse();
  const Eigen::Vector3d gravity_in_tracking =
      inverse_rotation * Eigen::Vector3d::UnitZ();
  const Eigen::Quaterniond gravity_alignment =
      Eigen::Quaterniond::FromTwoVectors(gravity_in_tracking,
                                         Eigen::Vector3d::UnitZ());
  const Eigen::Quaterniond inverse_gravity_alignment =
      binaryQuaternionInverse(gravity_alignment);
  const Eigen::Quaterniond denormalization_rotation =
      binaryQuaternionProduct(rotation, inverse_gravity_alignment);
  const Eigen::Quaterniond normalization_rotation =
      binaryQuaternionInverse(denormalization_rotation);
  const Eigen::Vector3d normalization_translation =
      -(normalization_rotation * position);
  // The helper computes translation with the raw inverse product.  Its
  // returned Rigid3 wrapper then normalizes only the copied quaternion, so
  // the translation must not be recomputed from the normalized coefficients.
  const Eigen::Quaterniond output_normalization_rotation =
      binaryQuaternionNormalized(normalization_rotation);
  output_pose[0] = normalization_translation.x();
  output_pose[1] = normalization_translation.y();
  output_pose[2] = normalization_translation.z();
  output_pose[3] = output_normalization_rotation.x();
  output_pose[4] = output_normalization_rotation.y();
  output_pose[5] = output_normalization_rotation.z();
  output_pose[6] = output_normalization_rotation.w();
  return 0;
}

int navvis_recon_slam_icp_normalization_diagnostics(
    const double* const quaternion_xyzw, double* const output_quaternions) {
  if (quaternion_xyzw == nullptr || output_quaternions == nullptr) {
    return 1;
  }
  const Eigen::Quaterniond rotation(
      quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1],
      quaternion_xyzw[2]);
  const Eigen::Quaterniond inverse_rotation = rotation.inverse();
  const Eigen::Vector3d gravity_in_tracking =
      inverse_rotation * Eigen::Vector3d::UnitZ();
  const Eigen::Quaterniond gravity_alignment =
      Eigen::Quaterniond::FromTwoVectors(gravity_in_tracking,
                                         Eigen::Vector3d::UnitZ());
  const Eigen::Quaterniond inverse_gravity_alignment =
      binaryQuaternionInverse(gravity_alignment);
  const Eigen::Quaterniond denormalization_rotation =
      binaryQuaternionProduct(rotation, inverse_gravity_alignment);
  const Eigen::Quaterniond normalization_rotation =
      binaryQuaternionInverse(denormalization_rotation);
  const Eigen::Quaterniond values[3] = {
      inverse_rotation, gravity_alignment, normalization_rotation};
  for (int index = 0; index < 3; ++index) {
    output_quaternions[4 * index] = values[index].x();
    output_quaternions[4 * index + 1] = values[index].y();
    output_quaternions[4 * index + 2] = values[index].z();
    output_quaternions[4 * index + 3] = values[index].w();
  }
  return 0;
}

int navvis_recon_slam_transform_points(
    const std::uint64_t point_count, const float* const input_points,
    const double* const translation,
    const double* const quaternion_xyzw, float* const output_double_pose,
    float* const output_float_pose) {
  if ((point_count != 0 && input_points == nullptr) || translation == nullptr ||
      quaternion_xyzw == nullptr || output_double_pose == nullptr ||
      output_float_pose == nullptr) {
    return 1;
  }
  Eigen::Quaterniond rotation_double(
      quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1],
      quaternion_xyzw[2]);
  rotation_double.normalize();
  const Eigen::Vector3d translation_double(translation[0], translation[1],
                                           translation[2]);
  const Eigen::Quaternionf rotation_float = rotation_double.cast<float>();
  const Eigen::Vector3f translation_float = translation_double.cast<float>();
  for (std::uint64_t index = 0; index < point_count; ++index) {
    const Eigen::Vector3f point(input_points[3 * index],
                                input_points[3 * index + 1],
                                input_points[3 * index + 2]);
    const Eigen::Vector3f transformed_double =
        (rotation_double * point.cast<double>() + translation_double)
            .cast<float>();
    const Eigen::Vector3f transformed_float =
        rotation_float * point + translation_float;
    for (int axis = 0; axis < 3; ++axis) {
      output_double_pose[3 * index + axis] = transformed_double[axis];
      output_float_pose[3 * index + axis] = transformed_float[axis];
    }
  }
  return 0;
}

// Reproduce the float-pose path used when a double trajectory pose is cast
// before range data is inserted into a surfel grid.  The input quaternion is
// intentionally not normalized: the first firmware IMU orientation can be a
// few ulps away from unit length, and Eigen's Quaternionf::toRotationMatrix()
// preserves that distinction.
int navvis_recon_slam_transform_points_raw_float_matrix(
    const std::uint64_t point_count, const float* const input_points,
    const double* const translation, const double* const quaternion_xyzw,
    float* const output_points) {
  if ((point_count != 0 && input_points == nullptr) || translation == nullptr ||
      quaternion_xyzw == nullptr || output_points == nullptr) {
    return 1;
  }
  const float translation_float[3] = {
      static_cast<float>(translation[0]), static_cast<float>(translation[1]),
      static_cast<float>(translation[2])};
  const float quaternion_float[4] = {
      static_cast<float>(quaternion_xyzw[0]),
      static_cast<float>(quaternion_xyzw[1]),
      static_cast<float>(quaternion_xyzw[2]),
      static_cast<float>(quaternion_xyzw[3])};
  const FloatPoseMatrix pose =
      make_float_pose_matrix(translation_float, quaternion_float);
  for (std::uint64_t index = 0; index < point_count; ++index) {
    transform_point_exact(pose, input_points + 3 * index,
                          output_points + 3 * index);
  }
  return 0;
}

// The local matcher's geometric correspondence filter evaluates the pose
// again instead of reusing the point cloud transformed for nearest-neighbour
// search.  It narrows the raw inverse Rigid3 to Quaternionf/Vector3f and uses
// Eigen's Quaternion::_transformVector path directly (image 0x6c93e2).
int navvis_recon_slam_transform_points_raw_float_quaternion(
    const std::uint64_t point_count, const float* const input_points,
    const double* const translation, const double* const quaternion_xyzw,
    float* const output_points) {
  if ((point_count != 0 && input_points == nullptr) || translation == nullptr ||
      quaternion_xyzw == nullptr || output_points == nullptr) {
    return 1;
  }
  const Eigen::Quaternionf rotation(
      static_cast<float>(quaternion_xyzw[3]),
      static_cast<float>(quaternion_xyzw[0]),
      static_cast<float>(quaternion_xyzw[1]),
      static_cast<float>(quaternion_xyzw[2]));
  const Eigen::Vector3f position(
      static_cast<float>(translation[0]),
      static_cast<float>(translation[1]),
      static_cast<float>(translation[2]));
  for (std::uint64_t index = 0; index < point_count; ++index) {
    const Eigen::Vector3f point(input_points[3 * index],
                                input_points[3 * index + 1],
                                input_points[3 * index + 2]);
    const Eigen::Vector3f transformed = rotation * point + position;
    output_points[3 * index] = transformed.x();
    output_points[3 * index + 1] = transformed.y();
    output_points[3 * index + 2] = transformed.z();
  }
  return 0;
}

// The iterative local matcher constructs an Eigen double homogeneous matrix,
// casts all sixteen coefficients to float, and only then transforms its float
// point vector.  This differs from range insertion, which narrows the
// quaternion before constructing the matrix.
int navvis_recon_slam_transform_points_double_matrix_cast(
    const std::uint64_t point_count, const float* const input_points,
    const double* const translation, const double* const quaternion_xyzw,
    float* const output_points) {
  if ((point_count != 0 && input_points == nullptr) || translation == nullptr ||
      quaternion_xyzw == nullptr || output_points == nullptr) {
    return 1;
  }
  const Eigen::Vector3d position(translation[0], translation[1],
                                 translation[2]);
  const Eigen::Quaterniond rotation(
      quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1],
      quaternion_xyzw[2]);
  Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
  matrix.topLeftCorner<3, 3>() = rotation.toRotationMatrix();
  matrix.topRightCorner<3, 1>() = position;
  const Eigen::Matrix4f matrix_float = matrix.cast<float>();
  FloatPoseMatrix pose;
  for (int column = 0; column < 4; ++column) {
    pose.columns[column] = _mm_set_ps(
        matrix_float(3, column), matrix_float(2, column),
        matrix_float(1, column), matrix_float(0, column));
  }
  for (std::uint64_t index = 0; index < point_count; ++index) {
    transform_point_exact(pose, input_points + 3 * index,
                          output_points + 3 * index);
  }
  return 0;
}

int navvis_recon_slam_transform_submap_range_data(
    const std::uint64_t point_count, const float* const input_points,
    const float* const input_normals, const float* const input_origins,
    const double* const node_translation,
    const double* const node_quaternion_xyzw,
    const double* const submap_translation,
    const double* const submap_quaternion_xyzw, float* const output_points,
    float* const output_normals, float* const output_origins,
    double* const output_relative_pose) {
  if ((point_count != 0 &&
       (input_points == nullptr || input_normals == nullptr ||
        input_origins == nullptr || output_points == nullptr ||
        output_normals == nullptr || output_origins == nullptr)) ||
      node_translation == nullptr || node_quaternion_xyzw == nullptr ||
      submap_translation == nullptr || submap_quaternion_xyzw == nullptr ||
      output_relative_pose == nullptr) {
    return 1;
  }

  const Eigen::Vector3d node_position(node_translation[0], node_translation[1],
                                      node_translation[2]);
  const Eigen::Quaterniond node_rotation(
      node_quaternion_xyzw[3], node_quaternion_xyzw[0],
      node_quaternion_xyzw[1], node_quaternion_xyzw[2]);
  const Eigen::Vector3d submap_position(
      submap_translation[0], submap_translation[1], submap_translation[2]);
  const Eigen::Quaterniond submap_rotation(
      submap_quaternion_xyzw[3], submap_quaternion_xyzw[0],
      submap_quaternion_xyzw[1], submap_quaternion_xyzw[2]);

  const Eigen::Quaterniond inverse_submap_rotation =
      submap_rotation.conjugate();
  const Eigen::Vector3d inverse_submap_translation =
      inverse_submap_rotation * -submap_position;
  const Eigen::Vector3d relative_translation =
      inverse_submap_translation + inverse_submap_rotation * node_position;
  const Eigen::Quaterniond relative_rotation =
      inverse_submap_rotation * node_rotation;

  output_relative_pose[0] = relative_translation.x();
  output_relative_pose[1] = relative_translation.y();
  output_relative_pose[2] = relative_translation.z();
  output_relative_pose[3] = relative_rotation.x();
  output_relative_pose[4] = relative_rotation.y();
  output_relative_pose[5] = relative_rotation.z();
  output_relative_pose[6] = relative_rotation.w();

  const float translation_float[3] = {
      static_cast<float>(relative_translation.x()),
      static_cast<float>(relative_translation.y()),
      static_cast<float>(relative_translation.z())};
  const float quaternion_float[4] = {
      static_cast<float>(relative_rotation.x()),
      static_cast<float>(relative_rotation.y()),
      static_cast<float>(relative_rotation.z()),
      static_cast<float>(relative_rotation.w())};
  const FloatPoseMatrix pose =
      make_float_pose_matrix(translation_float, quaternion_float);
  const float zero_translation[3] = {0.F, 0.F, 0.F};
  const FloatPoseMatrix direction_pose =
      make_float_pose_matrix(zero_translation, quaternion_float);

  for (std::uint64_t index = 0; index < point_count; ++index) {
    transform_point_exact(pose, input_points + 3 * index,
                          output_points + 3 * index);
    transform_point_exact(direction_pose, input_normals + 3 * index,
                          output_normals + 3 * index);
    transform_point_exact(pose, input_origins + 3 * index,
                          output_origins + 3 * index);
  }
  return 0;
}

}  // extern "C"
