#ifndef PACKEE_ARM_VISUAL_SERVO_MODULE_HPP_
#define PACKEE_ARM_VISUAL_SERVO_MODULE_HPP_

#include <mutex>

#include "packee_arm/constants.hpp"
#include "packee_arm/types.hpp"

namespace packee_arm {

// 시각 서보 명령 결과를 표현한다.
struct ServoCommand {
  double vx;
  double vy;
  double vz;
  double yaw_rate_deg;
  double error_norm;
  double confidence;
  bool goal_reached;
};

// VisualServoModule 클래스는 두-stream CNN 추론 결과를 이용한 P 제어를 담당한다.
class VisualServoModule {
public:
  VisualServoModule(
    double servo_gain_xy,
    double servo_gain_z,
    double servo_gain_yaw,
    double confidence_threshold,
    double max_translation_speed,
    double max_yaw_speed_deg);

  ServoCommand ComputeCommand(const PoseEstimate & current, const PoseEstimate & target);

  void UpdateGains(double servo_gain_xy, double servo_gain_z, double servo_gain_yaw);

  void UpdateConstraints(
    double confidence_threshold,
    double max_translation_speed,
    double max_yaw_speed_deg);

  double GetConfidenceThreshold() const;

private:
  mutable std::mutex mutex_;
  double servo_gain_xy_;
  double servo_gain_z_;
  double servo_gain_yaw_;
  double confidence_threshold_;
  double max_translation_speed_;
  double max_yaw_speed_deg_;
};

}  // namespace packee_arm

#endif  // PACKEE_ARM_VISUAL_SERVO_MODULE_HPP_
