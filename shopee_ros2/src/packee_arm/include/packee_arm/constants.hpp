#ifndef PACKEE_ARM_CONSTANTS_HPP_
#define PACKEE_ARM_CONSTANTS_HPP_

namespace packee_arm {

constexpr double kTranslationTolerance = 0.003;
constexpr double kYawToleranceDeg = 3.0;
constexpr int kMaxServoSteps = 15;
constexpr double kMyCobotReach = 0.28;  // myCobot 280 베이스에서 엔드이펙터까지의 유효 수평 반경 (m)
constexpr double kMyCobotMinZ = 0.05;
constexpr double kMyCobotMaxZ = 0.30;
constexpr double kBoundingBoxReferenceArea = 30000.0;  // eye-in-hand 카메라 해상도 기준 bbox 면적 정규화 팩터

}  // namespace packee_arm

#endif  // PACKEE_ARM_CONSTANTS_HPP_
