#include "pickee_mobile_wonho/components/localization_component.hpp"

namespace pickee_mobile_wonho {

LocalizationComponent::LocalizationComponent(std::shared_ptr<rclcpp::Logger> logger)
    : logger_(logger)
    , current_pose_(Eigen::Vector3d::Zero())
    , pose_covariance_(Eigen::Matrix3d::Identity())
{
    RCLCPP_INFO(*logger_, "[LocalizationComponent] 위치 추정 컴포넌트 초기화 완료");
}

void LocalizationComponent::UpdateSensorData(const sensor_msgs::msg::LaserScan::SharedPtr scan) {
    // TODO: 레이저 스캔 데이터 처리 구현
    RCLCPP_DEBUG(*logger_, "[LocalizationComponent] 레이저 스캔 데이터 업데이트");
}

void LocalizationComponent::UpdateOdometry(const nav_msgs::msg::Odometry::SharedPtr odom) {
    // 기본적인 오도메트리 데이터 업데이트
    current_pose_(0) = odom->pose.pose.position.x;
    current_pose_(1) = odom->pose.pose.position.y;
    
    // 쿼터니언에서 오일러 각도 변환 (간단한 yaw 계산)
    double siny_cosp = 2 * (odom->pose.pose.orientation.w * odom->pose.pose.orientation.z + 
                           odom->pose.pose.orientation.x * odom->pose.pose.orientation.y);
    double cosy_cosp = 1 - 2 * (odom->pose.pose.orientation.y * odom->pose.pose.orientation.y + 
                                odom->pose.pose.orientation.z * odom->pose.pose.orientation.z);
    current_pose_(2) = std::atan2(siny_cosp, cosy_cosp);
    
    RCLCPP_DEBUG(*logger_, "[LocalizationComponent] 위치 업데이트: (%.2f, %.2f, %.2f)",
        current_pose_(0), current_pose_(1), current_pose_(2));
}

void LocalizationComponent::UpdateImu(const sensor_msgs::msg::Imu::SharedPtr imu) {
    // TODO: IMU 데이터 처리 구현
    RCLCPP_DEBUG(*logger_, "[LocalizationComponent] IMU 데이터 업데이트");
}

Eigen::Vector3d LocalizationComponent::GetCurrentPose() const {
    return current_pose_;
}

Eigen::Matrix3d LocalizationComponent::GetPoseCovariance() const {
    return pose_covariance_;
}

} // namespace pickee_mobile_wonho