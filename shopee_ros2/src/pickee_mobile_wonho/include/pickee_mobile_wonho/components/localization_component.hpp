#pragma once

#include <memory>
#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"

namespace pickee_mobile_wonho {

/**
 * @brief 위치 추정 컴포넌트
 * 
 * 센서 데이터 융합을 통한 고정밀 위치 추정을 제공합니다.
 */
class LocalizationComponent {
public:
    /**
     * @brief 생성자
     * @param logger ROS2 로거
     */
    explicit LocalizationComponent(std::shared_ptr<rclcpp::Logger> logger);

    /**
     * @brief 소멸자
     */
    ~LocalizationComponent() = default;

    /**
     * @brief 센서 데이터 업데이트
     */
    void UpdateSensorData(const sensor_msgs::msg::LaserScan::SharedPtr scan);
    void UpdateOdometry(const nav_msgs::msg::Odometry::SharedPtr odom);
    void UpdateImu(const sensor_msgs::msg::Imu::SharedPtr imu);

    /**
     * @brief 현재 위치 반환
     */
    Eigen::Vector3d GetCurrentPose() const;
    
    /**
     * @brief 위치 공분산 반환
     */
    Eigen::Matrix3d GetPoseCovariance() const;

private:
    std::shared_ptr<rclcpp::Logger> logger_;
    Eigen::Vector3d current_pose_;      // [x, y, theta]
    Eigen::Matrix3d pose_covariance_;
};

} // namespace pickee_mobile_wonho