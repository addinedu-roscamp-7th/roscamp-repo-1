#ifndef VIC_PINKY_HPP
#define VIC_PINKY_HPP

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

// 로봇 파라미터
constexpr double WHEEL_RAD = 0.0825;
constexpr int PULSE_PER_ROT = 4096;
constexpr double WHEEL_BASE = 0.475;
constexpr double RPM2RAD = 0.104719755;
constexpr double CIRCUMFERENCE = 2 * M_PI * WHEEL_RAD;

class VicPinky : public rclcpp::Node {
public:
    VicPinky();
    ~VicPinky();

private:
    void twist_callback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void update_and_publish();
    void publish_tf(const rclcpp::Time &current_time);
    void publish_odometry(const rclcpp::Time &current_time, double v_x, double vth);
    void publish_joint_states(const rclcpp::Time &current_time, double vel_l_rads, double vel_r_rads);
    void on_shutdown();

    // ROS 통신 객체
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;

    // 로봇 상태 변수
    double x_, y_, theta_;
    rclcpp::Time last_time_;
    int last_encoder_l_, last_encoder_r_;

    bool is_initialized_;
};

#endif // VIC_PINKY_HPP
