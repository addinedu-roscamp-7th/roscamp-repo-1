#include "vicpinky_bringup/bringup.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include <cmath>

VicPinky::VicPinky() : Node("vic_pinky_bringup"),
                       x_(0.0), y_(0.0), theta_(0.0),
                       last_encoder_l_(0), last_encoder_r_(0),
                       is_initialized_(false) {
    RCLCPP_INFO(this->get_logger(), "Initializing Vic Pinky Node...");

    // ROS 통신 설정
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
    joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 10, std::bind(&VicPinky::twist_callback, this, std::placeholders::_1));
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    timer_ = this->create_wall_timer(std::chrono::milliseconds(33),
                                     std::bind(&VicPinky::update_and_publish, this));

    last_time_ = this->get_clock()->now();
    is_initialized_ = true;
    RCLCPP_INFO(this->get_logger(), "Vic Pinky Node started successfully.");
}

VicPinky::~VicPinky() {
    on_shutdown();
}

void VicPinky::twist_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    double linear_x = msg->linear.x;
    double angular_z = msg->angular.z;

    // 단순 속도 모델 (실제 모터 드라이버 연동 필요)
    double v_l = linear_x - (angular_z * WHEEL_BASE / 2.0);
    double v_r = linear_x + (angular_z * WHEEL_BASE / 2.0);

    int rpm_l = static_cast<int>(v_l / (WHEEL_RAD * RPM2RAD));
    int rpm_r = static_cast<int>(v_r / (WHEEL_RAD * RPM2RAD));

    rpm_l = std::max(std::min(rpm_l, 28), -28);
    rpm_r = std::max(std::min(rpm_r, 28), -28);

    RCLCPP_DEBUG(this->get_logger(), "cmd_vel → RPM L=%d, R=%d", rpm_l, rpm_r);

    // TODO: ZLACDriver 같은 하드웨어 드라이버와 연동 필요
}

void VicPinky::update_and_publish() {
    rclcpp::Time current_time = this->get_clock()->now();
    double dt = (current_time - last_time_).seconds();
    if (dt <= 0.0) return;

    // TODO: 모터 드라이버에서 엔코더 값 읽기
    int encoder_l = last_encoder_l_ + 10;
    int encoder_r = last_encoder_r_ + 10;

    int delta_l_pulses = encoder_l - last_encoder_l_;
    int delta_r_pulses = encoder_r - last_encoder_r_;
    last_encoder_l_ = encoder_l;
    last_encoder_r_ = encoder_r;

    double dist_l = (delta_l_pulses / static_cast<double>(PULSE_PER_ROT)) * CIRCUMFERENCE;
    double dist_r = (delta_r_pulses / static_cast<double>(PULSE_PER_ROT)) * CIRCUMFERENCE;

    double delta_distance = (dist_r + dist_l) / 2.0;
    double delta_theta = (dist_r - dist_l) / WHEEL_BASE;
    theta_ += delta_theta;

    x_ += delta_distance * cos(theta_);
    y_ += delta_distance * sin(theta_);

    double v_x = delta_distance / dt;
    double vth = delta_theta / dt;

    publish_tf(current_time);
    publish_odometry(current_time, v_x, vth);
    publish_joint_states(current_time, dist_l / dt, dist_r / dt);

    last_time_ = current_time;
}

void VicPinky::publish_tf(const rclcpp::Time &current_time) {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = current_time;
    t.header.frame_id = "odom";
    t.child_frame_id = "base_footprint";
    t.transform.translation.x = x_;
    t.transform.translation.y = y_;
    t.transform.translation.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0, 0, theta_);
    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(t);
}

void VicPinky::publish_odometry(const rclcpp::Time &current_time, double v_x, double vth) {
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = current_time;
    odom.header.frame_id = "odom";
    odom.child_frame_id = "base_footprint";

    odom.pose.pose.position.x = x_;
    odom.pose.pose.position.y = y_;

    tf2::Quaternion q;
    q.setRPY(0, 0, theta_);
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    odom.twist.twist.linear.x = v_x;
    odom.twist.twist.angular.z = vth;

    odom_pub_->publish(odom);
}

void VicPinky::publish_joint_states(const rclcpp::Time &current_time, double vel_l_rads, double vel_r_rads) {
    sensor_msgs::msg::JointState joint;
    joint.header.stamp = current_time;
    joint.name = {"left_wheel_joint", "right_wheel_joint"};
    joint.position = {
        (last_encoder_l_ / static_cast<double>(PULSE_PER_ROT)) * 2 * M_PI,
        (last_encoder_r_ / static_cast<double>(PULSE_PER_ROT)) * 2 * M_PI
    };
    joint.velocity = {vel_l_rads, vel_r_rads};
    joint_pub_->publish(joint);
}

void VicPinky::on_shutdown() {
    RCLCPP_INFO(this->get_logger(), "Shutting down VicPinky...");
    // TODO: 모터 정지, 드라이버 종료
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VicPinky>();  // VicPinky 클래스 사용
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}