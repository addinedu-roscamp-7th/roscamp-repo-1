#include <gtest/gtest.h>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "pickee_mobile_wonho/components/localization_component.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"

class LocalizationComponentTest : public ::testing::Test {
protected:
    void SetUp() override {
        rclcpp::init(0, nullptr);
        auto logger = std::make_shared<rclcpp::Logger>(rclcpp::get_logger("test"));
        localization_component_ = std::make_unique<pickee_mobile_wonho::LocalizationComponent>(logger);
    }
    
    void TearDown() override {
        rclcpp::shutdown();
    }
    
    std::unique_ptr<pickee_mobile_wonho::LocalizationComponent> localization_component_;
};

TEST_F(LocalizationComponentTest, InitialPose) {
    auto pose = localization_component_->GetCurrentPose();
    EXPECT_EQ(pose(0), 0.0);  // x
    EXPECT_EQ(pose(1), 0.0);  // y  
    EXPECT_EQ(pose(2), 0.0);  // theta
}

TEST_F(LocalizationComponentTest, OdometryUpdate) {
    auto odom_msg = std::make_shared<nav_msgs::msg::Odometry>();
    odom_msg->pose.pose.position.x = 1.0;
    odom_msg->pose.pose.position.y = 2.0;
    odom_msg->pose.pose.orientation.w = 1.0;  // 0도 회전
    
    localization_component_->UpdateOdometry(odom_msg);
    
    auto pose = localization_component_->GetCurrentPose();
    EXPECT_DOUBLE_EQ(pose(0), 1.0);
    EXPECT_DOUBLE_EQ(pose(1), 2.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}