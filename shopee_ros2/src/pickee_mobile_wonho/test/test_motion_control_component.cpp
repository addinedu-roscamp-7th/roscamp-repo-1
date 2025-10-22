#include <gtest/gtest.h>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "pickee_mobile_wonho/components/motion_control_component.hpp"

class MotionControlComponentTest : public ::testing::Test {
protected:
    void SetUp() override {
        rclcpp::init(0, nullptr);
        auto logger = std::make_shared<rclcpp::Logger>(rclcpp::get_logger("test"));
        motion_control_component_ = std::make_unique<pickee_mobile_wonho::MotionControlComponent>(logger);
    }
    
    void TearDown() override {
        rclcpp::shutdown();
    }
    
    std::unique_ptr<pickee_mobile_wonho::MotionControlComponent> motion_control_component_;
};

TEST_F(MotionControlComponentTest, EmergencyStop) {
    // 정상 상태에서 명령 계산
    auto cmd_vel = motion_control_component_->ComputeControlCommand();
    
    // 비상 정지 활성화
    motion_control_component_->EmergencyStop();
    
    cmd_vel = motion_control_component_->ComputeControlCommand();
    EXPECT_EQ(cmd_vel.linear.x, 0.0);
    EXPECT_EQ(cmd_vel.angular.z, 0.0);
}

TEST_F(MotionControlComponentTest, SpeedLimits) {
    motion_control_component_->SetSpeedLimits(1.5, 2.0);
    EXPECT_NO_THROW(motion_control_component_->SetSpeedLimits(1.5, 2.0));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}