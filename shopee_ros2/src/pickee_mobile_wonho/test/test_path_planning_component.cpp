#include <gtest/gtest.h>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "pickee_mobile_wonho/components/path_planning_component.hpp"
#include "pickee_mobile_wonho/components/motion_control_component.hpp"

class PathPlanningComponentTest : public ::testing::Test {
protected:
    void SetUp() override {
        rclcpp::init(0, nullptr);
        auto logger = std::make_shared<rclcpp::Logger>(rclcpp::get_logger("test"));
        path_planning_component_ = std::make_unique<pickee_mobile_wonho::PathPlanningComponent>(logger);
    }
    
    void TearDown() override {
        rclcpp::shutdown();
    }
    
    std::unique_ptr<pickee_mobile_wonho::PathPlanningComponent> path_planning_component_;
};

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

TEST_F(PathPlanningComponentTest, BasicPathPlanning) {
    geometry_msgs::msg::PoseStamped start, goal;
    start.pose.position.x = 0.0;
    start.pose.position.y = 0.0;
    goal.pose.position.x = 1.0;
    goal.pose.position.y = 1.0;
    
    bool success = path_planning_component_->PlanGlobalPath(start, goal);
    EXPECT_TRUE(success);
    
    auto path = path_planning_component_->GetCurrentPath();
    EXPECT_GE(path.poses.size(), 2u);  // 최소 시작점과 끝점
}

TEST_F(MotionControlComponentTest, EmergencyStop) {
    // 정상 상태에서 명령 계산
    auto cmd_vel = motion_control_component_->ComputeControlCommand();
    // 기본 상태에서는 정지
    
    // 비상 정지 활성화
    motion_control_component_->EmergencyStop();
    
    cmd_vel = motion_control_component_->ComputeControlCommand();
    EXPECT_EQ(cmd_vel.linear.x, 0.0);
    EXPECT_EQ(cmd_vel.angular.z, 0.0);
}

TEST_F(MotionControlComponentTest, SpeedLimits) {
    motion_control_component_->SetSpeedLimits(1.5, 2.0);
    // 속도 제한 설정이 정상적으로 이루어지는지 확인
    EXPECT_NO_THROW(motion_control_component_->SetSpeedLimits(1.5, 2.0));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}