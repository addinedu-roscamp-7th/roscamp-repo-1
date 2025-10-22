#include <gtest/gtest.h>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "pickee_mobile_wonho/state_machine.hpp"
#include "pickee_mobile_wonho/states/idle_state.hpp"
#include "pickee_mobile_wonho/states/moving_state.hpp"
#include "pickee_mobile_wonho/states/stopped_state.hpp"
#include "pickee_mobile_wonho/states/charging_state.hpp"
#include "pickee_mobile_wonho/states/error_state.hpp"

class StateMachineTest : public ::testing::Test {
protected:
    void SetUp() override {
        rclcpp::init(0, nullptr);
        auto logger = std::make_shared<rclcpp::Logger>(rclcpp::get_logger("test"));
        state_machine_ = std::make_unique<pickee_mobile_wonho::StateMachine>(logger);
        logger_ = logger;
    }
    
    void TearDown() override {
        rclcpp::shutdown();
    }
    
    std::unique_ptr<pickee_mobile_wonho::StateMachine> state_machine_;
    std::shared_ptr<rclcpp::Logger> logger_;
};

/**
 * @brief 초기 상태 테스트
 */
TEST_F(StateMachineTest, InitialState) {
    // 초기화되지 않은 상태에서는 IDLE 반환
    EXPECT_EQ(state_machine_->GetCurrentStateType(), pickee_mobile_wonho::StateType::IDLE);
    EXPECT_FALSE(state_machine_->IsInitialized());
}

/**
 * @brief 상태 전환 테스트
 */
TEST_F(StateMachineTest, StateTransition) {
    // IDLE 상태로 초기화
    auto idle_state = std::make_unique<pickee_mobile_wonho::IdleState>(logger_);
    state_machine_->TransitionTo(std::move(idle_state));
    
    EXPECT_EQ(state_machine_->GetCurrentStateType(), pickee_mobile_wonho::StateType::IDLE);
    EXPECT_TRUE(state_machine_->IsInitialized());
    
    // MOVING 상태로 전환
    auto moving_state = std::make_unique<pickee_mobile_wonho::MovingState>(logger_);
    state_machine_->TransitionTo(std::move(moving_state));
    
    EXPECT_EQ(state_machine_->GetCurrentStateType(), pickee_mobile_wonho::StateType::MOVING);
}

/**
 * @brief 상태 실행 테스트
 */
TEST_F(StateMachineTest, StateExecution) {
    // IDLE 상태로 초기화
    auto idle_state = std::make_unique<pickee_mobile_wonho::IdleState>(logger_);
    state_machine_->TransitionTo(std::move(idle_state));
    
    // 상태 실행 (예외 발생하지 않아야 함)
    EXPECT_NO_THROW(state_machine_->Execute());
}

/**
 * @brief 모든 상태 타입 테스트
 */
TEST_F(StateMachineTest, AllStateTypes) {
    // IDLE
    auto idle_state = std::make_unique<pickee_mobile_wonho::IdleState>(logger_);
    state_machine_->TransitionTo(std::move(idle_state));
    EXPECT_EQ(state_machine_->GetCurrentStateType(), pickee_mobile_wonho::StateType::IDLE);
    
    // MOVING
    auto moving_state = std::make_unique<pickee_mobile_wonho::MovingState>(logger_);
    state_machine_->TransitionTo(std::move(moving_state));
    EXPECT_EQ(state_machine_->GetCurrentStateType(), pickee_mobile_wonho::StateType::MOVING);
    
    // STOPPED
    auto stopped_state = std::make_unique<pickee_mobile_wonho::StoppedState>(logger_);
    state_machine_->TransitionTo(std::move(stopped_state));
    EXPECT_EQ(state_machine_->GetCurrentStateType(), pickee_mobile_wonho::StateType::STOPPED);
    
    // CHARGING
    auto charging_state = std::make_unique<pickee_mobile_wonho::ChargingState>(logger_);
    state_machine_->TransitionTo(std::move(charging_state));
    EXPECT_EQ(state_machine_->GetCurrentStateType(), pickee_mobile_wonho::StateType::CHARGING);
    
    // ERROR
    auto error_state = std::make_unique<pickee_mobile_wonho::ErrorState>(logger_);
    state_machine_->TransitionTo(std::move(error_state));
    EXPECT_EQ(state_machine_->GetCurrentStateType(), pickee_mobile_wonho::StateType::ERROR);
}

/**
 * @brief 널 포인터 처리 테스트
 */
TEST_F(StateMachineTest, NullPointerHandling) {
    // 널 상태 전환 시도
    EXPECT_THROW(state_machine_->TransitionTo(nullptr), std::invalid_argument);
}

/**
 * @brief 상태 전환 콜백 테스트
 */
TEST_F(StateMachineTest, StateTransitionCallback) {
    bool callback_called = false;
    pickee_mobile_wonho::StateType old_state, new_state;
    
    // 콜백 설정
    state_machine_->SetStateTransitionCallback(
        [&](pickee_mobile_wonho::StateType old_s, pickee_mobile_wonho::StateType new_s) {
            callback_called = true;
            old_state = old_s;
            new_state = new_s;
        }
    );
    
    // 상태 전환
    auto idle_state = std::make_unique<pickee_mobile_wonho::IdleState>(logger_);
    state_machine_->TransitionTo(std::move(idle_state));
    
    // 콜백 호출 확인
    EXPECT_TRUE(callback_called);
    EXPECT_EQ(new_state, pickee_mobile_wonho::StateType::IDLE);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}