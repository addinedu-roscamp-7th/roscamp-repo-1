#include "pickee_mobile_wonho/states/charging_state.hpp"
#include <algorithm>

namespace pickee_mobile_wonho {

ChargingState::ChargingState(std::shared_ptr<rclcpp::Logger> logger)
    : logger_(logger)
    , charge_start_time_(std::chrono::steady_clock::now())
    , battery_level_(0.0)
    , initial_battery_level_(0.0)
{
    if (!logger_) {
        throw std::invalid_argument("Logger cannot be null");
    }
}

void ChargingState::OnEnter() {
    charge_start_time_ = std::chrono::steady_clock::now();
    initial_battery_level_ = battery_level_;
    
    RCLCPP_INFO(*logger_, 
        "[ChargingState] 충전 상태에 진입했습니다. 현재 배터리: %.1f%%",
        battery_level_ * 100.0);
        
    if (battery_level_ >= CHARGE_COMPLETE_THRESHOLD) {
        RCLCPP_INFO(*logger_, "[ChargingState] 배터리가 이미 충분히 충전되어 있습니다.");
    }
}

void ChargingState::Execute() {
    // 배터리 충전 시뮬레이션
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - charge_start_time_);
    
    // 충전 진행 (CHARGE_RATE_PER_MINUTE 속도로)
    double minutes_elapsed = elapsed.count() / 60.0;
    double charge_gained = minutes_elapsed * CHARGE_RATE_PER_MINUTE;
    battery_level_ = std::min(1.0, initial_battery_level_ + charge_gained);
    
    // 주기적으로 충전 진행 상황 보고 (30초마다)
    if (elapsed.count() > 0 && elapsed.count() % 30 == 0) {
        double charge_progress = (battery_level_ - initial_battery_level_) * 100.0;
        
        RCLCPP_INFO_THROTTLE(*logger_, *rclcpp::Clock().get_clock_type_rcl_clock_t(), 30000,
            "[ChargingState] 충전 중... 현재: %.1f%% (+%.1f%%, 경과시간: %.1f분)",
            battery_level_ * 100.0, charge_progress, minutes_elapsed);
    }
    
    // 충전 완료 확인
    if (IsChargingComplete()) {
        RCLCPP_INFO(*logger_, 
            "[ChargingState] 배터리 충전이 완료되었습니다! (%.1f%%, 충전시간: %.1f분)",
            battery_level_ * 100.0, minutes_elapsed);
    }
}

void ChargingState::OnExit() {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - charge_start_time_);
    double minutes_elapsed = elapsed.count() / 60.0;
    double charge_gained = (battery_level_ - initial_battery_level_) * 100.0;
    
    RCLCPP_INFO(*logger_, 
        "[ChargingState] 충전 상태를 종료합니다. "
        "(최종 배터리: %.1f%%, 충전량: +%.1f%%, 충전시간: %.1f분)", 
        battery_level_ * 100.0, charge_gained, minutes_elapsed);
}

StateType ChargingState::GetType() const {
    return StateType::CHARGING;
}

void ChargingState::SetBatteryLevel(double level) {
    battery_level_ = std::clamp(level, 0.0, 1.0);
}

double ChargingState::GetBatteryLevel() const {
    return battery_level_;
}

bool ChargingState::IsChargingComplete() const {
    return battery_level_ >= CHARGE_COMPLETE_THRESHOLD;
}

} // namespace pickee_mobile_wonho