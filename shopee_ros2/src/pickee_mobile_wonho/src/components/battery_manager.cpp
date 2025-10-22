#include "pickee_mobile_wonho/components/battery_manager.hpp"

namespace pickee_mobile_wonho {

BatteryManager::BatteryManager(std::shared_ptr<rclcpp::Logger> logger)
    : logger_(logger)
    , battery_level_(1.0)  // 100% 시작
{
    RCLCPP_INFO(*logger_, "[BatteryManager] 배터리 매니저 초기화 완료 (배터리: %.0f%%)", 
        battery_level_.load() * 100.0);
}

void BatteryManager::UpdateBatteryLevel(double level) {
    battery_level_.store(std::clamp(level, 0.0, 1.0));
    RCLCPP_DEBUG(*logger_, "[BatteryManager] 배터리 레벨 업데이트: %.0f%%", 
        battery_level_.load() * 100.0);
}

double BatteryManager::GetCurrentLevel() const {
    return battery_level_.load();
}

} // namespace pickee_mobile_wonho