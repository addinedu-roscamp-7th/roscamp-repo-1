#include "pickee_mobile_wonho/components/sensor_manager.hpp"

namespace pickee_mobile_wonho {

SensorManager::SensorManager(std::shared_ptr<rclcpp::Logger> logger)
    : logger_(logger)
{
    RCLCPP_INFO(*logger_, "[SensorManager] 센서 매니저 초기화 완료");
}

} // namespace pickee_mobile_wonho