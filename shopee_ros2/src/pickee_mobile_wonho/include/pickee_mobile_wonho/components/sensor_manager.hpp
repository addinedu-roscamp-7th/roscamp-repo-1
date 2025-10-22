#pragma once

#include <memory>
#include "rclcpp/rclcpp.hpp"

namespace pickee_mobile_wonho {

/**
 * @brief 센서 데이터 관리 컴포넌트
 */
class SensorManager {
public:
    explicit SensorManager(std::shared_ptr<rclcpp::Logger> logger);
    ~SensorManager() = default;

private:
    std::shared_ptr<rclcpp::Logger> logger_;
};

} // namespace pickee_mobile_wonho