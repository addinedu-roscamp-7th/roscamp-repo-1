#pragma once

#include <memory>
#include "rclcpp/rclcpp.hpp"

namespace pickee_mobile_wonho {

class BatteryManager {
public:
    explicit BatteryManager(std::shared_ptr<rclcpp::Logger> logger);
    ~BatteryManager() = default;
    
    void UpdateBatteryLevel(double level);
    double GetCurrentLevel() const;

private:
    std::shared_ptr<rclcpp::Logger> logger_;
    std::atomic<double> battery_level_;
};

} // namespace pickee_mobile_wonho