#pragma once

#include <memory>
#include "rclcpp/rclcpp.hpp"

namespace pickee_mobile_wonho {

class SensorManager {
public:
    explicit SensorManager(std::shared_ptr<rclcpp::Logger> logger);
    ~SensorManager() = default;

private:
    std::shared_ptr<rclcpp::Logger> logger_;
};

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

class ArrivalDetector {
public:
    explicit ArrivalDetector(std::shared_ptr<rclcpp::Logger> logger);
    ~ArrivalDetector() = default;

private:
    std::shared_ptr<rclcpp::Logger> logger_;
};

} // namespace pickee_mobile_wonho