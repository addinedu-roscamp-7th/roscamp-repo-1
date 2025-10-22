#pragma once

#include <memory>
#include "rclcpp/rclcpp.hpp"

namespace pickee_mobile_wonho {

class ArrivalDetector {
public:
    explicit ArrivalDetector(std::shared_ptr<rclcpp::Logger> logger);
    ~ArrivalDetector() = default;

private:
    std::shared_ptr<rclcpp::Logger> logger_;
};

} // namespace pickee_mobile_wonho