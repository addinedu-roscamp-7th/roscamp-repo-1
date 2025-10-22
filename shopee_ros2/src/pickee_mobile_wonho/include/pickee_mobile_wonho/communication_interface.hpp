#pragma once

#include <memory>
#include "rclcpp/rclcpp.hpp"

namespace pickee_mobile_wonho {

class CommunicationInterface {
public:
    explicit CommunicationInterface(std::shared_ptr<rclcpp::Logger> logger);
    ~CommunicationInterface() = default;

private:
    std::shared_ptr<rclcpp::Logger> logger_;
};

} // namespace pickee_mobile_wonho