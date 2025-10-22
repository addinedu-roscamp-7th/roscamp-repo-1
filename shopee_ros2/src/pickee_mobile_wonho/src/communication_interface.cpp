#include "pickee_mobile_wonho/communication_interface.hpp"

namespace pickee_mobile_wonho {

CommunicationInterface::CommunicationInterface(std::shared_ptr<rclcpp::Logger> logger)
    : logger_(logger)
{
    RCLCPP_INFO(*logger_, "[CommunicationInterface] 통신 인터페이스 초기화 완료");
}

} // namespace pickee_mobile_wonho