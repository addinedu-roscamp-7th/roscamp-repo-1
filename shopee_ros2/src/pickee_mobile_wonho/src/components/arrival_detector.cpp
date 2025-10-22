#include "pickee_mobile_wonho/components/arrival_detector.hpp"

namespace pickee_mobile_wonho {

ArrivalDetector::ArrivalDetector(std::shared_ptr<rclcpp::Logger> logger)
    : logger_(logger)
{
    RCLCPP_INFO(*logger_, "[ArrivalDetector] 도착 감지기 초기화 완료");
}

} // namespace pickee_mobile_wonho