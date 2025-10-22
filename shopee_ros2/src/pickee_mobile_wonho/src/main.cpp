#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "pickee_mobile_wonho/mobile_controller.hpp"

/**
 * @brief 메인 함수
 * 
 * Pickee Mobile Controller 노드를 초기화하고 실행합니다.
 */
int main(int argc, char* argv[]) {
    // ROS2 초기화
    rclcpp::init(argc, argv);

    try {
        // Pickee Mobile Controller 노드 생성
        auto node = std::make_shared<pickee_mobile_wonho::PickeeMobileController>();
        
        RCLCPP_INFO(node->get_logger(), "Pickee Mobile Controller가 시작되었습니다.");
        
        // 멀티스레드 실행자 사용 (성능 향상)
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(node);
        
        // 노드 실행
        executor.spin();
        
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), 
            "Pickee Mobile Controller 실행 중 오류 발생: %s", e.what());
        return -1;
    }

    // ROS2 종료
    rclcpp::shutdown();
    RCLCPP_INFO(rclcpp::get_logger("main"), "Pickee Mobile Controller가 종료되었습니다.");
    
    return 0;
}