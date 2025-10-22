#include "pickee_mobile_wonho/components/path_planning_component.hpp"

namespace pickee_mobile_wonho {

PathPlanningComponent::PathPlanningComponent(std::shared_ptr<rclcpp::Logger> logger)
    : logger_(logger)
{
    RCLCPP_INFO(*logger_, "[PathPlanningComponent] 경로 계획 컴포넌트 초기화 완료");
}

bool PathPlanningComponent::PlanGlobalPath(const geometry_msgs::msg::PoseStamped& start, 
                                          const geometry_msgs::msg::PoseStamped& goal) {
    // TODO: A* 알고리즘 구현
    RCLCPP_INFO(*logger_, "[PathPlanningComponent] 경로 계획 시작: (%.2f, %.2f) -> (%.2f, %.2f)",
        start.pose.position.x, start.pose.position.y,
        goal.pose.position.x, goal.pose.position.y);
    
    // 임시 직선 경로 생성
    current_path_.header.frame_id = "map";
    current_path_.header.stamp = rclcpp::Clock().now();
    current_path_.poses.clear();
    
    // 시작점 추가
    current_path_.poses.push_back(start);
    
    // 목표점 추가
    current_path_.poses.push_back(goal);
    
    return true;
}

nav_msgs::msg::Path PathPlanningComponent::GetCurrentPath() const {
    return current_path_;
}

} // namespace pickee_mobile_wonho