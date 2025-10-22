#pragma once

#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

namespace pickee_mobile_wonho {

class PathPlanningComponent {
public:
    explicit PathPlanningComponent(std::shared_ptr<rclcpp::Logger> logger);
    ~PathPlanningComponent() = default;

    bool PlanGlobalPath(const geometry_msgs::msg::PoseStamped& start, 
                       const geometry_msgs::msg::PoseStamped& goal);
    nav_msgs::msg::Path GetCurrentPath() const;

private:
    std::shared_ptr<rclcpp::Logger> logger_;
    nav_msgs::msg::Path current_path_;
};

} // namespace pickee_mobile_wonho