#include "pickee_mobile_wonho/custom_planner.hpp"
#include <pluginlib/class_list_macros.hpp>

namespace custom_planner
{

    CustomPlanner::CustomPlanner()
    {
        global_path = nav_msgs::msg::Path();

        path_logged = false;

        // 각 좌표 x,y 상수 초기화
        x1_ = -3.5;
        x2_ = -2.0;
        x3_ = -1.1;
        x4_ = -0.195;
        x5_ = 0.98;
        x6_ = 2.25;
        x7_ = 3.13;

        y1_ = 1.6;
        y2_ = 0.83;
        y3_ = 0.16;
        y4_ = -1.0;
        y5_ = -4.95;

        waypoints_x_ = {x1_, x2_, x3_, x4_, x5_, x6_, x7_};
        waypoints_y_ = {y1_, y2_, y3_, y4_, y5_};

        // 웨이포인트 map 상수 초기화
        waypoints_ = {
            {"x1y2", {waypoints_y_[0], waypoints_x_[0]}},
            {"x1y2", {waypoints_y_[0], waypoints_x_[1]}},
            {"x1y3", {waypoints_y_[0], waypoints_x_[2]}},
            {"x1y4", {waypoints_y_[0], waypoints_x_[3]}},
            {"x1y5", {waypoints_y_[0], waypoints_x_[4]}},
            {"x1y6", {waypoints_y_[0], waypoints_x_[5]}}, // 1행
            {"x2y1", {waypoints_y_[1], waypoints_x_[0]}},
            {"x2y2", {waypoints_y_[1], waypoints_x_[1]}},
            {"x2y3", {waypoints_y_[1], waypoints_x_[2]}},
            {"x2y5", {waypoints_y_[1], waypoints_x_[4]}},
            {"x2y6", {waypoints_y_[1], waypoints_x_[5]}},
            {"x2y7", {waypoints_y_[1], waypoints_x_[6]}}, // 2행
            {"x3y1", {waypoints_y_[2], waypoints_x_[0]}},
            {"x3y3", {waypoints_y_[2], waypoints_x_[2]}},
            {"x3y4", {waypoints_y_[2], waypoints_x_[3]}},
            {"x3y5", {waypoints_y_[2], waypoints_x_[4]}},
            {"x3y7", {waypoints_y_[2], waypoints_x_[6]}}, // 3행
            {"x4y1", {waypoints_y_[3], waypoints_x_[0]}},
            {"x4y2", {waypoints_y_[3], waypoints_x_[1]}},
            {"x4y3", {waypoints_y_[3], waypoints_x_[2]}},
            {"x4y4", {waypoints_y_[3], waypoints_x_[3]}},
            {"x4y5", {waypoints_y_[3], waypoints_x_[4]}},
            {"x4y6", {waypoints_y_[3], waypoints_x_[5]}},
            {"x4y7", {waypoints_y_[3], waypoints_x_[6]}}, // 4행
            {"x5y1", {waypoints_y_[4], waypoints_x_[0]}},
            {"x5y4", {waypoints_y_[4], waypoints_x_[3]}},
            {"x5y7", {waypoints_y_[4], waypoints_x_[6]}} // 5행
        };
    }

    void CustomPlanner::configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr &parent,
        std::string name,
        std::shared_ptr<tf2_ros::Buffer> tf,
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
    {
        node_ = parent;
        name_ = name;
        tf_ = tf;
        costmap_ros_ = costmap_ros;

        auto node = node_.lock();
        if (node)
        {
            logger_ = node->get_logger();
            RCLCPP_INFO(logger_, "CustomPlanner가 구성되었습니다.");

            // 초기화 서비스 생성
            reset_service_ =  node->create_service<std_srvs::srv::Trigger>(
                "custom_planner/reset",
                std::bind(&CustomPlanner::ResetCallback, this, std::placeholders::_1, std::placeholders::_2)
            );
        }
    }

    void CustomPlanner::ResetCallback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        (void)request;
        
        // 초기화
        path_logged = false;
        global_path.poses.clear();
        
        response->success = true;
        response->message = "CustomPlanner 초기화 완료";
        RCLCPP_INFO(logger_, "CustomPlanner가 초기화되었습니다.");
    }


    void CustomPlanner::cleanup()
    {
        RCLCPP_INFO(logger_, "CustomPlanner가 cleanup되었습니다.");
    }

    void CustomPlanner::activate()
    {
        RCLCPP_INFO(logger_, "CustomPlanner가 activate되었습니다.");
    }

    void CustomPlanner::deactivate()
    {
        RCLCPP_INFO(logger_, "CustomPlanner가 deactivate되었습니다.");
    }

    nav_msgs::msg::Path CustomPlanner::createPlan(
        const geometry_msgs::msg::PoseStamped &start,
        const geometry_msgs::msg::PoseStamped &goal,
        std::function<bool()> cancel_checker)
    {
        (void)cancel_checker;
        global_path.header.frame_id = "map";
        global_path.header.stamp = rclcpp::Clock().now();
        
        if (!path_logged) {
            double diff_x = 10.0;
            double diff_y = 10.0;
            size_t x_index = 0;
            size_t y_index = 0;
    
            
            // 간단한 직선 경로 생성
            global_path.poses.push_back(start);
            // waypoints_ 순서대로 경유
            for (size_t i = 0; i < waypoints_x_.size(); ++i)
            {
                if (start.pose.position.x < goal.pose.position.x) {
                    if (waypoints_x_[i] > start.pose.position.x && waypoints_x_[i] < goal.pose.position.x) {
                        double current_diff_x = std::abs(start.pose.position.x - waypoints_x_[i]); // x
                        if (current_diff_x < diff_x) {
                            diff_x = current_diff_x;
                            x_index = i;
                        }
                    }
                } else if (start.pose.position.x > goal.pose.position.x) {
                    if (waypoints_x_[i] < start.pose.position.x && waypoints_x_[i] > goal.pose.position.x) {
                        double current_diff_x = std::abs(start.pose.position.x - waypoints_x_[i]); // x
                        if (current_diff_x < diff_x) {
                            diff_x = current_diff_x;
                            x_index = i;
                        }
                    }
                } else if (start.pose.position.x == goal.pose.position.x) {
                    continue;
                }
            }
            for (size_t i = 0; i < waypoints_y_.size(); ++i)
            {
                if (start.pose.position.y < goal.pose.position.y) {
                    if (waypoints_y_[i] > start.pose.position.y && waypoints_y_[i] < goal.pose.position.y) {
                        double current_diff_y = std::abs(start.pose.position.y - waypoints_y_[i]); // y
                        if (current_diff_y < diff_y) {
                            diff_y = current_diff_y;
                            y_index = i;
                        }
                    }
                } else if (start.pose.position.y > goal.pose.position.y) {
                    if (waypoints_y_[i] < start.pose.position.y && waypoints_y_[i] > goal.pose.position.y) {
                        double current_diff_y = std::abs(start.pose.position.y - waypoints_y_[i]); // y
                        if (current_diff_y < diff_y) {
                            diff_y = current_diff_y;
                            y_index = i;
                        }
                    }
                } else if (start.pose.position.y == goal.pose.position.y) {
                    continue;
                }
            }
            geometry_msgs::msg::PoseStamped waypoint;
            waypoint.pose.position.x = waypoints_x_[x_index];
            waypoint.pose.position.y = waypoints_y_[y_index];
            // waypoint.pose.orientation = goal.pose.orientation;
            global_path.poses.push_back(waypoint);
            global_path.poses.push_back(goal);
    
            RCLCPP_INFO(logger_, "CustomPlanner의 createPlan이 호출되었습니다.");
            path_logged = true;
            return global_path;
        }
        return global_path;
    }

} // namespace custom_planner

// 플러그인 등록
PLUGINLIB_EXPORT_CLASS(custom_planner::CustomPlanner, nav2_core::GlobalPlanner)
