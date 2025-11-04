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
            {"x1y1", {waypoints_y_[0], waypoints_x_[0]}},
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

    // 1. 출발지 좌표 찾기 함수
    std::optional<geometry_msgs::msg::PoseStamped> CustomPlanner::GetStartWaypoint(
        const geometry_msgs::msg::PoseStamped &start, 
        const geometry_msgs::msg::PoseStamped &goal)
    {
        double diff_x = 10.0;
        double diff_y = 10.0;
        size_t x_index = 10;
        size_t y_index = 10;

        // waypoints_ 순서대로 경유
        RCLCPP_INFO(logger_, "[GetStartWaypoint 시작]start x: %.2f, goal x: %.2f", 
                    start.pose.position.x, goal.pose.position.x);
        RCLCPP_INFO(logger_, "start y: %.2f, goal y: %.2f",
                    start.pose.position.y, goal.pose.position.y);
        RCLCPP_INFO(logger_, "[for1] waypoints_y_");
        for (size_t i = 0; i < waypoints_y_.size(); ++i)
        {   
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            RCLCPP_INFO(logger_, "--%zu번째", i);
            if (start.pose.position.x < goal.pose.position.x) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (waypoints_y_[i] > start.pose.position.x && waypoints_y_[i] <= goal.pose.position.x) {
                    RCLCPP_INFO(logger_, "----검토 중인 waypoint_y_: %.2f", waypoints_y_[i]);
                    double current_diff_y = std::abs(start.pose.position.x - waypoints_y_[i]); // x
                    if (current_diff_y < diff_y) {
                        diff_y = current_diff_y;
                        y_index = i;
                        RCLCPP_INFO(logger_, "------[y_index 변경: %zu]", y_index);
                    }
                } else {
                    RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
                }
            } else if (start.pose.position.x > goal.pose.position.x) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (waypoints_y_[i] < start.pose.position.x && waypoints_y_[i] >= goal.pose.position.x) {
                    RCLCPP_INFO(logger_, "----검토 중인 waypoint_y_: %.2f", waypoints_y_[i]);
                    double current_diff_y = std::abs(start.pose.position.x - waypoints_y_[i]); // x
                    if (current_diff_y < diff_y) {
                        diff_y = current_diff_y;
                        y_index = i;
                        RCLCPP_INFO(logger_, "------[y_index 변경: %zu]", y_index);
                    }
                } else {
                    RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
            }
        }
        // 잠시 0.5초 대기 (필요 시 상단에 <thread>와 <chrono> 포함)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        RCLCPP_INFO(logger_, "[for2] waypoints_x_");
        for (size_t i = 0; i < waypoints_x_.size(); ++i)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            RCLCPP_INFO(logger_, "--%zu번째", i);
            if (start.pose.position.y < goal.pose.position.y) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (waypoints_x_[i] > start.pose.position.y && waypoints_x_[i] <= goal.pose.position.y) {
                    RCLCPP_INFO(logger_, "----검토 중인 waypoint_x_: %.2f", waypoints_x_[i]);
                    double current_diff_x = std::abs(start.pose.position.y - waypoints_x_[i]); // y
                    if (current_diff_x < diff_x) {
                        diff_x = current_diff_x;
                        x_index = i;
                        RCLCPP_INFO(logger_, "------[x_index 변경: %zu]", x_index);
                    }
                } else {
                    RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
                }
            } else if (start.pose.position.y > goal.pose.position.y) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (waypoints_x_[i] < start.pose.position.y && waypoints_x_[i] >= goal.pose.position.y) {
                    RCLCPP_INFO(logger_, "----검토 중인 waypoint_x_: %.2f", waypoints_x_[i]);
                    double current_diff_x = std::abs(start.pose.position.y - waypoints_x_[i]); // y
                    if (current_diff_x < diff_x) {
                        diff_x = current_diff_x;
                        x_index = i;
                        RCLCPP_INFO(logger_, "------[x_index 변경: %zu]", x_index);
                    }
                } else {
                    RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
            }
        }
        if (x_index == 10 || y_index == 10) {
            RCLCPP_WARN(logger_, "적절한 웨이포인트를 찾지 못했습니다. 기본 경로를 사용합니다.");
            return std::nullopt;
        } else {
            geometry_msgs::msg::PoseStamped waypoint;
            waypoint.pose.position.x = waypoints_y_[y_index];
            waypoint.pose.position.y = waypoints_x_[x_index];
            // waypoint.pose.orientation = goal.pose.orientation;
            RCLCPP_INFO(logger_, "추가된 웨이포인트: (%.2f, %.2f)", 
                waypoint.pose.position.x, waypoint.pose.position.y);
            return waypoint;
        }
    }

    std::optional<geometry_msgs::msg::PoseStamped> CustomPlanner::GetEndWaypoint(
        const geometry_msgs::msg::PoseStamped &start, 
        const geometry_msgs::msg::PoseStamped &goal)
    {
        double diff_x = 10.0;
        double diff_y = 10.0;
        size_t x_index = 10;
        size_t y_index = 10;

        // waypoints_ 순서대로 경유
        RCLCPP_INFO(logger_, "[GetEndWaypoint 시작] start x: %.2f, goal x: %.2f", 
                    start.pose.position.x, goal.pose.position.x);
        RCLCPP_INFO(logger_, "start y: %.2f, goal y: %.2f",
                    start.pose.position.y, goal.pose.position.y);
        RCLCPP_INFO(logger_, "[for1] waypoints_y_");
        for (size_t i = 0; i < waypoints_y_.size(); ++i)
        {   
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            RCLCPP_INFO(logger_, "--%zu번째", i);
            if (start.pose.position.x < goal.pose.position.x) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (waypoints_y_[i] > start.pose.position.x && waypoints_y_[i] <= goal.pose.position.x) {
                    RCLCPP_INFO(logger_, "----검토 중인 waypoint_y_: %.2f", waypoints_y_[i]);
                    double current_diff_y = std::abs(goal.pose.position.x - waypoints_y_[i]); // x
                    if (current_diff_y < diff_y) {
                        diff_y = current_diff_y;
                        y_index = i;
                        RCLCPP_INFO(logger_, "------[y_index 변경: %zu]", y_index);
                    }
                } else {
                    RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
                }
            } else if (start.pose.position.x > goal.pose.position.x) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (waypoints_y_[i] < start.pose.position.x && waypoints_y_[i] >= goal.pose.position.x) {
                    RCLCPP_INFO(logger_, "----검토 중인 waypoint_y_: %.2f", waypoints_y_[i]);
                    double current_diff_y = std::abs(goal.pose.position.x - waypoints_y_[i]); // x
                    if (current_diff_y < diff_y) {
                        diff_y = current_diff_y;
                        y_index = i;
                        RCLCPP_INFO(logger_, "------[y_index 변경: %zu]", y_index);
                    }
                } else {
                    RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
            }
        }
        // 잠시 0.5초 대기 (필요 시 상단에 <thread>와 <chrono> 포함)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        RCLCPP_INFO(logger_, "[for2] waypoints_x_");
        for (size_t i = 0; i < waypoints_x_.size(); ++i)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            RCLCPP_INFO(logger_, "--%zu번째", i);
            if (start.pose.position.y < goal.pose.position.y) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (waypoints_x_[i] > start.pose.position.y && waypoints_x_[i] <= goal.pose.position.y) {
                    RCLCPP_INFO(logger_, "----검토 중인 waypoint_x_: %.2f", waypoints_x_[i]);
                    double current_diff_x = std::abs(goal.pose.position.y - waypoints_x_[i]); // y
                    if (current_diff_x < diff_x) {
                        diff_x = current_diff_x;
                        x_index = i;
                        RCLCPP_INFO(logger_, "------[x_index 변경: %zu]", x_index);
                    }
                } else {
                    RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
                }
            } else if (start.pose.position.y > goal.pose.position.y) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (waypoints_x_[i] < start.pose.position.y && waypoints_x_[i] >= goal.pose.position.y) {
                    RCLCPP_INFO(logger_, "----검토 중인 waypoint_x_: %.2f", waypoints_x_[i]);
                    double current_diff_x = std::abs(goal.pose.position.y - waypoints_x_[i]); // y
                    if (current_diff_x < diff_x) {
                        diff_x = current_diff_x;
                        x_index = i;
                        RCLCPP_INFO(logger_, "------[x_index 변경: %zu]", x_index);
                    }
                } else {
                    RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
            }
        }
        if (x_index == 10 || y_index == 10) {
            RCLCPP_WARN(logger_, "적절한 웨이포인트를 찾지 못했습니다. 기본 경로를 사용합니다.");
            return std::nullopt;
        } else {
            geometry_msgs::msg::PoseStamped waypoint;
            waypoint.pose.position.x = waypoints_y_[y_index];
            waypoint.pose.position.y = waypoints_x_[x_index];
            // waypoint.pose.orientation = goal.pose.orientation;
            RCLCPP_INFO(logger_, "추가된 웨이포인트: (%.2f, %.2f)", 
                waypoint.pose.position.x, waypoint.pose.position.y);
            return waypoint;
        }
    }

    // 1. 출발지에서 가까운 웨이포인트 좌표 찾기
    // 2. 목적지에서 가까운 웨이포인트 좌표 찾기
    // 3. 1과 2 좌표 사이의 웨이포인트 모든 값 찾기
    // 4. 현재 로봇 방향에 맞게 x/y 좌표 인덱스 증가/감소 시키기
    //   -  x/y 좌표 모두 증감 가능한 경우 turnable_index로 좌표 기억해두기
    //   -  이미 turnable_index가 있는 경우 최신 값으로 갱신
    // 5. 4번 반복하면서 경로 생성하다가 x/y 값 모두 증감 불가능할 경우 turnable_index 좌표로 다시 되돌아감
    //   - 이후 좌표는 다 삭제, turnable_index부터 turn하여 다시 4번 반복
    // 6. 목적지에서 가까운 웨이포인트 좌표까지 4,5 반복
    nav_msgs::msg::Path CustomPlanner::createPlan(
        const geometry_msgs::msg::PoseStamped &start,
        const geometry_msgs::msg::PoseStamped &goal,
        std::function<bool()> cancel_checker)
    {
        (void)cancel_checker;
        global_path.header.frame_id = "map";
        global_path.header.stamp = rclcpp::Clock().now();
        
        if (!path_logged) {
            // 경로 생성
            global_path.poses.push_back(start);
            
            // 1. 출발지 좌표 찾기
            if (GetStartWaypoint(start, goal) != std::nullopt) {
                global_path.poses.push_back(*GetStartWaypoint(start, goal));
            }
            // 2. 목적지 좌표 찾기
            if (GetEndWaypoint(start, goal) != std::nullopt) {
                global_path.poses.push_back(*GetEndWaypoint(start, goal));
            }

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
