#include "pickee_mobile_wonho/custom_planner.hpp"
#include <pluginlib/class_list_macros.hpp>

namespace custom_planner
{

    CustomPlanner::CustomPlanner()
    {
        global_path = nav_msgs::msg::Path();

        path_logged = false;

        // 경로 뒤에서 2번째 포즈의 y좌표 얻기
        second_to_last_y = 0.0;
        second_to_last_x = 0.0;
        // 좁은 문 주행 여부
        is_narrow_passage = false;
        path_after_narrow = startAndGoalPoses();

        // 좁은 문 주행 후 재주행 여부
        is_after_plan = false;
        
        // 각 좌표 x,y 상수 초기화
        x1_ = -2.3;
        x2_ = -1.0;
        x3_ = 0.1;
        x4_ = 0.9;
        x5_ = 1.9;
        x6_ = 3.2;
        x7_ = 3.8;

        y1_ = 7.0;
        y2_ = 4.5;
        y3_ = 2.4;
        y4_ = 1.5;
        y5_ = 0.8;
        y6_ = 0.1;

        waypoints_x_ = {x1_, x2_, x3_, x4_, x5_, x6_, x7_};
        waypoints_y_ = {y1_, y2_, y3_, y4_, y5_, y6_};

        // 웨이포인트 map 상수 초기화
        waypoints_ = {
            {"x1y6", {waypoints_x_[0], waypoints_y_[5]}},
            {"x2y6", {waypoints_x_[1], waypoints_y_[5]}},
            {"x3y6", {waypoints_x_[2], waypoints_y_[5]}},
            {"x4y6", {waypoints_x_[3], waypoints_y_[5]}},
            {"x5y6", {waypoints_x_[4], waypoints_y_[5]}},
            {"x6y6", {waypoints_x_[5], waypoints_y_[5]}}, 
            {"x7y6", {waypoints_x_[6], waypoints_y_[5]}}, // 1행
            {"x1y5", {waypoints_x_[0], waypoints_y_[4]}},
            {"x2y5", {waypoints_x_[1], waypoints_y_[4]}},
            {"x3y5", {waypoints_x_[2], waypoints_y_[4]}},
            {"x5y5", {waypoints_x_[4], waypoints_y_[4]}},
            {"x6y5", {waypoints_x_[5], waypoints_y_[4]}},
            {"x7y5", {waypoints_x_[6], waypoints_y_[4]}}, // 2행
            {"x1y4", {waypoints_x_[0], waypoints_y_[3]}},
            {"x3y4", {waypoints_x_[2], waypoints_y_[3]}},
            {"x4y4", {waypoints_x_[3], waypoints_y_[3]}},
            {"x5y4", {waypoints_x_[4], waypoints_y_[3]}},
            {"x7y4", {waypoints_x_[6], waypoints_y_[3]}}, // 3행
            {"x1y3", {waypoints_x_[0], waypoints_y_[2]}},
            {"x2y3", {waypoints_x_[1], waypoints_y_[2]}},
            {"x3y3", {waypoints_x_[2], waypoints_y_[2]}},
            {"x4y3", {waypoints_x_[3], waypoints_y_[2]}},
            {"x5y3", {waypoints_x_[4], waypoints_y_[2]}},
            {"x6y3", {waypoints_x_[5], waypoints_y_[2]}},
            {"x7y3", {waypoints_x_[6], waypoints_y_[2]}}, // 4행
            {"x1y2", {waypoints_x_[0], waypoints_y_[1]}},
            {"x7y2", {waypoints_x_[6], waypoints_y_[1]}}, // 5행
            {"x1y1", {waypoints_x_[0], waypoints_y_[0]}},
            {"x4y1", {waypoints_x_[3], waypoints_y_[0]}},
            {"x7y1", {waypoints_x_[6], waypoints_y_[0]}}  // 6행
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

            amcl_pose_sub_ = node->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
                "amcl_pose", 10,
                [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                    current_pose = *msg;
            });

            cmd_vel_pub_ = node->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

            nav_to_pose_client_ = rclcpp_action::create_client<nav2_msgs::action::NavigateToPose>(node, "navigate_to_pose");
        }
    }

    void CustomPlanner::ResetCallback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        (void)request;
        
        RCLCPP_INFO(logger_, "CustomPlanner 초기화 시작...");
        // 초기화
        path_logged = false;
        is_after_plan = false; 
        global_path.poses.clear();
        
        if (narrow_passage_timer_) {
            narrow_passage_timer_->cancel();
        }

        RCLCPP_INFO(logger_, "is_narrow_passage 상태: %s", is_narrow_passage ? "true" : "false");

        // 기존 타이머가 있다면 리셋
        if (narrow_passage_timer_) {
            narrow_passage_timer_->cancel();
        }

        // 100ms 주기로 타이머 콜백 실행
        auto node = node_.lock();
        if (node) {
            if (is_narrow_passage) {
                narrow_passage_timer_ = node->create_wall_timer(
                    std::chrono::milliseconds(100),
                    std::bind(&CustomPlanner::NarrowPassageTimerCallback, this)
                );
                // 좁은 문 통과후 다시 주행 시작
                RCLCPP_INFO(logger_, "좁은 문 통과 완료, 마지막 주행 시작");
            }
        }
        
        response->success = true;
        response->message = "CustomPlanner 초기화 완료";
        RCLCPP_INFO(logger_, "CustomPlanner가 초기화되었습니다.");
    }

    void CustomPlanner::NarrowPassageTimerCallback()
    {
        if (!is_narrow_passage) {
            RCLCPP_INFO(logger_, "is_narrow_passage가 false이므로 타이머를 중지합니다.");
            if (narrow_passage_timer_) {
                narrow_passage_timer_->cancel();
            }
            return;
        }

        RCLCPP_INFO(logger_, "좁은 문 통과 중 (타이머 실행)...");
        
        geometry_msgs::msg::Twist cmd_vel_msg;
        double current_x = current_pose.pose.pose.position.x;
        double current_y = current_pose.pose.pose.position.y;
        double current_theta = tf2::getYaw(current_pose.pose.pose.orientation);
        
        RCLCPP_INFO(logger_, "second_to_last_y: %.2f, current_y: %.2f, diff: %.2f",
            second_to_last_y, current_y, std::abs(second_to_last_y - current_y));
        RCLCPP_INFO(logger_, "second_to_last_x: %.2f, current_x: %.2f, diff: %.2f",
            second_to_last_x, current_x, std::abs(second_to_last_x - current_x));
        RCLCPP_INFO(logger_, "current_theta: %.2f radians", current_theta);

        // 목표 도달 조건 확인
        if (std::abs(second_to_last_y - current_y) < 0.3) {
            RCLCPP_INFO(logger_, "목표 지점 도달. 로봇 정지 및 타이머 중지.");
            cmd_vel_msg.linear.x = 0.0;
            cmd_vel_msg.angular.z = 0.0;
            if (cmd_vel_pub_) {
                cmd_vel_pub_->publish(cmd_vel_msg);
            }
            is_narrow_passage = false; // 플래그 리셋
            if (narrow_passage_timer_) {
                narrow_passage_timer_->cancel(); // 타이머 스스로 중지
            }

            // ★★★ 재주행 시작 ★★★
            RCLCPP_INFO(logger_, "재주행 시작: NavigateToPose 목표 전송 시도...");
            if (!nav_to_pose_client_ || !nav_to_pose_client_->wait_for_action_server(std::chrono::seconds(2))) {
                RCLCPP_ERROR(logger_, "NavigateToPose Action 서버를 찾을 수 없습니다!");
                return;
            }

            nav2_msgs::action::NavigateToPose::Goal goal_msg;
            goal_msg.pose = path_after_narrow.goal; // CheckNarrowPassage에서 저장해둔 최종 목표
            
            auto send_goal_options = rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions();
            // 필요하다면 여기서 goal_response, feedback, result 콜백을 설정할 수 있습니다.
            
            nav_to_pose_client_->async_send_goal(goal_msg, send_goal_options);
            RCLCPP_INFO(logger_, "최종 목표 (%.2f, %.2f)로 재주행 요청 전송 완료.", 
                        goal_msg.pose.pose.position.x, goal_msg.pose.pose.position.y);
        } else {
            // 목표 도달 전까지 직진
            cmd_vel_msg.linear.x = 0.15;
            if (current_theta - 0.0 > 0.1) {
                cmd_vel_msg.angular.z = 0.05;
            } else if (current_theta - 0.0 < -0.1) {
                cmd_vel_msg.angular.z = -0.05;
            } else {
                cmd_vel_msg.angular.z = 0.0;
            }
            // cmd_vel_msg.angular.z = 0.0;
            if (cmd_vel_pub_) {
                cmd_vel_pub_->publish(cmd_vel_msg);
            }
        }
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

    // 1. 출발지 좌표 찾기 함수 & 2. 도착지 좌표 찾기 함수 통합
    std::optional<WaypointResult> CustomPlanner::GetStartEndWaypoint(
        const geometry_msgs::msg::PoseStamped &start, 
        const geometry_msgs::msg::PoseStamped &goal,
        const bool is_start)
    {
        double diff_x = 10.0;
        double diff_y = 10.0;
        size_t x_index = 10;
        size_t y_index = 10;

        // waypoints_y_ 순서대로 경유
        if (is_start) {
            RCLCPP_INFO(logger_, "[GetStartWaypoint 시작]start x: %.2f, goal y: %.2f", 
                        start.pose.position.x, goal.pose.position.y);
            // RCLCPP_INFO(logger_, "start y: %.2f, goal y: %.2f",
                        // start.pose.position.y, goal.pose.position.y);
            // RCLCPP_INFO(logger_, "[for1] waypoints_y_");
        } else {
            RCLCPP_INFO(logger_, "[GetEndWaypoint 시작]start x: %.2f, goal y: %.2f", 
                        start.pose.position.x, goal.pose.position.y);
            // RCLCPP_INFO(logger_, "start y: %.2f, goal y: %.2f",
                        // start.pose.position.y, goal.pose.position.y);
            // RCLCPP_INFO(logger_, "[for2] waypoints_y_");
        }
        for (size_t i = 0; i < waypoints_x_.size(); ++i)
        {   
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            // RCLCPP_INFO(logger_, "--%zu번째", i);
            if (start.pose.position.x < goal.pose.position.x) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (waypoints_x_[i] > start.pose.position.x && waypoints_x_[i] <= goal.pose.position.x) {
                    // RCLCPP_INFO(logger_, "----검토 중인 waypoints_x_: %.2f", waypoints_x_[i]);
                    double current_diff_x;
                    if (is_start) {
                        current_diff_x = std::abs(start.pose.position.x - waypoints_x_[i]); // x
                    } else {
                        current_diff_x = std::abs(goal.pose.position.x - waypoints_x_[i]); // x
                    }
                    if (current_diff_x < diff_x) {
                        diff_x = current_diff_x;
                        x_index = i;
                        // RCLCPP_INFO(logger_, "------[x_index 변경: %zu]", x_index);
                    }
                } else {
                    // RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
                }
            } else if (start.pose.position.x > goal.pose.position.x) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (waypoints_x_[i] < start.pose.position.x && waypoints_x_[i] >= goal.pose.position.x) {
                    // RCLCPP_INFO(logger_, "----검토 중인 waypoints_x_: %.2f", waypoints_x_[i]);
                    double current_diff_x;
                    if (is_start) {
                        current_diff_x = std::abs(start.pose.position.x - waypoints_x_[i]); // x
                    } else {
                        current_diff_x = std::abs(goal.pose.position.x - waypoints_x_[i]); // x
                    }
                    if (current_diff_x < diff_x) {
                        diff_x = current_diff_x;
                        x_index = i;
                        // RCLCPP_INFO(logger_, "------[x_index 변경: %zu]", x_index);
                    }
                } else {
                    // RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                // RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
            }
        }

        // waypoints_y_ 순서대로 경유
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        RCLCPP_INFO(logger_, "[for2] waypoints_y_");
        for (size_t i = 0; i < waypoints_y_.size(); ++i)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            // RCLCPP_INFO(logger_, "--%zu번째", i);
            if (start.pose.position.y < goal.pose.position.y) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (waypoints_y_[i] > start.pose.position.y && waypoints_y_[i] <= goal.pose.position.y) {
                    // RCLCPP_INFO(logger_, "----검토 중인 waypoints_y_: %.2f", waypoints_y_[i]);
                    double current_diff_y;
                    if (is_start) {
                        current_diff_y = std::abs(start.pose.position.y - waypoints_y_[i]); // y
                    } else {
                        current_diff_y = std::abs(goal.pose.position.y - waypoints_y_[i]); // y
                    }
                    if (current_diff_y < diff_y) {
                        diff_y = current_diff_y;
                        y_index = i;
                        // RCLCPP_INFO(logger_, "------[y_index 변경: %zu]", y_index);
                    }
                } else {
                    // RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
                }
            } else if (start.pose.position.y > goal.pose.position.y) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (waypoints_y_[i] < start.pose.position.y && waypoints_y_[i] >= goal.pose.position.y) {
                    // RCLCPP_INFO(logger_, "----검토 중인 waypoints_y_: %.2f", waypoints_y_[i]);
                    double current_diff_y;
                    if (is_start) {
                        current_diff_y = std::abs(start.pose.position.y - waypoints_y_[i]); // y
                    } else {
                        current_diff_y = std::abs(goal.pose.position.y - waypoints_y_[i]); // y
                    }
                    if (current_diff_y < diff_y) {
                        diff_y = current_diff_y;
                        y_index = i;
                        // RCLCPP_INFO(logger_, "------[y_index 변경: %zu]", y_index);
                    }
                } else {
                    // RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                // RCLCPP_INFO(logger_, "----검토안됨: %zu", i);
            }
        }
        if (x_index == 10 || y_index == 10) {
            RCLCPP_WARN(logger_, "적절한 웨이포인트를 찾지 못했습니다. 기본 경로를 사용합니다.");
            return std::nullopt;
        } else {
            char key[20];
            snprintf(key, sizeof(key), "x%zuy%zu", x_index + 1, y_index + 1);
            if (waypoints_.find(key) != waypoints_.end()) {
                WaypointResult result;
                geometry_msgs::msg::PoseStamped waypoint;
                waypoint.pose.position.x = waypoints_x_[x_index];
                waypoint.pose.position.y = waypoints_y_[y_index];
                // waypoint.pose.orientation = goal.pose.orientation;
                RCLCPP_INFO(logger_, "추가된 웨이포인트: (%.2f, %.2f)", 
                    waypoint.pose.position.x, waypoint.pose.position.y);
                result.waypoint = waypoint;
                result.x_index = x_index;
                result.y_index = y_index;
                return result;
            } else {
                RCLCPP_WARN(logger_, "웨이포인트 키가 없습니다.");
                return std::nullopt;
            }
            return std::nullopt;
        }
        return std::nullopt;
    }

    // 3. start에서 goal 사이 (Between Waypoint) 경로 생성 함수
    std::optional<std::map<std::string, geometry_msgs::msg::PoseStamped>> CustomPlanner::GetBetweenWaypoints(
        const geometry_msgs::msg::PoseStamped &start, 
        const geometry_msgs::msg::PoseStamped &goal,
        const std::string &startKey,
        const std::string &endKey)
    {
        std::map<std::string, geometry_msgs::msg::PoseStamped> result;
        double min_x = std::min(start.pose.position.x, goal.pose.position.x);
        double max_x = std::max(start.pose.position.x, goal.pose.position.x);
        double min_y = std::min(start.pose.position.y, goal.pose.position.y);
        double max_y = std::max(start.pose.position.y, goal.pose.position.y);
        for (const auto &wp : waypoints_) {
            if (wp.first == startKey || wp.first == endKey) continue;

            double wp_x = wp.second.first;
            double wp_y = wp.second.second;

            if (wp_x >= min_x && wp_x <= max_x &&
                wp_y >= min_y && wp_y <= max_y) {
                geometry_msgs::msg::PoseStamped pose;
                pose.pose.position.x = wp_x;
                pose.pose.position.y = wp_y;
                result[wp.first] = pose;
            }
        }
        if (!result.empty()) {
            RCLCPP_INFO(logger_, "중간 웨이포인트 %zu개 발견", result.size());
            return result;
        }
        return std::nullopt;
    }

    // 4. Between Waypoint에서 필요한 다음 웨이포인트 선택 함수
    void CustomPlanner::MakePathByBetweenWaypoints(
        std::map<std::string, geometry_msgs::msg::PoseStamped> &betweenWaypoints,
        const WaypointIndex &startWaypointIndex,
        const WaypointIndex &endWaypointIndex,
        const geometry_msgs::msg::PoseStamped &start
    )
    {
        size_t turnable_index = 2;
        bool is_adding_waypoint = true;
        bool is_x_dir_increase = endWaypointIndex.x_idx < startWaypointIndex.x_idx;
        bool is_y_dir_increase = endWaypointIndex.y_idx > startWaypointIndex.y_idx;
        RCLCPP_INFO(logger_, "is_x_dir_increase: %s, is_y_dir_increase: %s", 
                    is_x_dir_increase ? "true" : "false",
                    is_y_dir_increase ? "true" : "false");
        RCLCPP_INFO(logger_, "startWaypointIndex: (x: %zu, y: %zu)", 
                    startWaypointIndex.x_idx, startWaypointIndex.y_idx);
        RCLCPP_INFO(logger_, "endWaypointIndex: (x: %zu, y: %zu)", 
                    endWaypointIndex.x_idx, endWaypointIndex.y_idx);

        size_t debug_counter = 0;

        while (is_adding_waypoint) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            RCLCPP_INFO(logger_, "=== 루프 시작 (debug_counter: %zu) ===", debug_counter);
            // 무한 루프 방지
            if (debug_counter > 10) {
                RCLCPP_ERROR(logger_, "무한 루프 방지: 강제 종료");
                is_adding_waypoint = false;
                break;
            }
                // betweenWaypoints가 비었으면 종료
            if (betweenWaypoints.empty()) {
                RCLCPP_INFO(logger_, "중간 웨이포인트 모두 소진. 경로 생성 완료.");
                break;
            }
            RCLCPP_INFO(logger_, "남은 웨이포인트: %zu개", betweenWaypoints.size());

            // 현재 및 이전 웨이포인트 키 찾기
            std::string current_wp_key;
            std::string before_wp_key;
            std::string end_wp_key;
            if (global_path.poses.size() >= 2) {
                const auto &pose = global_path.poses[global_path.poses.size() - 2];
                for (const auto &wp : waypoints_) {
                    if (std::abs(wp.second.first - pose.pose.position.x) < 1e-6 &&
                        std::abs(wp.second.second - pose.pose.position.y) < 1e-6) {
                        current_wp_key = wp.first;
                        break;
                    }
                }
            }
            if (global_path.poses.size() >= 3) {
                const auto &pose = global_path.poses[global_path.poses.size() - 3];
                for (const auto &wp : waypoints_) {
                    if (std::abs(wp.second.first - pose.pose.position.x) < 1e-6 &&
                        std::abs(wp.second.second - pose.pose.position.y) < 1e-6) {
                        before_wp_key = wp.first;
                        break;
                    } else {
                        before_wp_key = "x0y0";
                    }
                }
            }
            if (global_path.poses.size() >= 3) {
                const auto &pose = global_path.poses[global_path.poses.size() - 1];
                for (const auto &wp : waypoints_) {
                    if (std::abs(wp.second.first - pose.pose.position.x) < 1e-6 &&
                        std::abs(wp.second.second - pose.pose.position.y) < 1e-6) {
                        end_wp_key = wp.first;
                        break;
                    }
                } 
            }

            for (const auto &pose : global_path.poses) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                RCLCPP_INFO(logger_, "--현재 경로 웨이포인트: (%.2f, %.2f)", 
                            pose.pose.position.x, pose.pose.position.y);
            }
            // 방향 얻기
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::optional<bool> direction_result = GetProgressedDirection(current_wp_key, before_wp_key, start);
            bool is_progressed_x_value;
            if (direction_result == std::nullopt) {
                RCLCPP_WARN(logger_, "방향 계산 실패, x축으로 기본 설정");
                is_progressed_x_value = true;
            } else {
                is_progressed_x_value = *direction_result;
                RCLCPP_INFO(logger_, "현재 진행 방향: %s", is_progressed_x_value ? "x축" : "y축");
            }

            
            // 다음 웨이포인트 선택
            std::string x_result;
            std::string y_result;
            if (!current_wp_key.empty()) {
                size_t current_x_index = current_wp_key[1] - '0';
                size_t current_y_index = current_wp_key[3] - '0';
                
                RCLCPP_INFO(logger_, "현재 위치: %s (x_idx=%zu, y_idx=%zu)", 
                            current_wp_key.c_str(), current_x_index, current_y_index);
                
                // x 방향 다음 웨이포인트 키 생성
                size_t target_x_index = is_x_dir_increase ? current_x_index + 1 : current_x_index - 1;
                std::string x_target_key = "x" + std::to_string(target_x_index) + "y" + std::to_string(current_y_index);
                
                // y 방향 다음 웨이포인트 키 생성
                size_t target_y_index = is_y_dir_increase ? current_y_index + 1 : current_y_index - 1;
                std::string y_target_key = "x" + std::to_string(current_x_index) + "y" + std::to_string(target_y_index);
                
                RCLCPP_INFO(logger_, "찾을 키 - x방향: %s, y방향: %s", 
                            x_target_key.c_str(), y_target_key.c_str());
                if (x_target_key == end_wp_key) {
                    RCLCPP_INFO(logger_, "x 방향이 도착지와 일치하여 종료합니다.");
                    is_adding_waypoint = false;
                    break;
                } else if (y_target_key == end_wp_key) {
                    RCLCPP_INFO(logger_, "y 방향이 도착지와 일치하여 종료합니다.");
                    is_adding_waypoint = false;
                    break;
                }

                // betweenWaypoints에 해당 키가 있는지 확인
                if (betweenWaypoints.find(x_target_key) != betweenWaypoints.end()) {
                    x_result = x_target_key;
                    RCLCPP_INFO(logger_, "x 방향 웨이포인트 발견: %s", x_result.c_str());
                }
                if (betweenWaypoints.find(y_target_key) != betweenWaypoints.end()) {
                    y_result = y_target_key;
                    RCLCPP_INFO(logger_, "y 방향 웨이포인트 발견: %s", y_result.c_str());
                }
            } else {
                RCLCPP_WARN(logger_, "current_wp_key가 비어있습니다.");
            }
            
            RCLCPP_INFO(logger_, "후보 웨이포인트 - x: %s, y: %s", 
                        x_result.empty() ? "없음" : x_result.c_str(),
                        y_result.empty() ? "없음" : y_result.c_str());
            for (const auto &wp : betweenWaypoints) {
                RCLCPP_INFO(logger_, "--남은 Btw 웨이포인트: %s (x: %.2f, y: %.2f)", 
                            wp.first.c_str(), wp.second.pose.position.x, wp.second.pose.position.y);
            }
            // turnable_index 업데이트
            if (!x_result.empty() && !y_result.empty()) {
                turnable_index = global_path.poses.size();
                RCLCPP_INFO(logger_, "회전 가능 지점 저장: index=%zu", turnable_index);
            }

            bool added = false;
            if (is_progressed_x_value) {
                if (!x_result.empty()) { 
                    global_path.poses.insert(global_path.poses.end() - 1, betweenWaypoints.at(x_result)); 
                    betweenWaypoints.erase(x_result);
                    added = true;
                    RCLCPP_INFO(logger_, "✓ x 좌표 진행: %s 추가됨", x_result.c_str());
                } else if (!y_result.empty()) { 
                    global_path.poses.insert(global_path.poses.end() - 1, betweenWaypoints.at(y_result)); 
                    betweenWaypoints.erase(y_result);
                    is_progressed_x_value = false;
                    added = true;
                    RCLCPP_INFO(logger_, "✓ y 좌표 진행: %s 추가됨", y_result.c_str());
                }
            } else {
                if (!y_result.empty()) { 
                    global_path.poses.insert(global_path.poses.end() - 1, betweenWaypoints.at(y_result)); 
                    betweenWaypoints.erase(y_result);
                    added = true;
                    RCLCPP_INFO(logger_, "✓ y 좌표 진행: %s 추가됨", y_result.c_str());
                } else if (!x_result.empty()) { 
                    global_path.poses.insert(global_path.poses.end() - 1, betweenWaypoints.at(x_result)); 
                    betweenWaypoints.erase(x_result);
                    is_progressed_x_value = true;
                    added = true;
                    RCLCPP_INFO(logger_, "✓ x 좌표 진행: %s 추가됨", x_result.c_str());
                }
            }
            if (added) {
                debug_counter = 0; // 성공 시 카운터 리셋
                continue;
            }

            // 추가 실패한 경우
            RCLCPP_WARN(logger_, "✗ 추가할 웨이포인트 없음");

            if (x_result.empty() && y_result.empty()) {
                if (turnable_index > 2) {
                    RCLCPP_WARN(logger_, "되돌아가기 시도 (turnable_index: %zu)", turnable_index);
                    
                    // 되돌아가기
                    size_t removed_count = 0;
                    while (global_path.poses.size() > turnable_index) {
                        global_path.poses.pop_back();
                        removed_count++;
                    }
                    
                    RCLCPP_INFO(logger_, "%zu개 포즈 제거됨. 현재 경로 크기: %zu", 
                                removed_count, global_path.poses.size());
                    
                    turnable_index = 2;
                    is_progressed_x_value = !is_progressed_x_value;
                    debug_counter++;
                    
                    RCLCPP_WARN(logger_, "방향 전환: %s (재시도: %zu번)", 
                                is_progressed_x_value ? "x축" : "y축", debug_counter);
                    continue;
                } else {
                    RCLCPP_INFO(logger_, "더 이상 되돌아갈 수 없음. 경로 생성 종료.");
                    break;
                }
            }

            debug_counter++;
        }
    }

    // 각 path 사이에 path 촘촘히 추가
    void CustomPlanner::AddDetailedPath()
    {
        for (size_t i = 0; i < global_path.poses.size() - 1; ++i) {
            geometry_msgs::msg::PoseStamped first_pose = global_path.poses[i];
            geometry_msgs::msg::PoseStamped second_pose = global_path.poses[i + 1];

            double dx = second_pose.pose.position.x - first_pose.pose.position.x;
            double dy = second_pose.pose.position.y - first_pose.pose.position.y;
            double distance = std::hypot(dx, dy);

            if (distance > 0.0) {
                size_t num_intermediate_poses = static_cast<size_t>(distance / 0.5); // 0.5m 간격
                for (size_t j = 1; j <= num_intermediate_poses; ++j) {
                    geometry_msgs::msg::PoseStamped intermediate_pose;
                    intermediate_pose.header = global_path.header;
                    intermediate_pose.pose.position.x = first_pose.pose.position.x + (dx * j) / (num_intermediate_poses + 1);
                    intermediate_pose.pose.position.y = first_pose.pose.position.y + (dy * j) / (num_intermediate_poses + 1);
                    global_path.poses.insert(global_path.poses.begin() + i + j, intermediate_pose);
                }
                i += num_intermediate_poses; // 인덱스 조정
            }
        }
    }

    // 진행 방향 얻기 함수
    void CustomPlanner::AddOrientationToPath()
    {
        geometry_msgs::msg::PoseStamped before_pose;
        for (size_t i = 0; i < global_path.poses.size(); ++i) {
            geometry_msgs::msg::PoseStamped first_pose;
            geometry_msgs::msg::PoseStamped second_pose;

            if (is_narrow_passage && i == global_path.poses.size() - 1) {
                global_path.poses[i] = before_pose;
                break;
            }
            if (i + 1 >= global_path.poses.size()) break;
            first_pose = global_path.poses[i];
            second_pose = global_path.poses[i + 1];


            double dx = second_pose.pose.position.x - first_pose.pose.position.x;
            double dy = second_pose.pose.position.y - first_pose.pose.position.y;
            double yaw = std::atan2(dy, dx);

            // set planar quaternion (x,y = 0)
            first_pose.pose.orientation.x = 0.0;
            first_pose.pose.orientation.y = 0.0;
            first_pose.pose.orientation.z = std::sin(yaw * 0.5);
            first_pose.pose.orientation.w = std::cos(yaw * 0.5);

            // write back to path and log
            global_path.poses[i] = first_pose;
            before_pose = first_pose;
            RCLCPP_INFO(logger_, "Pose %zu -> %zu yaw: %.4f rad (%.2f deg)", i, i+1, yaw, yaw * 180.0 / M_PI);
        }
    }

    std::optional<bool> CustomPlanner::GetProgressedDirection(const std::string &current_wp_key, const std::string &before_wp_key, const geometry_msgs::msg::PoseStamped &start)
    {
        bool is_progressed_x = true;
        
        if (current_wp_key.empty() || before_wp_key.empty()) {
            RCLCPP_WARN(logger_, "현재 또는 이전 웨이포인트 키를 찾지 못했습니다.");
            return std::nullopt;
        }
        auto before = std::make_pair(0.0, 0.0);
        if (before_wp_key == "x0y0") {
            before = {start.pose.position.x, start.pose.position.y};
        } else {
            before = waypoints_[before_wp_key];
        }
        auto current = waypoints_[current_wp_key];
        double dx = current.first - before.first;
        double dy = current.second - before.second;
        double yaw = std::atan2(dy, dx);
        RCLCPP_INFO(logger_, "yaw: %.4f", yaw);

        // 방향 얻기
        if (yaw >= -M_PI / 4 && yaw < M_PI / 4) { is_progressed_x = true; } 
        else if (yaw >= M_PI / 4 && yaw < 3 * M_PI / 4) { is_progressed_x = false; } 
        else if (yaw >= -3 * M_PI / 4 && yaw < -M_PI / 4) { is_progressed_x = false; } 
        else { is_progressed_x = true; }

        return is_progressed_x;
    }

    void CustomPlanner::CheckNarrowPassage()
    {
        // 경로중 x7y2나 x7y1이 동시에 포함되어 있거나, x1y2나 x1y1이 동시에 포함되어 있을 경우
        bool has_x7y2 = false;
        bool has_x7y1 = false;
        bool has_x1y2 = false;
        bool has_x1y1 = false;
        for (const auto &pose : global_path.poses) {
            if (std::abs(pose.pose.position.y - y2_) < 1e-6 && std::abs(pose.pose.position.x - x7_) < 1e-6) {
                has_x7y2 = true;
            }
            if (std::abs(pose.pose.position.y - y1_) < 1e-6 && std::abs(pose.pose.position.x - x7_) < 1e-6) {
                has_x7y1 = true;
            }
            if (std::abs(pose.pose.position.y - y2_) < 1e-6 && std::abs(pose.pose.position.x - x1_) < 1e-6) {
                has_x1y2 = true;
            }
            if (std::abs(pose.pose.position.y - y1_) < 1e-6 && std::abs(pose.pose.position.x - x1_) < 1e-6) {
                has_x1y1 = true;
            }
        }
        // 경로 뒤에서 2번째 포즈의 y좌표 얻기
        second_to_last_y = global_path.poses[global_path.poses.size() - 2].pose.position.y;
        second_to_last_x = global_path.poses[global_path.poses.size() - 2].pose.position.x;
        path_after_narrow.start = global_path.poses[global_path.poses.size() - 2];
        path_after_narrow.goal = global_path.poses[global_path.poses.size() - 1];
        // 경로 뒤에서 2개 포즈 제거
        if (has_x7y2 && has_x7y1) {
            RCLCPP_INFO(logger_, "경로에 x7y2와 x7y1이 모두 포함되어 있어 x7y1 제거");
            global_path.poses.pop_back();
            global_path.poses.pop_back();
            is_narrow_passage = true;
        } else if (has_x1y2 && has_x1y1) {
            RCLCPP_INFO(logger_, "경로에 x1y2와 x1y1이 모두 포함되어 있어 x1y1 제거");
            global_path.poses.pop_back();
            global_path.poses.pop_back();
            is_narrow_passage = true;
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

            WaypointIndex startWaypointIndex;
            WaypointIndex endWaypointIndex;

            // 1. 출발지 좌표 찾기
            std::optional<WaypointResult> startResult = GetStartEndWaypoint(start, goal, true);
            if (startResult != std::nullopt) {
                startWaypointIndex.x_idx = startResult->x_index;
                startWaypointIndex.y_idx = startResult->y_index;
                global_path.poses.push_back(startResult->waypoint);
            }
            // 2. 목적지 좌표 찾기
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::optional<WaypointResult> endResult = GetStartEndWaypoint(start, goal, false);
            if (endResult != std::nullopt) {
                endWaypointIndex.x_idx = endResult->x_index;
                endWaypointIndex.y_idx = endResult->y_index;
                global_path.poses.push_back(endResult->waypoint);
            }

            // 3. 중간 웨이포인트 좌표 찾기
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::string startKey = 
                "x" + std::to_string(startWaypointIndex.x_idx + 1) + 
                "y" + std::to_string(startWaypointIndex.y_idx + 1);
            std::string endKey = 
                "x" + std::to_string(endWaypointIndex.x_idx + 1) + 
                "y" + std::to_string(endWaypointIndex.y_idx + 1);
            std::optional<std::map<std::string, geometry_msgs::msg::PoseStamped>> betweenWaypoints = GetBetweenWaypoints(start, goal, startKey, endKey);

            // 4. 경로에 중간 웨이포인트 추가
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::optional<std::map<std::string, geometry_msgs::msg::PoseStamped>> betweenPath;
            if (betweenWaypoints != std::nullopt) {
                MakePathByBetweenWaypoints(*betweenWaypoints, startWaypointIndex, endWaypointIndex, start);
            }

            // 마지막 목적지 좌표 추가
            global_path.poses.push_back(goal);

            // 좁은길인지 체크 후 경로 수정
            CheckNarrowPassage();

            // 경로의 각 포즈에 올바른 오리엔테이션(방향) 설정
            AddDetailedPath();
            AddOrientationToPath();
            
            RCLCPP_INFO(logger_, "Create Plan 호출. 경로 생성됨 (총 %zu 개의 포즈)", global_path.poses.size());
            RCLCPP_INFO(logger_, "생성된 경로: ");
            for (const auto &pose : global_path.poses) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                RCLCPP_INFO(logger_, "-> (x: %.2f, y: %.2f)", 
                            pose.pose.position.x, pose.pose.position.y);
            }
            path_logged = true;
            return global_path;
        }
        
        if (is_after_plan) {
            RCLCPP_INFO(logger_, "재주행 시작 start: (%.2f, %.2f), goal: (%.2f, %.2f)", 
                        start.pose.position.x, start.pose.position.y,
                        goal.pose.position.x, goal.pose.position.y);
            global_path.poses.clear();
            global_path.poses.push_back(start);
            global_path.poses.push_back(goal);
            is_after_plan = false;
            return global_path;
        }

        return global_path;
    }

} // namespace custom_planner

// 플러그인 등록
PLUGINLIB_EXPORT_CLASS(custom_planner::CustomPlanner, nav2_core::GlobalPlanner)
