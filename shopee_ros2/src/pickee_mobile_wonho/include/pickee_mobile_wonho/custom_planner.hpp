#ifndef PICKEE_MOBILE_WONHO__CUSTOM_PLANNER_HPP_
#define PICKEE_MOBILE_WONHO__CUSTOM_PLANNER_HPP_

#include <string>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <nav2_core/global_planner.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/buffer.h>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <rclcpp_action/rclcpp_action.hpp>          
#include <nav2_msgs/action/navigate_to_pose.hpp>

namespace custom_planner
{

    struct WaypointResult {
        geometry_msgs::msg::PoseStamped waypoint;
        size_t x_index;
        size_t y_index;
    };
    struct WaypointIndex {
        size_t x_idx;
        size_t y_idx;
    };
    struct NextWaypointResult {
        geometry_msgs::msg::PoseStamped waypoint;
        std::string key;
        size_t turnable_index;
    };
    struct startAndGoalPoses {
        geometry_msgs::msg::PoseStamped start;
        geometry_msgs::msg::PoseStamped goal;
    };
    class CustomPlanner : public nav2_core::GlobalPlanner
    {
    public:
        CustomPlanner();
        ~CustomPlanner() = default;

        void configure(
            const rclcpp_lifecycle::LifecycleNode::WeakPtr &parent,
            std::string name,
            std::shared_ptr<tf2_ros::Buffer> tf,
            std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

        void cleanup() override;

        void activate() override;

        void deactivate() override;
        
        std::optional<WaypointResult> GetStartEndWaypoint(
            const geometry_msgs::msg::PoseStamped &start,
            const geometry_msgs::msg::PoseStamped &goal,
            const bool is_start);

        std::optional<std::map<std::string, geometry_msgs::msg::PoseStamped>> GetBetweenWaypoints(
            const geometry_msgs::msg::PoseStamped &start, 
            const geometry_msgs::msg::PoseStamped &goal,
            const std::string &startKey,
            const std::string &endKey);

        void MakePathByBetweenWaypoints(
            std::map<std::string, geometry_msgs::msg::PoseStamped> &betweenWaypoints,
            const WaypointIndex &startWaypointIndex,
            const WaypointIndex &endWaypointIndex,
            const geometry_msgs::msg::PoseStamped &start);

        void AddDetailedPath();
        void AddOrientationToPath();

        std::optional<bool> GetProgressedDirection(
            const std::string &current_wp_key,
            const std::string &before_wp_key,
            const geometry_msgs::msg::PoseStamped &start);

        nav_msgs::msg::Path createPlan(
            const geometry_msgs::msg::PoseStamped &start,
            const geometry_msgs::msg::PoseStamped &goal,
            std::function<bool()> cancel_checker) override;

    private:
        rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
        std::shared_ptr<tf2_ros::Buffer> tf_;
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
        std::string name_;
        rclcpp::Logger logger_{rclcpp::get_logger("CustomPlanner")};
        
        rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
        void ResetCallback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
            std::shared_ptr<std_srvs::srv::Trigger::Response> response);
        void NarrowPassageTimerCallback();
        void CheckNarrowPassage();
        
        bool path_logged;
        
        double second_to_last_x;
        double second_to_last_y;
        bool is_narrow_passage;
        rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SharedPtr nav_to_pose_client_;
        startAndGoalPoses path_after_narrow;
        bool is_after_plan;
        bool aruco_mode;
        nav_msgs::msg::Path global_path;
        
        double x1_;
        double x2_;
        double x3_;
        double x4_;
        double x5_;
        double x6_;
        double x7_;

        double y1_;
        double y2_;
        double y3_;
        double y4_;
        double y5_;
        double y6_;
        
        std::vector<double> waypoints_x_;
        std::vector<double> waypoints_y_;
        std::map<std::string, std::pair<double, double>> waypoints_;

        // 포즈
        rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_pose_sub_;
        geometry_msgs::msg::PoseWithCovarianceStamped current_pose;
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
        rclcpp::TimerBase::SharedPtr narrow_passage_timer_;
    };

} // namespace custom_planner

#endif // PICKEE_MOBILE_WONHO__CUSTOM_PLANNER_HPP_
