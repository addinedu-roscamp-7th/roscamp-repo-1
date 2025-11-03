#ifndef PICKEE_MOBILE_WONHO__CUSTOM_PLANNER_HPP_
#define PICKEE_MOBILE_WONHO__CUSTOM_PLANNER_HPP_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <nav2_core/global_planner.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/buffer.h>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace custom_planner
{

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

        nav_msgs::msg::Path createPlan(
            const geometry_msgs::msg::PoseStamped &start,
            const geometry_msgs::msg::PoseStamped &goal,
            std::function<bool()> cancel_checker) override;

    private:
        struct Waypoint
        {
            std::string id;
            double x;
            double y;
            std::vector<std::string> neighbours;
        };

        rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
        std::shared_ptr<tf2_ros::Buffer> tf_;
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
        std::string name_;
        rclcpp::Logger logger_{rclcpp::get_logger("CustomPlanner")};

        rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
        void ResetCallback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
            std::shared_ptr<std_srvs::srv::Trigger::Response> response);

        nav_msgs::msg::Path cached_path_;
        geometry_msgs::msg::PoseStamped cached_start_;
        geometry_msgs::msg::PoseStamped cached_goal_;
        bool has_cached_path_{false};

        std::unordered_map<std::string, Waypoint> waypoint_graph_;

        void initialise_waypoints();
        const Waypoint *find_closest_waypoint(const geometry_msgs::msg::PoseStamped &pose,
            std::string &waypoint_id) const;
        std::vector<std::string> compute_route(const std::string &from,
            const std::string &to) const;
        geometry_msgs::msg::PoseStamped build_pose(const std::string &frame,
            const rclcpp::Time &stamp, double x, double y) const;
        void attach_headings(nav_msgs::msg::Path &path,
            const geometry_msgs::msg::PoseStamped &goal) const;
        bool poses_are_close(const geometry_msgs::msg::PoseStamped &lhs,
            const geometry_msgs::msg::PoseStamped &rhs, double position_tol,
            double yaw_tol) const;
    };

} // namespace custom_planner

#endif // PICKEE_MOBILE_WONHO__CUSTOM_PLANNER_HPP_
