#ifndef PICKEE_MOBILE_WONHO__CUSTOM_PLANNER_HPP_
#define PICKEE_MOBILE_WONHO__CUSTOM_PLANNER_HPP_

#include <string>
#include <memory>

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

        // void GetNearestWaypoint(
        //     const geometry_msgs::msg::PoseStamped &start,
        //     const geometry_msgs::msg::PoseStamped &goal);

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
            
        bool path_logged;

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
        
        std::vector<double> waypoints_x_;
        std::vector<double> waypoints_y_;
        std::map<std::string, std::pair<double, double>> waypoints_;
    };

} // namespace custom_planner

#endif // PICKEE_MOBILE_WONHO__CUSTOM_PLANNER_HPP_
