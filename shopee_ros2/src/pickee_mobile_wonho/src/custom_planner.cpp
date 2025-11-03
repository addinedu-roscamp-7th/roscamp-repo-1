#include "pickee_mobile_wonho/custom_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>

#include <pluginlib/class_list_macros.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace custom_planner
{

namespace
{
constexpr double kCachePositionTolerance = 0.05;  // metres
constexpr double kCacheYawTolerance = 0.10;        // radians
constexpr double kPoseMergeTolerance = 1e-3;
}

CustomPlanner::CustomPlanner()
{
    initialise_waypoints();
}

void CustomPlanner::configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr &parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
    node_ = parent;
    name_ = std::move(name);
    tf_ = std::move(tf);
    costmap_ros_ = std::move(costmap_ros);

    if (auto node = node_.lock())
    {
        logger_ = node->get_logger();
        RCLCPP_INFO(logger_, "CustomPlanner configured with %zu waypoint crossings.",
            waypoint_graph_.size());

        reset_service_ = node->create_service<std_srvs::srv::Trigger>(
            "custom_planner/reset",
            std::bind(&CustomPlanner::ResetCallback, this,
                std::placeholders::_1, std::placeholders::_2));
    }

    has_cached_path_ = false;
    cached_path_.poses.clear();
}

void CustomPlanner::ResetCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    (void)request;

    has_cached_path_ = false;
    cached_path_.poses.clear();

    response->success = true;
    response->message = "CustomPlanner cache cleared";

    RCLCPP_INFO(logger_, "CustomPlanner cache reset via trigger service.");
}

void CustomPlanner::cleanup()
{
    RCLCPP_INFO(logger_, "CustomPlanner cleanup invoked.");
}

void CustomPlanner::activate()
{
    RCLCPP_INFO(logger_, "CustomPlanner activated.");
}

void CustomPlanner::deactivate()
{
    RCLCPP_INFO(logger_, "CustomPlanner deactivated.");
}

nav_msgs::msg::Path CustomPlanner::createPlan(
    const geometry_msgs::msg::PoseStamped &start,
    const geometry_msgs::msg::PoseStamped &goal,
    std::function<bool()> cancel_checker)
{
    (void)cancel_checker;

    nav_msgs::msg::Path plan;
    plan.header.frame_id = costmap_ros_->getGlobalFrameID();
    if (auto node = node_.lock())
    {
        plan.header.stamp = node->now();
    }
    else
    {
        plan.header.stamp = rclcpp::Time();
    }

    if (has_cached_path_ &&
        poses_are_close(start, cached_start_, kCachePositionTolerance, kCacheYawTolerance) &&
        poses_are_close(goal, cached_goal_, kCachePositionTolerance, kCacheYawTolerance))
    {
        RCLCPP_DEBUG(logger_, "CustomPlanner returning cached path.");
        return cached_path_;
    }

    plan.poses.reserve(8);
    plan.poses.push_back(start);

    std::string start_wp_id;
    std::string goal_wp_id;
    const auto *start_wp = find_closest_waypoint(start, start_wp_id);
    const auto *goal_wp = find_closest_waypoint(goal, goal_wp_id);

    std::vector<std::string> route;

    if (start_wp && goal_wp)
    {
        route = compute_route(start_wp_id, goal_wp_id);
    }

    auto append_if_distinct = [this, &plan](double x, double y)
    {
        const auto &last_pose = plan.poses.back();
        const double dx = last_pose.pose.position.x - x;
        const double dy = last_pose.pose.position.y - y;
        if (std::fabs(dx) < kPoseMergeTolerance && std::fabs(dy) < kPoseMergeTolerance)
        {
            return;
        }
        plan.poses.emplace_back(build_pose(plan.header.frame_id, plan.header.stamp, x, y));
    };

    if (!route.empty())
    {
        const auto &first_wp = waypoint_graph_.at(route.front());
        append_if_distinct(first_wp.x, start.pose.position.y);
        append_if_distinct(first_wp.x, first_wp.y);

        for (size_t i = 1; i + 1 < route.size(); ++i)
        {
            const auto &mid_wp = waypoint_graph_.at(route[i]);
            append_if_distinct(mid_wp.x, mid_wp.y);
        }

        const auto &last_wp = waypoint_graph_.at(route.back());
        append_if_distinct(last_wp.x, last_wp.y);
        append_if_distinct(last_wp.x, goal.pose.position.y);
    }
    else if (start_wp)
    {
        append_if_distinct(start_wp->x, start.pose.position.y);
        append_if_distinct(start_wp->x, start_wp->y);
        append_if_distinct(start_wp->x, goal.pose.position.y);
    }

    plan.poses.push_back(goal);

    attach_headings(plan, goal);

    cached_path_ = plan;
    cached_start_ = start;
    cached_goal_ = goal;
    has_cached_path_ = true;

    RCLCPP_INFO(logger_, "CustomPlanner created path with %zu poses.", plan.poses.size());

    return plan;
}

void CustomPlanner::initialise_waypoints()
{
    waypoint_graph_.clear();

    const std::vector<double> xs = {-3.5, -2.0, -1.1, -0.195, 0.98, 2.25, 3.13};
    const std::vector<double> ys = {1.6, 0.83, 0.16, -1.0, -4.95};

    for (size_t row = 0; row < ys.size(); ++row)
    {
        for (size_t col = 0; col < xs.size(); ++col)
        {
            Waypoint wp;
            wp.id = "wp_" + std::to_string(row) + "_" + std::to_string(col);
            wp.x = xs[col];
            wp.y = ys[row];

            if (row > 0)
            {
                wp.neighbours.emplace_back("wp_" + std::to_string(row - 1) + "_" + std::to_string(col));
            }
            if (row + 1 < ys.size())
            {
                wp.neighbours.emplace_back("wp_" + std::to_string(row + 1) + "_" + std::to_string(col));
            }
            if (col > 0)
            {
                wp.neighbours.emplace_back("wp_" + std::to_string(row) + "_" + std::to_string(col - 1));
            }
            if (col + 1 < xs.size())
            {
                wp.neighbours.emplace_back("wp_" + std::to_string(row) + "_" + std::to_string(col + 1));
            }

            waypoint_graph_.emplace(wp.id, std::move(wp));
        }
    }
}

const CustomPlanner::Waypoint *CustomPlanner::find_closest_waypoint(
    const geometry_msgs::msg::PoseStamped &pose,
    std::string &waypoint_id) const
{
    const Waypoint *closest = nullptr;
    double best_distance = std::numeric_limits<double>::max();
    waypoint_id.clear();

    for (const auto &entry : waypoint_graph_)
    {
        const auto &candidate = entry.second;
        const double dx = pose.pose.position.x - candidate.x;
        const double dy = pose.pose.position.y - candidate.y;
        const double distance = std::hypot(dx, dy);
        if (distance < best_distance)
        {
            best_distance = distance;
            closest = &candidate;
            waypoint_id = entry.first;
        }
    }

    return closest;
}

std::vector<std::string> CustomPlanner::compute_route(const std::string &from,
    const std::string &to) const
{
    if (from == to)
    {
        return {from};
    }

    std::queue<std::string> queue;
    std::unordered_map<std::string, std::string> parent;

    queue.push(from);
    parent[from] = {};

    while (!queue.empty())
    {
        const auto current = queue.front();
        queue.pop();

        if (current == to)
        {
            break;
        }

        const auto &node = waypoint_graph_.at(current);
        for (const auto &neighbour_id : node.neighbours)
        {
            if (!waypoint_graph_.count(neighbour_id) || parent.count(neighbour_id))
            {
                continue;
            }
            parent[neighbour_id] = current;
            queue.push(neighbour_id);
        }
    }

    if (!parent.count(to))
    {
        return {};
    }

    std::vector<std::string> route;
    for (std::string current = to; !current.empty(); current = parent[current])
    {
        route.push_back(current);
    }
    std::reverse(route.begin(), route.end());
    return route;
}

geometry_msgs::msg::PoseStamped CustomPlanner::build_pose(const std::string &frame,
    const rclcpp::Time &stamp, double x, double y) const
{
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = frame;
    pose.header.stamp = stamp;
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, 0.0);
    pose.pose.orientation = tf2::toMsg(q);
    return pose;
}

void CustomPlanner::attach_headings(nav_msgs::msg::Path &path,
    const geometry_msgs::msg::PoseStamped &goal) const
{
    if (path.poses.size() < 2)
    {
        return;
    }

    path.poses.back().pose.orientation = goal.pose.orientation;

    for (size_t i = 1; i + 1 < path.poses.size(); ++i)
    {
        auto &current = path.poses[i];
        const auto &next = path.poses[i + 1];
        const double dx = next.pose.position.x - current.pose.position.x;
        const double dy = next.pose.position.y - current.pose.position.y;
        if (std::fabs(dx) < kPoseMergeTolerance && std::fabs(dy) < kPoseMergeTolerance)
        {
            continue;
        }
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, std::atan2(dy, dx));
        current.pose.orientation = tf2::toMsg(q);
    }
}

bool CustomPlanner::poses_are_close(const geometry_msgs::msg::PoseStamped &lhs,
    const geometry_msgs::msg::PoseStamped &rhs, double position_tol,
    double yaw_tol) const
{
    const double dx = lhs.pose.position.x - rhs.pose.position.x;
    const double dy = lhs.pose.position.y - rhs.pose.position.y;
    const double distance = std::hypot(dx, dy);
    if (distance > position_tol)
    {
        return false;
    }

    const double lhs_yaw = tf2::getYaw(lhs.pose.orientation);
    const double rhs_yaw = tf2::getYaw(rhs.pose.orientation);
    const double yaw_diff = std::fabs(std::atan2(std::sin(lhs_yaw - rhs_yaw),
        std::cos(lhs_yaw - rhs_yaw)));

    return yaw_diff <= yaw_tol;
}

} // namespace custom_planner

PLUGINLIB_EXPORT_CLASS(custom_planner::CustomPlanner, nav2_core::GlobalPlanner)
