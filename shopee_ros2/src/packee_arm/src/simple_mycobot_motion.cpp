#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

using namespace std::chrono_literals;

class MyCobotRemoteController : public rclcpp::Node
{
public:
  MyCobotRemoteController() : Node("mycobot_remote_controller"), step_(0)
  {
    pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/joint_trajectory_controller/joint_trajectory", 10);

    timer_ = this->create_wall_timer(3s, std::bind(&MyCobotRemoteController::next_step, this));
    RCLCPP_INFO(this->get_logger(), "✅ JetCobot 원격 제어 시작");
  }

private:
  void next_step()
  {
    trajectory_msgs::msg::JointTrajectory traj;
    traj.joint_names = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};

    trajectory_msgs::msg::JointTrajectoryPoint p;

    if (step_ == 0) {
      RCLCPP_INFO(this->get_logger(), "홈 자세 이동");
      p.positions = {0, 0, 0, 0, 0, 0};
    } else if (step_ == 1) {
      RCLCPP_INFO(this->get_logger(), "앞으로 숙이기");
      p.positions = {0.0, -0.5, 0.3, 0.0, 0.5, 0.0};
    } else if (step_ == 2) {
      RCLCPP_INFO(this->get_logger(), "복귀");
      p.positions = {0, 0, 0, 0, 0, 0};
    } else {
      RCLCPP_INFO(this->get_logger(), "✅ 종료");
      rclcpp::shutdown();
      return;
    }

    p.time_from_start = rclcpp::Duration::from_seconds(3.0);
    traj.points.push_back(p);
    pub_->publish(traj);
    step_++;
  }

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  int step_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MyCobotRemoteController>());
  rclcpp::shutdown();
  return 0;
}
