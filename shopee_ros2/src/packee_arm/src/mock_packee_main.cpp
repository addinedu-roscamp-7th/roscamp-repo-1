#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/msg/arm_pose_status.hpp"
#include "shopee_interfaces/msg/b_box.hpp"
#include "shopee_interfaces/msg/arm_task_status.hpp"
#include "shopee_interfaces/msg/pose6_d.hpp"
#include "shopee_interfaces/msg/detected_product.hpp"
#include "shopee_interfaces/srv/arm_move_to_pose.hpp"
#include "shopee_interfaces/srv/arm_pick_product.hpp"
#include "shopee_interfaces/srv/arm_place_product.hpp"

using namespace std::chrono_literals;

namespace detail {

template<typename T, typename = void>
struct HasPoseMember : std::false_type {};

template<typename T>
struct HasPoseMember<
  T,
  std::void_t<
    decltype(std::declval<T>().pose.joint_1),
    decltype(std::declval<T>().pose.joint_2),
    decltype(std::declval<T>().pose.joint_3),
    decltype(std::declval<T>().pose.joint_4)>> : std::true_type {};

template<typename T, typename = void>
struct HasPositionMember : std::false_type {};

template<typename T>
struct HasPositionMember<
  T,
  std::void_t<
    decltype(std::declval<T>().position.x),
    decltype(std::declval<T>().position.y),
    decltype(std::declval<T>().position.z)>> : std::true_type {};

template<typename T, typename = void>
struct HasBoxPositionMember : std::false_type {};

template<typename T>
struct HasBoxPositionMember<
  T,
  std::void_t<
    decltype(std::declval<T>().box_position.x),
    decltype(std::declval<T>().box_position.y),
    decltype(std::declval<T>().box_position.z)>> : std::true_type {};

}  // namespace detail

template<typename DetectedProductT>
void AssignDetectedProductPose(
  DetectedProductT * product,
  float x,
  float y,
  float z,
  float yaw_deg) {
  if constexpr (detail::HasPoseMember<DetectedProductT>::value) {
    product->pose.joint_1 = x;
    product->pose.joint_2 = y;
    product->pose.joint_3 = z;
    product->pose.joint_4 = yaw_deg;
    product->pose.joint_5 = 0.0F;
    product->pose.joint_6 = 0.0F;
  } else if constexpr (detail::HasPositionMember<DetectedProductT>::value) {
    product->position.x = x;
    product->position.y = y;
    product->position.z = z;
  }
}

template<typename RequestT>
void AssignPlacePose(
  RequestT * request,
  float x,
  float y,
  float z,
  float yaw_deg) {
  if constexpr (detail::HasPoseMember<RequestT>::value) {
    request->pose.joint_1 = x;
    request->pose.joint_2 = y;
    request->pose.joint_3 = z;
    request->pose.joint_4 = yaw_deg;
    request->pose.joint_5 = 0.0F;
    request->pose.joint_6 = 0.0F;
  } else if constexpr (detail::HasBoxPositionMember<RequestT>::value) {
    request->box_position.x = x;
    request->box_position.y = y;
    request->box_position.z = z;
  }
}

class MockPackeeMain : public rclcpp::Node {
public:
  MockPackeeMain()
  : rclcpp::Node("mock_packee_main"),
    state_("wait_services"),
    current_arm_index_(0)
  {
    // 파라미터 선언 및 로드
    this->declare_parameter<int>("robot_id", 1);
    this->declare_parameter<int>("order_id", 100);
    this->declare_parameter<int>("product_id", 501);
    this->declare_parameter<std::string>("arm_side", "left");
    this->declare_parameter<std::string>("arm_sides", "left,right");

    robot_id_ = this->get_parameter("robot_id").as_int();
    order_id_ = this->get_parameter("order_id").as_int();
    base_product_id_ = this->get_parameter("product_id").as_int();
    default_arm_side_ = this->get_parameter("arm_side").as_string();

    ParseArmSides(this->get_parameter("arm_sides").as_string());
    if (arm_sides_.empty()) arm_sides_.push_back(default_arm_side_);

    // ---- 서비스 클라이언트 등록 ----
    move_client_ = this->create_client<shopee_interfaces::srv::ArmMoveToPose>("/packee/arm/move_to_pose");
    pick_client_ = this->create_client<shopee_interfaces::srv::ArmPickProduct>("/packee/arm/pick_product");
    place_client_ = this->create_client<shopee_interfaces::srv::ArmPlaceProduct>("/packee/arm/place_product");

    // ---- 상태 구독 ----
    pose_status_sub_ = this->create_subscription<shopee_interfaces::msg::ArmPoseStatus>(
      "/packee/arm/pose_status", 10,
      std::bind(&MockPackeeMain::OnPoseStatus, this, std::placeholders::_1));

    pick_status_sub_ = this->create_subscription<shopee_interfaces::msg::ArmTaskStatus>(
      "/packee/arm/pick_status", 10,
      std::bind(&MockPackeeMain::OnPickStatus, this, std::placeholders::_1));

    place_status_sub_ = this->create_subscription<shopee_interfaces::msg::ArmTaskStatus>(
      "/packee/arm/place_status", 10,
      std::bind(&MockPackeeMain::OnPlaceStatus, this, std::placeholders::_1));

    // ---- 상태 루프 ----
    timer_ = this->create_wall_timer(200ms, std::bind(&MockPackeeMain::ProcessSteps, this));

    RCLCPP_INFO(this->get_logger(),
      "Packee Arm 통신 모의 테스트를 시작합니다. robot_id=%d, order_id=%d, product_id 시작값=%d, arm_sides=%s",
      robot_id_, order_id_, base_product_id_, ArmSidesToString().c_str());
  }

private:
  using MoveClient = rclcpp::Client<shopee_interfaces::srv::ArmMoveToPose>;
  using PickClient = rclcpp::Client<shopee_interfaces::srv::ArmPickProduct>;
  using PlaceClient = rclcpp::Client<shopee_interfaces::srv::ArmPlaceProduct>;
  using MoveFuture = MoveClient::SharedFuture;
  using PickFuture = PickClient::SharedFuture;
  using PlaceFuture = PlaceClient::SharedFuture;
  using FutureVariant = std::variant<std::monostate, MoveFuture, PickFuture, PlaceFuture>;

  // -------- 유틸 함수들 --------
  void ParseArmSides(const std::string & raw_value) {
    std::string token;
    for (char ch : raw_value) {
      if (ch == ',') { AppendArmSide(token); token.clear(); }
      else token.push_back(ch);
    }
    AppendArmSide(token);
  }

  void AppendArmSide(const std::string & value) {
    const auto trimmed = Trim(value);
    if (!trimmed.empty()) arm_sides_.push_back(trimmed);
  }

  std::string Trim(const std::string & value) const {
    const auto begin = value.find_first_not_of(" \t");
    if (begin == std::string::npos) return "";
    const auto end = value.find_last_not_of(" \t");
    return value.substr(begin, end - begin + 1);
  }

  std::string ArmSidesToString() const {
    std::string result;
    for (size_t i = 0; i < arm_sides_.size(); ++i) {
      if (i > 0) result += ",";
      result += arm_sides_[i];
    }
    return result;
  }

  // -------- 상태 콜백 --------
  void OnPoseStatus(const shopee_interfaces::msg::ArmPoseStatus::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(),
      "[PoseStatus] pose_type=%s, status=%s, progress=%.2f, message=%s",
      msg->pose_type.c_str(), msg->status.c_str(), msg->progress, msg->message.c_str());
  }

  void OnPickStatus(const shopee_interfaces::msg::ArmTaskStatus::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(),
      "[PickStatus] product_id=%d, phase=%s, status=%s, progress=%.2f, message=%s",
      msg->product_id, msg->current_phase.c_str(), msg->status.c_str(),
      msg->progress, msg->message.c_str());
  }

  void OnPlaceStatus(const shopee_interfaces::msg::ArmTaskStatus::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(),
      "[PlaceStatus] product_id=%d, phase=%s, status=%s, progress=%.2f, message=%s",
      msg->product_id, msg->current_phase.c_str(), msg->status.c_str(),
      msg->progress, msg->message.c_str());
  }

  // -------- 상태 기계 --------
  void ProcessSteps() {
    if (state_ == "wait_services") {
      if (move_client_->service_is_ready() && pick_client_->service_is_ready() && place_client_->service_is_ready()) {
        RCLCPP_INFO(this->get_logger(), "Packee Arm 서비스가 준비되었습니다. 자세 변경을 요청합니다.");
        SendMoveRequest();
        state_ = "await_move";
      }
    } else if (state_ == "await_move") {
      HandleFuture("자세 변경");
    } else if (state_ == "request_pick") {
      if (current_arm_index_ >= arm_sides_.size()) state_ = "completed";
      else { SendPickRequest(); state_ = "await_pick"; }
    } else if (state_ == "await_pick") {
      HandleFuture("상품 픽업");
    } else if (state_ == "request_place") {
      SendPlaceRequest();
      state_ = "await_place";
    } else if (state_ == "await_place") {
      HandleFuture("상품 담기");
    } else if (state_ == "completed") {
      RCLCPP_INFO(this->get_logger(), "모의 통신 테스트를 완료했습니다. 노드를 종료합니다.");
      rclcpp::shutdown();
    }
  }

  // -------- 서비스 요청 --------
  void HandleFuture(const std::string & action_name) {
    bool future_ready = false;
    bool success = true;
    std::string message;

    std::visit([&](auto & future) {
      using FutureType = std::decay_t<decltype(future)>;
      if constexpr (!std::is_same_v<FutureType, std::monostate>) {
        if (!future.valid()) return;
        if (future.wait_for(0s) != std::future_status::ready) return;
        auto response = future.get();
        success = response->success;
        message = response->message;
        future_ready = true;
      }
    }, current_future_);

    if (!future_ready) return;
    current_future_ = std::monostate{};

    if (!success) {
      RCLCPP_ERROR(this->get_logger(), "%s 명령 실패: %s", action_name.c_str(), message.c_str());
      state_ = "completed";
      return;
    }

    RCLCPP_INFO(this->get_logger(), "%s 성공: %s", action_name.c_str(), message.c_str());

    if (action_name == "자세 변경") state_ = "request_pick";
    else if (action_name == "상품 픽업") state_ = "request_place";
    else if (action_name == "상품 담기") {
      ++current_arm_index_;
      if (current_arm_index_ < arm_sides_.size()) {
        RCLCPP_INFO(this->get_logger(), "다음 팔 테스트: %s (product_id=%d)",
          CurrentArmSide().c_str(), CurrentProductId());
        state_ = "request_pick";
      } else state_ = "completed";
    }
  }

  void SendMoveRequest() {
    auto req = std::make_shared<shopee_interfaces::srv::ArmMoveToPose::Request>();
    req->robot_id = robot_id_;
    req->order_id = order_id_;
    req->pose_type = "cart_view";
    current_future_ = move_client_->async_send_request(req);
  }

  void SendPickRequest() {
    auto req = std::make_shared<shopee_interfaces::srv::ArmPickProduct::Request>();
    req->robot_id = robot_id_;
    req->order_id = order_id_;
    req->arm_side = CurrentArmSide();
    req->target_product.product_id = CurrentProductId();
    AssignDetectedProductPose(&req->target_product, 0.25F, 0.0F, 0.12F, 0.0F);
    req->target_product.bbox = CreateBBox(120, 180, 250, 320);
    current_future_ = pick_client_->async_send_request(req);
  }

  void SendPlaceRequest() {
    auto req = std::make_shared<shopee_interfaces::srv::ArmPlaceProduct::Request>();
    req->robot_id = robot_id_;
    req->order_id = order_id_;
    req->product_id = CurrentProductId();
    req->arm_side = CurrentArmSide();
    AssignPlacePose(req.get(), 0.35F, 0.1F, 0.15F, 0.0F);
    current_future_ = place_client_->async_send_request(req);
  }

  // -------- 유틸 생성 --------
  std::string CurrentArmSide() const {
    return (current_arm_index_ < arm_sides_.size()) ? arm_sides_[current_arm_index_] : default_arm_side_;
  }

  int32_t CurrentProductId() const {
    return base_product_id_ + static_cast<int32_t>(current_arm_index_);
  }

  shopee_interfaces::msg::BBox CreateBBox(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const {
    shopee_interfaces::msg::BBox b;
    b.x1 = x1; b.y1 = y1; b.x2 = x2; b.y2 = y2;
    return b;
  }

  // -------- 멤버 변수 --------
  int robot_id_, order_id_, base_product_id_;
  std::string default_arm_side_;
  std::vector<std::string> arm_sides_;
  std::string state_;
  size_t current_arm_index_;
  FutureVariant current_future_;
  MoveClient::SharedPtr move_client_;
  PickClient::SharedPtr pick_client_;
  PlaceClient::SharedPtr place_client_;
  rclcpp::Subscription<shopee_interfaces::msg::ArmPoseStatus>::SharedPtr pose_status_sub_;
  rclcpp::Subscription<shopee_interfaces::msg::ArmTaskStatus>::SharedPtr pick_status_sub_;
  rclcpp::Subscription<shopee_interfaces::msg::ArmTaskStatus>::SharedPtr place_status_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MockPackeeMain>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
