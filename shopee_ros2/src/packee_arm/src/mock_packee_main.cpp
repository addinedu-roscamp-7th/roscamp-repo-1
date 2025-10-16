#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "shopee_interfaces/msg/arm_pose_status.hpp"
#include "shopee_interfaces/msg/b_box.hpp"
#include "shopee_interfaces/msg/packee_arm_task_status.hpp"
#include "shopee_interfaces/msg/point3_d.hpp"
#include "shopee_interfaces/srv/packee_arm_move_to_pose.hpp"
#include "shopee_interfaces/srv/packee_arm_pick_product.hpp"
#include "shopee_interfaces/srv/packee_arm_place_product.hpp"

using namespace std::chrono_literals;

class MockPackeeMain : public rclcpp::Node {
public:
  MockPackeeMain()
  : rclcpp::Node("mock_packee_main"),
    state_("wait_services"),
    current_arm_index_(0) {
    // 테스트 파라미터 선언 및 로드
    this->declare_parameter<int>("robot_id", 1);
    this->declare_parameter<int>("order_id", 100);
    this->declare_parameter<int>("product_id", 501);
    this->declare_parameter<std::string>("arm_side", "left");
    this->declare_parameter<std::string>("arm_sides", "left,right");

    robot_id_ = this->get_parameter("robot_id").as_int();
    order_id_ = this->get_parameter("order_id").as_int();
    base_product_id_ = this->get_parameter("product_id").as_int();
    default_arm_side_ = this->get_parameter("arm_side").as_string();

    const auto arm_sides_param = this->get_parameter("arm_sides").as_string();
    ParseArmSides(arm_sides_param);

    if (arm_sides_.empty()) {
      arm_sides_.push_back(default_arm_side_);
    }

    // 서비스 클라이언트 생성
    move_client_ = this->create_client<shopee_interfaces::srv::PackeeArmMoveToPose>(
      "/packee/arm/move_to_pose");
    pick_client_ = this->create_client<shopee_interfaces::srv::PackeeArmPickProduct>(
      "/packee/arm/pick_product");
    place_client_ = this->create_client<shopee_interfaces::srv::PackeeArmPlaceProduct>(
      "/packee/arm/place_product");

    // 상태 토픽 구독
    pose_status_sub_ = this->create_subscription<shopee_interfaces::msg::ArmPoseStatus>(
      "/packee/arm/pose_status", 10,
      std::bind(&MockPackeeMain::OnPoseStatus, this, std::placeholders::_1));
    pick_status_sub_ = this->create_subscription<shopee_interfaces::msg::PackeeArmTaskStatus>(
      "/packee/arm/pick_status", 10,
      std::bind(&MockPackeeMain::OnPickStatus, this, std::placeholders::_1));
    place_status_sub_ = this->create_subscription<shopee_interfaces::msg::PackeeArmTaskStatus>(
      "/packee/arm/place_status", 10,
      std::bind(&MockPackeeMain::OnPlaceStatus, this, std::placeholders::_1));

    // 타이머를 통해 상태 기계 실행
    timer_ = this->create_wall_timer(
      200ms, std::bind(&MockPackeeMain::ProcessSteps, this));

    RCLCPP_INFO(
      this->get_logger(),
      "Packee Arm 통신 모의 테스트를 시작합니다. robot_id=%d, order_id=%d, product_id 시작값=%d, arm_sides=%s",
      robot_id_, order_id_, base_product_id_, ArmSidesToString().c_str());
  }

private:
  using MoveClient = rclcpp::Client<shopee_interfaces::srv::PackeeArmMoveToPose>;
  using PickClient = rclcpp::Client<shopee_interfaces::srv::PackeeArmPickProduct>;
  using PlaceClient = rclcpp::Client<shopee_interfaces::srv::PackeeArmPlaceProduct>;
  using MoveFuture = MoveClient::SharedFuture;
  using PickFuture = PickClient::SharedFuture;
  using PlaceFuture = PlaceClient::SharedFuture;
  using FutureVariant = std::variant<std::monostate, MoveFuture, PickFuture, PlaceFuture>;

  void ParseArmSides(const std::string & raw_value) {
    // arm_sides 파라미터 문자열을 분리하여 리스트로 저장
    std::string token;
    for (char ch : raw_value) {
      if (ch == ',') {
        AppendArmSide(token);
        token.clear();
      } else {
        token.push_back(ch);
      }
    }
    AppendArmSide(token);
  }

  void AppendArmSide(const std::string & value) {
    // arm_sides 리스트에 값 추가
    const auto trimmed = Trim(value);
    if (!trimmed.empty()) {
      arm_sides_.push_back(trimmed);
    }
  }

  std::string Trim(const std::string & value) const {
    // 문자열 양 끝 공백 제거
    const auto begin = value.find_first_not_of(" \t");
    if (begin == std::string::npos) {
      return "";
    }
    const auto end = value.find_last_not_of(" \t");
    return value.substr(begin, end - begin + 1);
  }

  std::string ArmSidesToString() const {
    // arm_sides 리스트를 문자열로 변환
    std::string result;
    for (size_t index = 0; index < arm_sides_.size(); ++index) {
      if (index > 0) {
        result += ",";
      }
      result += arm_sides_[index];
    }
    return result;
  }

  void OnPoseStatus(const shopee_interfaces::msg::ArmPoseStatus::SharedPtr status_msg) {
    // 자세 상태 수신 로그
    RCLCPP_INFO(
      this->get_logger(),
      "[PoseStatus] pose_type=%s, status=%s, progress=%.2f, message=%s",
      status_msg->pose_type.c_str(),
      status_msg->status.c_str(),
      status_msg->progress,
      status_msg->message.c_str());
  }

  void OnPickStatus(const shopee_interfaces::msg::PackeeArmTaskStatus::SharedPtr status_msg) {
    // 픽업 상태 수신 로그
    RCLCPP_INFO(
      this->get_logger(),
      "[PickStatus] product_id=%d, phase=%s, status=%s, progress=%.2f, message=%s",
      status_msg->product_id,
      status_msg->current_phase.c_str(),
      status_msg->status.c_str(),
      status_msg->progress,
      status_msg->message.c_str());
  }

  void OnPlaceStatus(const shopee_interfaces::msg::PackeeArmTaskStatus::SharedPtr status_msg) {
    // 담기 상태 수신 로그
    RCLCPP_INFO(
      this->get_logger(),
      "[PlaceStatus] product_id=%d, phase=%s, status=%s, progress=%.2f, message=%s",
      status_msg->product_id,
      status_msg->current_phase.c_str(),
      status_msg->status.c_str(),
      status_msg->progress,
      status_msg->message.c_str());
  }

  void ProcessSteps() {
    // 상태 기계 흐름 제어
    if (state_ == "wait_services") {
      if (move_client_->service_is_ready() &&
        pick_client_->service_is_ready() &&
        place_client_->service_is_ready())
      {
        RCLCPP_INFO(this->get_logger(), "Packee Arm 서비스가 준비되었습니다. 자세 변경을 요청합니다.");
        SendMoveRequest();
        state_ = "await_move";
      }
    } else if (state_ == "await_move") {
      HandleFuture("자세 변경");
    } else if (state_ == "request_pick") {
      if (current_arm_index_ >= arm_sides_.size()) {
        state_ = "completed";
      } else {
        SendPickRequest();
        state_ = "await_pick";
      }
    } else if (state_ == "await_pick") {
      HandleFuture("상품 픽업");
    } else if (state_ == "request_place") {
      SendPlaceRequest();
      state_ = "await_place";
    } else if (state_ == "await_place") {
      HandleFuture("상품 담기");
    } else if (state_ == "completed") {
      RCLCPP_INFO(this->get_logger(), "모의 통신 테스트를 완료했습니다. 노드를 종료합니다.");
      state_ = "stopping";
      rclcpp::shutdown();
    }
  }

  void HandleFuture(const std::string & action_name) {
    // 비동기 서비스 응답 처리
    bool future_ready = false;
    bool accepted = true;
    std::string message;

    std::visit(
      [&](auto & future) {
        using FutureType = std::decay_t<decltype(future)>;
        if constexpr (!std::is_same_v<FutureType, std::monostate>) {
          if (!future.valid()) {
            return;
          }
          if (future.wait_for(0s) != std::future_status::ready) {
            return;
          }
          auto response = future.get();
          accepted = response->accepted;
          message = response->message;
          future_ready = true;
        }
      }, current_future_);

    if (!future_ready) {
      return;
    }

    current_future_ = std::monostate{};

    if (!accepted) {
      RCLCPP_ERROR(
        this->get_logger(),
        "%s 명령이 Arm 컨트롤러에서 거부되었습니다: message=%s",
        action_name.c_str(), message.c_str());
      state_ = "completed";
      return;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "%s 서비스 응답 수신: accepted=true, message=%s",
      action_name.c_str(), message.c_str());

    if (action_name == "자세 변경") {
      state_ = "request_pick";
    } else if (action_name == "상품 픽업") {
      state_ = "request_place";
    } else if (action_name == "상품 담기") {
      ++current_arm_index_;
      if (current_arm_index_ < arm_sides_.size()) {
        RCLCPP_INFO(
          this->get_logger(),
          "다음 팔 테스트 준비: arm_side=%s, product_id=%d",
          CurrentArmSide().c_str(), CurrentProductId());
        state_ = "request_pick";
      } else {
        state_ = "completed";
      }
    }
  }

  void SendMoveRequest() {
    // 자세 변경 서비스 호출
    auto request = std::make_shared<shopee_interfaces::srv::PackeeArmMoveToPose::Request>();
    request->robot_id = robot_id_;
    request->order_id = order_id_;
    request->pose_type = "cart_view";
    current_future_ = move_client_->async_send_request(request);
  }

  void SendPickRequest() {
    // 상품 픽업 서비스 호출
    auto request = std::make_shared<shopee_interfaces::srv::PackeeArmPickProduct::Request>();
    request->robot_id = robot_id_;
    request->order_id = order_id_;
    request->product_id = CurrentProductId();
    request->arm_side = CurrentArmSide();
    request->target_position = CreatePoint3D(0.3F, 0.1F, 0.75F);
    request->bbox = CreateBBox(120, 180, 250, 320);
    current_future_ = pick_client_->async_send_request(request);
  }

  void SendPlaceRequest() {
    // 상품 담기 서비스 호출
    auto request = std::make_shared<shopee_interfaces::srv::PackeeArmPlaceProduct::Request>();
    request->robot_id = robot_id_;
    request->order_id = order_id_;
    request->product_id = CurrentProductId();
    request->arm_side = CurrentArmSide();
    request->box_position = CreatePoint3D(0.5F, 0.2F, 0.2F);
    current_future_ = place_client_->async_send_request(request);
  }

  std::string CurrentArmSide() const {
    // 현재 테스트 대상 팔 구분 반환
    if (current_arm_index_ < arm_sides_.size()) {
      return arm_sides_[current_arm_index_];
    }
    return default_arm_side_;
  }

  int32_t CurrentProductId() const {
    // 현재 테스트 대상 상품 ID 반환
    return base_product_id_ + static_cast<int32_t>(current_arm_index_);
  }

  shopee_interfaces::msg::Point3D CreatePoint3D(float x, float y, float z) const {
    // 3차원 좌표 메시지 생성
    shopee_interfaces::msg::Point3D point;
    point.x = x;
    point.y = y;
    point.z = z;
    return point;
  }

  shopee_interfaces::msg::BBox CreateBBox(
    int32_t x1, int32_t y1, int32_t x2, int32_t y2) const {
    // BBox 메시지 생성
    shopee_interfaces::msg::BBox bbox;
    bbox.x1 = x1;
    bbox.y1 = y1;
    bbox.x2 = x2;
    bbox.y2 = y2;
    return bbox;
  }

  int robot_id_;
  int order_id_;
  int base_product_id_;
  std::string default_arm_side_;
  std::vector<std::string> arm_sides_;
  std::string state_;
  size_t current_arm_index_;
  FutureVariant current_future_;

  MoveClient::SharedPtr move_client_;
  PickClient::SharedPtr pick_client_;
  PlaceClient::SharedPtr place_client_;

  rclcpp::Subscription<shopee_interfaces::msg::ArmPoseStatus>::SharedPtr pose_status_sub_;
  rclcpp::Subscription<shopee_interfaces::msg::PackeeArmTaskStatus>::SharedPtr pick_status_sub_;
  rclcpp::Subscription<shopee_interfaces::msg::PackeeArmTaskStatus>::SharedPtr place_status_sub_;

  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
  // ROS 2 초기화
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MockPackeeMain>();
  try {
    // 노드 스핀
    rclcpp::spin(node);
  } catch (const std::exception & error) {
    RCLCPP_ERROR(node->get_logger(), "예외 발생: %s", error.what());
  }
  // 자원 정리
  rclcpp::shutdown();
  return 0;
}
