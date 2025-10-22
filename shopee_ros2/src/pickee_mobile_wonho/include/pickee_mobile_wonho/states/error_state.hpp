#pragma once

#include <memory>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "pickee_mobile_wonho/states/state.hpp"

namespace pickee_mobile_wonho {

/**
 * @brief 오류 상태 클래스
 * 
 * 시스템 오류가 발생한 상태입니다.
 */
class ErrorState : public State {
public:
    /**
     * @brief 오류 타입 열거형
     */
    enum class ErrorType {
        COMMUNICATION_ERROR,    ///< 통신 오류
        SENSOR_ERROR,          ///< 센서 오류
        ACTUATOR_ERROR,        ///< 액추에이터 오류
        NAVIGATION_ERROR,      ///< 네비게이션 오류
        MEMORY_ERROR,          ///< 메모리 오류
        UNKNOWN_ERROR          ///< 알 수 없는 오류
    };

    /**
     * @brief 생성자
     * @param logger ROS2 로거 포인터
     * @param error_type 오류 타입
     * @param error_message 오류 메시지
     */
    explicit ErrorState(std::shared_ptr<rclcpp::Logger> logger,
                       ErrorType error_type = ErrorType::UNKNOWN_ERROR,
                       const std::string& error_message = "");

    /**
     * @brief 소멸자
     */
    ~ErrorState() override = default;

    /**
     * @brief 상태 진입 시 호출
     */
    void OnEnter() override;

    /**
     * @brief 상태 실행 중 주기적으로 호출
     */
    void Execute() override;

    /**
     * @brief 상태 이탈 시 호출
     */
    void OnExit() override;

    /**
     * @brief 상태 타입 반환
     * @return StateType::ERROR
     */
    StateType GetType() const override;

    /**
     * @brief 오류 타입 반환
     * @return ErrorType 오류 타입
     */
    ErrorType GetErrorType() const;

    /**
     * @brief 오류 메시지 반환
     * @return 오류 메시지
     */
    const std::string& GetErrorMessage() const;

    /**
     * @brief 복구 가능 여부 확인
     * @return true 복구 가능, false 수동 개입 필요
     */
    bool IsRecoverable() const;

    /**
     * @brief 복구 시도
     * @return true 복구 성공, false 복구 실패
     */
    bool AttemptRecovery();

private:
    /**
     * @brief ROS2 로거
     */
    std::shared_ptr<rclcpp::Logger> logger_;

    /**
     * @brief 오류 타입
     */
    ErrorType error_type_;

    /**
     * @brief 오류 메시지
     */
    std::string error_message_;

    /**
     * @brief 오류 발생 시간
     */
    std::chrono::steady_clock::time_point error_start_time_;

    /**
     * @brief 복구 시도 횟수
     */
    int recovery_attempts_;

    /**
     * @brief 최대 복구 시도 횟수
     */
    static constexpr int MAX_RECOVERY_ATTEMPTS = 3;

    /**
     * @brief 오류 타입을 문자열로 변환
     * @param type 오류 타입
     * @return 오류 타입 문자열
     */
    std::string ErrorTypeToString(ErrorType type) const;
};

} // namespace pickee_mobile_wonho