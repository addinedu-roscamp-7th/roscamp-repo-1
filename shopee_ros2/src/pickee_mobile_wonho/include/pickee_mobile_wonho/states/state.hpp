#pragma once

namespace pickee_mobile_wonho {

/**
 * @brief 상태 타입 열거형
 */
enum class StateType {
    IDLE,
    MOVING,
    STOPPED,
    CHARGING,
    ERROR
};

/**
 * @brief 상태 기계의 추상 기본 클래스
 * 
 * 모든 상태 클래스는 이 클래스를 상속받아 구현해야 합니다.
 * RAII 패턴을 적용하여 안전한 상태 전환을 보장합니다.
 */
class State {
public:
    /**
     * @brief 가상 소멸자
     */
    virtual ~State() = default;

    /**
     * @brief 상태 진입 시 호출되는 메서드
     */
    virtual void OnEnter() = 0;

    /**
     * @brief 상태 실행 중 주기적으로 호출되는 메서드
     */
    virtual void Execute() = 0;

    /**
     * @brief 상태 이탈 시 호출되는 메서드
     */
    virtual void OnExit() = 0;

    /**
     * @brief 현재 상태의 타입을 반환
     * @return StateType 상태 타입
     */
    virtual StateType GetType() const = 0;

protected:
    /**
     * @brief 기본 생성자 (상속 클래스에서만 사용)
     */
    State() = default;

    /**
     * @brief 복사 생성자 삭제
     */
    State(const State&) = delete;

    /**
     * @brief 할당 연산자 삭제
     */
    State& operator=(const State&) = delete;
};

} // namespace pickee_mobile_wonho