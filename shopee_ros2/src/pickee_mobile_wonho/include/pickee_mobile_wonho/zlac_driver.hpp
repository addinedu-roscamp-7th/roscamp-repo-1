#ifndef PICKEE_MOBILE_WONHO_ZLAC_DRIVER_HPP
#define PICKEE_MOBILE_WONHO_ZLAC_DRIVER_HPP

#include <string>
#include <vector>
#include <memory>
#include <rclcpp/rclcpp.hpp>

namespace pickee_mobile_wonho {

/**
 * @brief ZLAC 모터 드라이버 클래스
 * 
 * ZLAC8015D 스텝 서보 드라이버와 Modbus RTU 프로토콜로 통신하여
 * 모터를 제어하고 상태를 읽어오는 기능을 제공합니다.
 */
class ZlacDriver {
public:
    // 모터 제어 모드 정의
    enum class ControlMode : uint16_t {
        POSITION_MODE = 1,      // 위치 제어 모드
        VELOCITY_MODE = 2,      // 속도 제어 모드  
        TORQUE_MODE = 3         // 토크 제어 모드
    };

    // 모터 상태 구조체
    struct MotorStatus {
        bool is_enabled;        // 모터 활성화 상태
        bool is_alarm;          // 알람 상태
        bool is_in_position;    // 위치 도달 상태
        int32_t position;       // 현재 위치 (펄스)
        int32_t velocity;       // 현재 속도 (RPM)
        int16_t torque;         // 현재 토크 (0.1%)
    };

    /**
     * @brief 생성자
     * @param device 시리얼 포트 디바이스 경로 (예: "/dev/ttyUSB0")
     * @param baudrate 통신 속도 (기본값: 115200)
     * @param timeout_ms 통신 타임아웃 (밀리초, 기본값: 100)
     */
    ZlacDriver(const std::string& device, 
               int baudrate = 115200, 
               int timeout_ms = 100);
    
    /**
     * @brief 소멸자
     */
    ~ZlacDriver();

    /**
     * @brief 시리얼 연결 초기화
     * @return 성공 시 true, 실패 시 false
     */
    bool Initialize();

    /**
     * @brief 연결 해제
     */
    void Disconnect();

    /**
     * @brief 연결 상태 확인
     * @return 연결된 경우 true
     */
    bool IsConnected() const;

    /**
     * @brief 모터 활성화
     * @param motor_id 모터 ID (1 또는 2)
     * @return 성공 시 true
     */
    bool EnableMotor(int motor_id);

    /**
     * @brief 모터 비활성화
     * @param motor_id 모터 ID (1 또는 2)
     * @return 성공 시 true
     */
    bool DisableMotor(int motor_id);

    /**
     * @brief 모터 제어 모드 설정
     * @param motor_id 모터 ID (1 또는 2)
     * @param mode 제어 모드
     * @return 성공 시 true
     */
    bool SetControlMode(int motor_id, ControlMode mode);

    /**
     * @brief 속도 제어 명령 전송
     * @param motor_id 모터 ID (1 또는 2)
     * @param velocity_rpm 목표 속도 (RPM)
     * @return 성공 시 true
     */
    bool SetVelocity(int motor_id, int32_t velocity_rpm);

    /**
     * @brief 위치 제어 명령 전송
     * @param motor_id 모터 ID (1 또는 2)
     * @param position_pulse 목표 위치 (펄스)
     * @return 성공 시 true
     */
    bool SetPosition(int motor_id, int32_t position_pulse);

    /**
     * @brief 모터 상태 읽기
     * @param motor_id 모터 ID (1 또는 2)
     * @param status 상태를 저장할 구조체
     * @return 성공 시 true
     */
    bool ReadMotorStatus(int motor_id, MotorStatus& status);

    /**
     * @brief 현재 위치 읽기
     * @param motor_id 모터 ID (1 또는 2)
     * @param position 위치값을 저장할 변수
     * @return 성공 시 true
     */
    bool ReadPosition(int motor_id, int32_t& position);

    /**
     * @brief 현재 속도 읽기
     * @param motor_id 모터 ID (1 또는 2)
     * @param velocity 속도값을 저장할 변수
     * @return 성공 시 true
     */
    bool ReadVelocity(int motor_id, int32_t& velocity);

    /**
     * @brief 알람 클리어
     * @param motor_id 모터 ID (1 또는 2)
     * @return 성공 시 true
     */
    bool ClearAlarm(int motor_id);

    /**
     * @brief 비상정지
     * @param motor_id 모터 ID (1 또는 2, 0이면 모든 모터)
     * @return 성공 시 true
     */
    bool EmergencyStop(int motor_id = 0);

private:
    // Modbus RTU 관련 상수
    static const uint8_t FUNC_READ_HOLDING_REGISTERS = 0x03;
    static const uint8_t FUNC_WRITE_SINGLE_REGISTER = 0x06;
    static const uint8_t FUNC_WRITE_MULTIPLE_REGISTERS = 0x10;
    
    // ZLAC 레지스터 주소 정의
    static const uint16_t REG_CONTROL_MODE = 0x200E;
    static const uint16_t REG_ENABLE_MOTOR = 0x200F;
    static const uint16_t REG_VELOCITY_CMD = 0x2088;
    static const uint16_t REG_POSITION_CMD = 0x208A;
    static const uint16_t REG_CURRENT_POSITION = 0x20AB;
    static const uint16_t REG_CURRENT_VELOCITY = 0x20AC;
    static const uint16_t REG_MOTOR_STATUS = 0x20B1;
    static const uint16_t REG_ALARM_CODE = 0x20B2;
    static const uint16_t REG_CLEAR_ALARM = 0x2031;
    static const uint16_t REG_EMERGENCY_STOP = 0x2030;

    std::string device_path_;        // 시리얼 디바이스 경로
    int baudrate_;                   // 통신 속도
    int timeout_ms_;                 // 타임아웃
    int serial_fd_;                  // 시리얼 파일 디스크립터
    bool is_connected_;              // 연결 상태
    
    rclcpp::Logger logger_;          // 로거

    /**
     * @brief Modbus RTU CRC 계산
     * @param data 데이터 버퍼
     * @param length 데이터 길이
     * @return CRC 값
     */
    uint16_t CalculateCRC(const uint8_t* data, size_t length);

    /**
     * @brief Modbus RTU 프레임 전송
     * @param slave_id 슬레이브 ID
     * @param function_code 기능 코드
     * @param reg_addr 레지스터 주소
     * @param reg_count 레지스터 개수 또는 값
     * @param data 추가 데이터 (nullptr 가능)
     * @param data_len 추가 데이터 길이
     * @return 성공 시 true
     */
    bool SendModbusFrame(uint8_t slave_id, 
                        uint8_t function_code,
                        uint16_t reg_addr, 
                        uint16_t reg_count,
                        const uint8_t* data = nullptr, 
                        size_t data_len = 0);

    /**
     * @brief Modbus RTU 응답 수신
     * @param response 응답 데이터를 저장할 버퍼
     * @param max_len 버퍼 최대 크기
     * @return 수신된 바이트 수 (오류 시 -1)
     */
    int ReceiveModbusResponse(uint8_t* response, size_t max_len);

    /**
     * @brief 레지스터 읽기
     * @param motor_id 모터 ID
     * @param reg_addr 레지스터 주소
     * @param reg_count 읽을 레지스터 개수
     * @param values 읽은 값을 저장할 벡터
     * @return 성공 시 true
     */
    bool ReadRegisters(int motor_id, 
                      uint16_t reg_addr, 
                      uint16_t reg_count, 
                      std::vector<uint16_t>& values);

    /**
     * @brief 단일 레지스터 쓰기
     * @param motor_id 모터 ID
     * @param reg_addr 레지스터 주소
     * @param value 쓸 값
     * @return 성공 시 true
     */
    bool WriteSingleRegister(int motor_id, 
                           uint16_t reg_addr, 
                           uint16_t value);

    /**
     * @brief 다중 레지스터 쓰기
     * @param motor_id 모터 ID
     * @param reg_addr 시작 레지스터 주소
     * @param values 쓸 값들
     * @return 성공 시 true
     */
    bool WriteMultipleRegisters(int motor_id, 
                              uint16_t reg_addr, 
                              const std::vector<uint16_t>& values);

    /**
     * @brief 32비트 값을 16비트 레지스터 2개로 분할
     * @param value 32비트 값
     * @return 16비트 값 2개의 벡터 (상위, 하위 순)
     */
    std::vector<uint16_t> SplitInt32(int32_t value);

    /**
     * @brief 16비트 레지스터 2개를 32비트 값으로 결합
     * @param high 상위 16비트
     * @param low 하위 16비트
     * @return 32비트 값
     */
    int32_t CombineInt32(uint16_t high, uint16_t low);
};

} // namespace pickee_mobile_wonho

#endif // PICKEE_MOBILE_WONHO_ZLAC_DRIVER_HPP