#include "pickee_mobile_wonho/zlac_driver.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <chrono>
#include <thread>

namespace pickee_mobile_wonho {

ZlacDriver::ZlacDriver(const std::string& device, int baudrate, int timeout_ms)
    : device_path_(device)
    , baudrate_(baudrate)
    , timeout_ms_(timeout_ms)
    , serial_fd_(-1)
    , is_connected_(false)
    , logger_(rclcpp::get_logger("zlac_driver"))
{
    RCLCPP_INFO(logger_, "ZLAC 드라이버 생성: device=%s, baudrate=%d", 
                device_path_.c_str(), baudrate_);
}

ZlacDriver::~ZlacDriver() {
    Disconnect();
}

bool ZlacDriver::Initialize() {
    if (is_connected_) {
        RCLCPP_WARN(logger_, "이미 연결되어 있습니다.");
        return true;
    }

    // 시리얼 포트 열기
    serial_fd_ = open(device_path_.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (serial_fd_ < 0) {
        RCLCPP_ERROR(logger_, "시리얼 포트 열기 실패: %s", device_path_.c_str());
        return false;
    }

    // 시리얼 포트 설정
    struct termios options;
    tcgetattr(serial_fd_, &options);

    // 입출력 속도 설정
    speed_t baud;
    switch (baudrate_) {
        case 9600:   baud = B9600; break;
        case 19200:  baud = B19200; break;
        case 38400:  baud = B38400; break;
        case 57600:  baud = B57600; break;
        case 115200: baud = B115200; break;
        default:
            RCLCPP_ERROR(logger_, "지원하지 않는 baudrate: %d", baudrate_);
            close(serial_fd_);
            serial_fd_ = -1;
            return false;
    }

    cfsetispeed(&options, baud);
    cfsetospeed(&options, baud);

    // 8N1 설정 (8비트, 패리티 없음, 1 스톱비트)
    options.c_cflag &= ~PARENB;     // 패리티 비활성화
    options.c_cflag &= ~CSTOPB;     // 1 스톱비트
    options.c_cflag &= ~CSIZE;      // 비트 크기 마스크 클리어
    options.c_cflag |= CS8;         // 8비트
    options.c_cflag |= CREAD | CLOCAL; // 수신 활성화, 모뎀 제어 비활성화

    // Raw 모드 설정
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_oflag &= ~OPOST;

    // 타임아웃 설정
    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = timeout_ms_ / 100; // 0.1초 단위

    // 설정 적용
    if (tcsetattr(serial_fd_, TCSANOW, &options) != 0) {
        RCLCPP_ERROR(logger_, "시리얼 포트 설정 실패");
        close(serial_fd_);
        serial_fd_ = -1;
        return false;
    }

    // 버퍼 플러시
    tcflush(serial_fd_, TCIOFLUSH);

    is_connected_ = true;
    RCLCPP_INFO(logger_, "ZLAC 드라이버 초기화 성공");
    
    // 초기화 후 잠시 대기
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    return true;
}

void ZlacDriver::Disconnect() {
    if (serial_fd_ >= 0) {
        // 모든 모터 비상정지
        EmergencyStop(0);
        
        close(serial_fd_);
        serial_fd_ = -1;
        is_connected_ = false;
        RCLCPP_INFO(logger_, "ZLAC 드라이버 연결 해제");
    }
}

bool ZlacDriver::IsConnected() const {
    return is_connected_;
}

bool ZlacDriver::EnableMotor(int motor_id) {
    if (motor_id < 1 || motor_id > 2) {
        RCLCPP_ERROR(logger_, "잘못된 모터 ID: %d", motor_id);
        return false;
    }

    bool success = WriteSingleRegister(motor_id, REG_ENABLE_MOTOR, 1);
    if (success) {
        RCLCPP_INFO(logger_, "모터 %d 활성화 성공", motor_id);
    } else {
        RCLCPP_ERROR(logger_, "모터 %d 활성화 실패", motor_id);
    }
    return success;
}

bool ZlacDriver::DisableMotor(int motor_id) {
    if (motor_id < 1 || motor_id > 2) {
        RCLCPP_ERROR(logger_, "잘못된 모터 ID: %d", motor_id);
        return false;
    }

    bool success = WriteSingleRegister(motor_id, REG_ENABLE_MOTOR, 0);
    if (success) {
        RCLCPP_INFO(logger_, "모터 %d 비활성화 성공", motor_id);
    } else {
        RCLCPP_ERROR(logger_, "모터 %d 비활성화 실패", motor_id);
    }
    return success;
}

bool ZlacDriver::SetControlMode(int motor_id, ControlMode mode) {
    if (motor_id < 1 || motor_id > 2) {
        RCLCPP_ERROR(logger_, "잘못된 모터 ID: %d", motor_id);
        return false;
    }

    bool success = WriteSingleRegister(motor_id, REG_CONTROL_MODE, static_cast<uint16_t>(mode));
    if (success) {
        RCLCPP_INFO(logger_, "모터 %d 제어 모드 설정 성공: %d", motor_id, static_cast<int>(mode));
    } else {
        RCLCPP_ERROR(logger_, "모터 %d 제어 모드 설정 실패", motor_id);
    }
    return success;
}

bool ZlacDriver::SetVelocity(int motor_id, int32_t velocity_rpm) {
    if (motor_id < 1 || motor_id > 2) {
        RCLCPP_ERROR(logger_, "잘못된 모터 ID: %d", motor_id);
        return false;
    }

    // 32비트 값을 16비트 2개로 분할
    std::vector<uint16_t> values = SplitInt32(velocity_rpm);
    
    bool success = WriteMultipleRegisters(motor_id, REG_VELOCITY_CMD, values);
    if (!success) {
        RCLCPP_ERROR(logger_, "모터 %d 속도 설정 실패: %d RPM", motor_id, velocity_rpm);
    }
    return success;
}

bool ZlacDriver::SetPosition(int motor_id, int32_t position_pulse) {
    if (motor_id < 1 || motor_id > 2) {
        RCLCPP_ERROR(logger_, "잘못된 모터 ID: %d", motor_id);
        return false;
    }

    // 32비트 값을 16비트 2개로 분할
    std::vector<uint16_t> values = SplitInt32(position_pulse);
    
    bool success = WriteMultipleRegisters(motor_id, REG_POSITION_CMD, values);
    if (!success) {
        RCLCPP_ERROR(logger_, "모터 %d 위치 설정 실패: %d pulse", motor_id, position_pulse);
    }
    return success;
}

bool ZlacDriver::ReadMotorStatus(int motor_id, MotorStatus& status) {
    if (motor_id < 1 || motor_id > 2) {
        RCLCPP_ERROR(logger_, "잘못된 모터 ID: %d", motor_id);
        return false;
    }

    // 상태 레지스터들을 한번에 읽기 (위치, 속도, 상태)
    std::vector<uint16_t> values;
    if (!ReadRegisters(motor_id, REG_CURRENT_POSITION, 6, values)) {
        return false;
    }

    if (values.size() >= 6) {
        // 현재 위치 (32비트)
        status.position = CombineInt32(values[0], values[1]);
        
        // 현재 속도 (32비트)
        status.velocity = CombineInt32(values[2], values[3]);
        
        // 모터 상태
        uint16_t motor_status = values[4];
        status.is_enabled = (motor_status & 0x01) != 0;
        status.is_alarm = (motor_status & 0x02) != 0;
        status.is_in_position = (motor_status & 0x04) != 0;
        
        // 토크 정보
        status.torque = static_cast<int16_t>(values[5]);
        
        return true;
    }

    return false;
}

bool ZlacDriver::ReadPosition(int motor_id, int32_t& position) {
    if (motor_id < 1 || motor_id > 2) {
        RCLCPP_ERROR(logger_, "잘못된 모터 ID: %d", motor_id);
        return false;
    }

    std::vector<uint16_t> values;
    if (ReadRegisters(motor_id, REG_CURRENT_POSITION, 2, values) && values.size() >= 2) {
        position = CombineInt32(values[0], values[1]);
        return true;
    }
    return false;
}

bool ZlacDriver::ReadVelocity(int motor_id, int32_t& velocity) {
    if (motor_id < 1 || motor_id > 2) {
        RCLCPP_ERROR(logger_, "잘못된 모터 ID: %d", motor_id);
        return false;
    }

    std::vector<uint16_t> values;
    if (ReadRegisters(motor_id, REG_CURRENT_VELOCITY, 2, values) && values.size() >= 2) {
        velocity = CombineInt32(values[0], values[1]);
        return true;
    }
    return false;
}

bool ZlacDriver::ClearAlarm(int motor_id) {
    if (motor_id < 1 || motor_id > 2) {
        RCLCPP_ERROR(logger_, "잘못된 모터 ID: %d", motor_id);
        return false;
    }

    bool success = WriteSingleRegister(motor_id, REG_CLEAR_ALARM, 1);
    if (success) {
        RCLCPP_INFO(logger_, "모터 %d 알람 클리어 성공", motor_id);
    } else {
        RCLCPP_ERROR(logger_, "모터 %d 알람 클리어 실패", motor_id);
    }
    return success;
}

bool ZlacDriver::EmergencyStop(int motor_id) {
    bool success = true;
    
    if (motor_id == 0) {
        // 모든 모터 정지
        success &= WriteSingleRegister(1, REG_EMERGENCY_STOP, 1);
        success &= WriteSingleRegister(2, REG_EMERGENCY_STOP, 1);
        RCLCPP_WARN(logger_, "모든 모터 비상정지 실행");
    } else if (motor_id >= 1 && motor_id <= 2) {
        success = WriteSingleRegister(motor_id, REG_EMERGENCY_STOP, 1);
        RCLCPP_WARN(logger_, "모터 %d 비상정지 실행", motor_id);
    } else {
        RCLCPP_ERROR(logger_, "잘못된 모터 ID: %d", motor_id);
        return false;
    }
    
    return success;
}

// Private 메서드들

uint16_t ZlacDriver::CalculateCRC(const uint8_t* data, size_t length) {
    uint16_t crc = 0xFFFF;
    
    for (size_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 0x0001) {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    
    return crc;
}

bool ZlacDriver::SendModbusFrame(uint8_t slave_id, uint8_t function_code, 
                                uint16_t reg_addr, uint16_t reg_count,
                                const uint8_t* data, size_t data_len) {
    if (!is_connected_) {
        RCLCPP_ERROR(logger_, "시리얼 연결이 되어있지 않습니다.");
        return false;
    }

    std::vector<uint8_t> frame;
    frame.push_back(slave_id);
    frame.push_back(function_code);
    frame.push_back((reg_addr >> 8) & 0xFF);    // 주소 상위 바이트
    frame.push_back(reg_addr & 0xFF);           // 주소 하위 바이트
    frame.push_back((reg_count >> 8) & 0xFF);   // 카운트 상위 바이트
    frame.push_back(reg_count & 0xFF);          // 카운트 하위 바이트

    // 추가 데이터가 있으면 추가
    if (data && data_len > 0) {
        frame.insert(frame.end(), data, data + data_len);
    }

    // CRC 계산 및 추가
    uint16_t crc = CalculateCRC(frame.data(), frame.size());
    frame.push_back(crc & 0xFF);        // CRC 하위 바이트
    frame.push_back((crc >> 8) & 0xFF); // CRC 상위 바이트

    // 전송
    ssize_t bytes_written = write(serial_fd_, frame.data(), frame.size());
    if (bytes_written != static_cast<ssize_t>(frame.size())) {
        RCLCPP_ERROR(logger_, "프레임 전송 실패");
        return false;
    }

    // 전송 완료 대기
    tcdrain(serial_fd_);
    
    return true;
}

int ZlacDriver::ReceiveModbusResponse(uint8_t* response, size_t max_len) {
    if (!is_connected_) {
        RCLCPP_ERROR(logger_, "시리얼 연결이 되어있지 않습니다.");
        return -1;
    }

    // 응답 대기
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    ssize_t bytes_read = read(serial_fd_, response, max_len);
    if (bytes_read <= 0) {
        RCLCPP_ERROR(logger_, "응답 수신 실패 또는 타임아웃");
        return -1;
    }

    // CRC 검증
    if (bytes_read >= 3) {
        uint16_t received_crc = response[bytes_read - 2] | (response[bytes_read - 1] << 8);
        uint16_t calculated_crc = CalculateCRC(response, bytes_read - 2);
        
        if (received_crc != calculated_crc) {
            RCLCPP_ERROR(logger_, "CRC 오류: 수신=%04X, 계산=%04X", received_crc, calculated_crc);
            return -1;
        }
    }

    return static_cast<int>(bytes_read);
}

bool ZlacDriver::ReadRegisters(int motor_id, uint16_t reg_addr, uint16_t reg_count, 
                              std::vector<uint16_t>& values) {
    if (!SendModbusFrame(motor_id, FUNC_READ_HOLDING_REGISTERS, reg_addr, reg_count)) {
        return false;
    }

    uint8_t response[256];
    int response_len = ReceiveModbusResponse(response, sizeof(response));
    
    if (response_len < 5) { // 최소 응답 길이
        return false;
    }

    // 응답 구조: [ID][FUNC][BYTE_COUNT][DATA...][CRC_L][CRC_H]
    uint8_t byte_count = response[2];
    if (response_len != 5 + byte_count) {
        RCLCPP_ERROR(logger_, "응답 길이 오류");
        return false;
    }

    values.clear();
    for (int i = 0; i < byte_count; i += 2) {
        uint16_t value = (response[3 + i] << 8) | response[3 + i + 1];
        values.push_back(value);
    }

    return true;
}

bool ZlacDriver::WriteSingleRegister(int motor_id, uint16_t reg_addr, uint16_t value) {
    if (!SendModbusFrame(motor_id, FUNC_WRITE_SINGLE_REGISTER, reg_addr, value)) {
        return false;
    }

    uint8_t response[8];
    int response_len = ReceiveModbusResponse(response, sizeof(response));
    
    // 응답 검증 (에코 응답이어야 함)
    return (response_len == 8) && 
           (response[0] == motor_id) && 
           (response[1] == FUNC_WRITE_SINGLE_REGISTER);
}

bool ZlacDriver::WriteMultipleRegisters(int motor_id, uint16_t reg_addr, 
                                       const std::vector<uint16_t>& values) {
    if (values.empty()) {
        return false;
    }

    // 데이터 준비
    std::vector<uint8_t> data;
    data.push_back(values.size() * 2); // 바이트 카운트
    
    for (uint16_t value : values) {
        data.push_back((value >> 8) & 0xFF); // 상위 바이트
        data.push_back(value & 0xFF);        // 하위 바이트
    }

    if (!SendModbusFrame(motor_id, FUNC_WRITE_MULTIPLE_REGISTERS, reg_addr, 
                        values.size(), data.data(), data.size())) {
        return false;
    }

    uint8_t response[8];
    int response_len = ReceiveModbusResponse(response, sizeof(response));
    
    // 응답 검증
    return (response_len == 8) && 
           (response[0] == motor_id) && 
           (response[1] == FUNC_WRITE_MULTIPLE_REGISTERS);
}

std::vector<uint16_t> ZlacDriver::SplitInt32(int32_t value) {
    std::vector<uint16_t> result(2);
    result[0] = (value >> 16) & 0xFFFF; // 상위 16비트
    result[1] = value & 0xFFFF;         // 하위 16비트
    return result;
}

int32_t ZlacDriver::CombineInt32(uint16_t high, uint16_t low) {
    return (static_cast<int32_t>(high) << 16) | low;
}

} // namespace pickee_mobile_wonho