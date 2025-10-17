#pragma once
#include <boost/asio.hpp>
#include <mutex>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>

class ZLACDriver {
public:
    ZLACDriver(const std::string& port, int baudrate, uint8_t modbus_id)
        : port_(port), baudrate_(baudrate), modbus_id_(modbus_id),
          io_(), serial_(io_), lock_() {}

    bool begin() {
        try {
            serial_.open(port_);
            serial_.set_option(boost::asio::serial_port_base::baud_rate(baudrate_));
            serial_.set_option(boost::asio::serial_port_base::character_size(8));
            serial_.set_option(boost::asio::serial_port_base::parity(boost::asio::serial_port_base::parity::none));
            serial_.set_option(boost::asio::serial_port_base::stop_bits(boost::asio::serial_port_base::stop_bits::one));
            return true;
        } catch (const boost::system::system_error& e) {
            std::cerr << "Serial open error: " << e.what() << std::endl;
            return false;
        }
    }

    bool enable() {
        auto cmd = build_write_cmd(CONTROL_WORD, MOTOR_ENABLE);
        return send_command(cmd, 8);
    }

    bool disable() {
        auto cmd = build_write_cmd(CONTROL_WORD, MOTOR_DISABLE);
        return send_command(cmd, 8);
    }

    bool set_vel_mode() {
        auto cmd = build_write_cmd(CONTROL_MODE, VEL_MODE);
        return send_command(cmd, 8);
    }

    bool set_double_rpm(int l_rpm, int r_rpm) {
        int r_rpm_neg = -r_rpm;
        std::vector<uint8_t> cmd;
        cmd.push_back(modbus_id_);
        cmd.push_back(MULTI_WRITE);
        append_uint16(cmd, SET_L_RPM);
        append_uint16(cmd, 2); // Quantity of Registers
        cmd.push_back(4);      // Byte Count
        append_int16(cmd, l_rpm);
        append_int16(cmd, r_rpm_neg);
        append_crc(cmd);
        return send_command(cmd, 8);
    }

    std::pair<double, double> get_rpm() {
        std::vector<uint8_t> cmd;
        cmd.push_back(modbus_id_);
        cmd.push_back(READ);
        append_uint16(cmd, GET_RPM);
        append_uint16(cmd, 4);
        append_crc(cmd);
        auto resp = read_response(cmd, 13);
        if (resp.size() == 13) {
            double l_rpm = int16_from(resp, 3) / 10.0;
            double r_rpm = -int16_from(resp, 5) / 10.0;
            return {l_rpm, r_rpm};
        }
        return {0.0, 0.0};
    }

    std::pair<int, int> get_position() {
        std::vector<uint8_t> cmd;
        cmd.push_back(modbus_id_);
        cmd.push_back(READ);
        append_uint16(cmd, GET_ENCODER_PULSE);
        append_uint16(cmd, 5);
        append_crc(cmd);
        auto resp = read_response(cmd, 15);
        if (resp.size() == 15) {
            int encoder_l = int32_from(resp, 5);
            int encoder_r = -int32_from(resp, 9);
            return {encoder_l, encoder_r};
        }
        return {0, 0};
    }

    void terminate() {
        std::cout << "Terminating motor..." << std::endl;
        set_double_rpm(0, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        disable();
        if (serial_.is_open()) serial_.close();
        std::cout << "Motor terminated." << std::endl;
    }

private:
    // Constants
    const uint8_t READ = 0x03;
    const uint8_t WRITE = 0x06;
    const uint8_t MULTI_WRITE = 0x10;
    const uint16_t CONTROL_WORD = 0x200E;
    const uint16_t MOTOR_ENABLE = 0x0008;
    const uint16_t MOTOR_DISABLE = 0x0007;
    const uint16_t CONTROL_MODE = 0x200D;
    const uint16_t VEL_MODE = 0x0003;
    const uint16_t SET_L_RPM = 0x2088;
    const uint16_t GET_RPM = 0x20AB;
    const uint16_t GET_ENCODER_PULSE = 0x20A6;

    std::string port_;
    int baudrate_;
    uint8_t modbus_id_;
    boost::asio::io_service io_;
    boost::asio::serial_port serial_;
    std::mutex lock_;

    // Helper functions
    std::vector<uint8_t> build_write_cmd(uint16_t addr, uint16_t value) {
        std::vector<uint8_t> cmd;
        cmd.push_back(modbus_id_);
        cmd.push_back(WRITE);
        append_uint16(cmd, addr);
        append_uint16(cmd, value);
        append_crc(cmd);
        return cmd;
    }

    void append_uint16(std::vector<uint8_t>& v, uint16_t val) {
        v.push_back((val >> 8) & 0xFF);
        v.push_back(val & 0xFF);
    }

    void append_int16(std::vector<uint8_t>& v, int16_t val) {
        v.push_back((val >> 8) & 0xFF);
        v.push_back(val & 0xFF);
    }

    void append_crc(std::vector<uint8_t>& v) {
        uint16_t crc = calculate_crc(v.data(), v.size());
        v.push_back(crc & 0xFF);
        v.push_back((crc >> 8) & 0xFF);
    }

    uint16_t calculate_crc(const uint8_t* data, size_t len) {
        uint16_t crc = 0xFFFF;
        for (size_t i = 0; i < len; ++i) {
            crc ^= data[i];
            for (int j = 0; j < 8; ++j) {
                if (crc & 1)
                    crc = (crc >> 1) ^ 0xA001;
                else
                    crc >>= 1;
            }
        }
        return crc;
    }

    bool send_command(const std::vector<uint8_t>& cmd, size_t read_bytes) {
        std::lock_guard<std::mutex> guard(lock_);
        try {
            serial_.flush();
            boost::asio::write(serial_, boost::asio::buffer(cmd));
            if (read_bytes > 0) {
                std::vector<uint8_t> resp(read_bytes);
                boost::asio::read(serial_, boost::asio::buffer(resp));
                if (resp.size() == read_bytes &&
                    calculate_crc(resp.data(), resp.size() - 2) == (resp[resp.size() - 2] | (resp[resp.size() - 1] << 8))) {
                    return true;
                }
                return false;
            }
            return true;
        } catch (...) {
            std::cerr << "Serial communication error" << std::endl;
            return false;
        }
    }

    std::vector<uint8_t> read_response(const std::vector<uint8_t>& cmd, size_t read_bytes) {
        std::lock_guard<std::mutex> guard(lock_);
        try {
            serial_.flush();
            boost::asio::write(serial_, boost::asio::buffer(cmd));
            std::vector<uint8_t> resp(read_bytes);
            boost::asio::read(serial_, boost::asio::buffer(resp));
            if (resp.size() == read_bytes &&
                calculate_crc(resp.data(), resp.size() - 2) == (resp[resp.size() - 2] | (resp[resp.size() - 1] << 8))) {
                return resp;
            }
            return {};
        } catch (...) {
            std::cerr << "Serial communication error" << std::endl;
            return {};
        }
    }

    int16_t int16_from(const std::vector<uint8_t>& v, size_t idx) {
        return (v[idx] << 8) | v[idx + 1];
    }

    int32_t int32_from(const std::vector<uint8_t>& v, size_t idx) {
        return (v[idx] << 24) | (v[idx + 1] << 16) | (v[idx + 2] << 8) | v[idx + 3];
    }
};