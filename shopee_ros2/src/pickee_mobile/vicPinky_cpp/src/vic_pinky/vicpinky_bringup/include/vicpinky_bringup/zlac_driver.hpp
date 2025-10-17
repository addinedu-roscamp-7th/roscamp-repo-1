#pragma once
#include <string>
#include <utility>

class ZLACDriver {
public:
    ZLACDriver(const std::string& port, int baudrate, uint8_t modbus_id);
    bool begin();
    bool enable();
    bool disable();
    bool set_vel_mode();
    bool set_double_rpm(int l_rpm, int r_rpm);
    std::pair<double, double> get_rpm();
    std::pair<int, int> get_position();
    void terminate();
};