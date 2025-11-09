import time
from pymycobot.mycobot280 import MyCobot280

class ArmControl:
    """Hardware Abstraction Layer for the Pickee Arm."""

    def __init__(self, logger, port='/dev/ttyJETCOBOT', baudrate=1000000):
        self.mc = None
        self.logger = logger
        try:
            self.mc = MyCobot280(port, baudrate)
            self.logger.info("myCobot arm connected successfully.")
            if self.mc.is_power_on() != 1:
                self.mc.power_on()
                self.logger.info("Powering on the robot.")
            self.mc.focus_all_servos()
            self.logger.info("All servos focused.")
        except Exception as e:
            self.logger.error(f"Failed to connect to myCobot arm: {e}")
            raise e

    def is_connected(self):
        """Check if the arm is connected and powered on."""
        return self.mc and self.mc.is_power_on() == 1

    def get_angles(self):
        """현재 로봇 팔의 6축 관절 각도를 읽어옵니다."""
        if self.is_connected():
            return self.mc.get_angles()
        else:
            self.logger.error("myCobot arm is not connected. Cannot get angles.")
            return None

    def get_coords(self):
        """현재 로봇 팔의 6축 관절 각도를 읽어옵니다."""
        if self.is_connected():
            return self.mc.get_coords()
        else:
            self.logger.error("myCobot arm is not connected. Cannot get angles.")
            return None

    def move_to_joints(self, joint_angles, speed=40, timeout=5):
        """목표 관절 각도로 로봇 팔을 움직이고, 동작이 끝날 때까지 기다립니다."""
        if self.is_connected():
            self.logger.info(f"Executing move to: {joint_angles}")
            self.mc.send_angles(joint_angles, speed, timeout)
            self.logger.info("Move complete.")
        else:
            self.logger.error("myCobot arm is not connected. Cannot send command.")

    def move_to_coords(self, joint_angles, speed=40, timeout=5):
        """목표 관절 각도로 로봇 팔을 움직이고, 동작이 끝날 때까지 기다립니다."""
        if self.is_connected():
            self.logger.info(f"Executing move to: {joint_angles}")
            self.mc.send_coords(joint_angles, speed, timeout)
            self.logger.info("Move complete.")
        else:
            self.logger.error("myCobot arm is not connected. Cannot send command.")

    def control_gripper(self, value, speed=40):
        """그리퍼를 제어합니다. (100: 열기, 0: 닫기)"""
        if self.is_connected():
            action = "OPENING" if value == 100 else "CLOSING"
            self.logger.info(f"GRIPPER {action} (value: {value})")
            self.mc.set_gripper_value(value, speed)
            time.sleep(1.0)
            self.logger.info("Gripper action complete.")
        else:
            self.logger.error("myCobot arm is not connected. Cannot control gripper.")

    def release_servos(self):
        """모든 서보의 토크를 해제합니다."""
        if self.mc:
            self.logger.info("Releasing all servos.")
            self.mc.release_all_servos()