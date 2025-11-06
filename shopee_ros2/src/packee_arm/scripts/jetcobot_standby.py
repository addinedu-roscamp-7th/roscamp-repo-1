import time

from pymycobot.mycobot280 import MyCobot280

mc = MyCobot280('/dev/ttyUSB0', 1000000)

mc.power_on()                          # 전원 ON
mc.init_gripper()                      # 그리퍼 초기화 (원점 복귀)
time.sleep(3)

mc.send_angles([0, 0, 0, 0, 0, 0], 50)  # 팔을 제자리(기본자세)로 이동
mc.set_gripper_value(0, 50)            # 그리퍼 완전 열림
