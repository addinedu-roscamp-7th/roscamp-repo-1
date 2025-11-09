import time

from pymycobot.mycobot280 import MyCobot280
# MyCobot 연결 (포트와 보드레이트 확인)
mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)

# 전원 및 초기화
# mc.power_on()
# time.sleep(2)
# mc.init_gripper()
# time.sleep(3)

# 기본 자세(예: 안전한 초기 위치)로 이동
# 형식: send_coords([x, y, z, rx, ry, rz], speed)
# 단위: mm, degree
mc.send_coords([107.7, -20.6, 187.6, -178.91, 2.51, -87.48], 30)
time.sleep(5)

# 그리퍼 완전 열기 (0~100)
