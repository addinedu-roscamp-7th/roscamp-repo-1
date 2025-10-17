# VicPinky의 ROS 프로젝트를 CPP로 변환한 레포지토리입니다.

### ROS 의존성 설치

```
rosdep install --from-paths src --ignore-src -r -y
```
이후 빌드가 안된다면 오류메세지에 뜬 의존성을 추가적으로 설치해주세요.

### 빌드

```
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
// domain id 설정 (상황에 따라 설정할 것)
// venv 설정 (상황에 따라 설정할 것)
```

### vicpinky 연결
와이파이 연결 : vicpinky_ee94 / pw : vicpinky

도메인 아이디 알아보기
echo $ROS_DOMAIN_ID
export ROS_DOMAIN_ID=31 (예시입니다.)

* ssh [robot] 연결
ssh vic@192.168.5.1 / pw : 1

* robot 와이파이 연결
./wifi_setup.sh
SSID : addinedu_201class_2-5G
PW :201class2!

* 와이파이 연결 확인
ping 8.8.8.8

* 현재 와이파이 id 확인
iwgetid -r


### 실행

[robot]
- **bringup** : ros2 launch vicpinky_bringup bringup.launch.xml
- **slam 실행** : ros2 launch vicpinky_navigation map_building.launch.xml
- **맵 저장** : ros2 run nav2_map_server map_saver_cli -f "저장할 맵 이름"
- **네비게이션 실행** : ros2 launch vicpinky_navigation bringup_launch.xml map:=저장한_맵.yaml use-sim-time:=True
(use_sim_time:=True 는 gazebo일 때만)

[pc]
- **gazebo** ros2 launch vicpinky_bringup gazebo_bringup.launch.xml
- **rviz** : ros2 launch vicpinky_navigation map_view.launch.xml
- **키조작** : ros2 run teleop_twist_keyboard teleop_twist_keyboard
- **rviz 실행** : ros2 launch vicpinky_navigation nav2_view.launch.xml

---

