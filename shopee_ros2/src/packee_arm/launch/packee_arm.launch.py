import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import GroupAction
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    # Packee Arm Controller 파라미터를 런치 인자로 노출
    servo_gain_xy_arg = DeclareLaunchArgument(
        'servo_gain_xy',
        default_value='0.03',
        description='X/Y 축 시각 서보 게인')
    servo_gain_z_arg = DeclareLaunchArgument(
        'servo_gain_z',
        default_value='0.03',
        description='Z 축 시각 서보 게인')
    servo_gain_yaw_arg = DeclareLaunchArgument(
        'servo_gain_yaw',
        default_value='0.03',
        description='Yaw 축 시각 서보 게인')
    cnn_confidence_threshold_arg = DeclareLaunchArgument(
        'cnn_confidence_threshold',
        default_value='0.75',
        description='CNN 신뢰도 임계값')
    max_translation_speed_arg = DeclareLaunchArgument(
        'max_translation_speed',
        default_value='0.2',
        description='엔드이펙터 최대 병진 속도 (m/s)')
    max_yaw_speed_deg_arg = DeclareLaunchArgument(
        'max_yaw_speed_deg',
        default_value='15.0',
        description='엔드이펙터 최대 yaw 속도 (deg/s)')
    gripper_force_limit_arg = DeclareLaunchArgument(
        'gripper_force_limit',
        default_value='35.0',
        description='그리퍼 힘 제한 (N)')
    preset_pose_cart_view_arg = DeclareLaunchArgument(
        'preset_pose_cart_view',
        default_value='[0.16, 0.0, 0.18, 0.0]',
        description='카트 확인 자세 (x,y,z,yaw_deg)')
    preset_pose_standby_arg = DeclareLaunchArgument(
        'preset_pose_standby',
        default_value='[0.10, 0.0, 0.14, 0.0]',
        description='대기 자세 (x,y,z,yaw_deg)')
    progress_publish_interval_arg = DeclareLaunchArgument(
        'progress_publish_interval',
        default_value='0.2',
        description='상태 토픽 발행 주기 (초)')
    command_timeout_sec_arg = DeclareLaunchArgument(
        'command_timeout_sec',
        default_value='5.0',
        description='명령 타임아웃 (초)')
    left_arm_velocity_topic_arg = DeclareLaunchArgument(
        'left_arm_velocity_topic',
        default_value='/packee/jetcobot/left/cmd_vel',
        description='좌측 팔 속도 명령 토픽')
    right_arm_velocity_topic_arg = DeclareLaunchArgument(
        'right_arm_velocity_topic',
        default_value='/packee/jetcobot/right/cmd_vel',
        description='우측 팔 속도 명령 토픽')
    left_gripper_topic_arg = DeclareLaunchArgument(
        'left_gripper_topic',
        default_value='/packee/jetcobot/left/gripper_cmd',
        description='좌측 팔 그리퍼 명령 토픽')
    right_gripper_topic_arg = DeclareLaunchArgument(
        'right_gripper_topic',
        default_value='/packee/jetcobot/right/gripper_cmd',
        description='우측 팔 그리퍼 명령 토픽')
    velocity_frame_id_arg = DeclareLaunchArgument(
        'velocity_frame_id',
        default_value='packee_base',
        description='TwistStamped frame_id 설정')
    run_mock_main_arg = DeclareLaunchArgument(
        'run_mock_main',
        default_value='false',
        description='Packee Main 모의 노드를 함께 실행')
    run_jetcobot_bridge_arg = DeclareLaunchArgument(
        'run_jetcobot_bridge',
        default_value='true',
        description='JetCobot 브릿지 노드 실행 여부')
    run_packee_main_arg = DeclareLaunchArgument(
        'run_packee_main',
        default_value='false',
        description='Packee Main 패키지 노드 실행')
    use_pymycobot_controller_arg = DeclareLaunchArgument(
        'use_pymycobot_controller',
        default_value='false',
        description='pymycobot 기반 Python 컨트롤러 실행')
    left_serial_port_arg = DeclareLaunchArgument(
        'left_serial_port',
        default_value='/dev/ttyUSB0',
        description='좌측 JetCobot 시리얼 포트')
    right_serial_port_arg = DeclareLaunchArgument(
        'right_serial_port',
        default_value='/dev/ttyUSB1',
        description='우측 JetCobot 시리얼 포트 (단일 팔 환경이면 비워두세요)')
    jetcobot_move_speed_arg = DeclareLaunchArgument(
        'jetcobot_move_speed',
        default_value='30',
        description='JetCobot sync_send_coords 속도 (0~100)')
    jetcobot_command_period_arg = DeclareLaunchArgument(
        'jetcobot_command_period',
        default_value='0.15',
        description='JetCobot 브릿지 적분 주기 (초)')
    pymycobot_baud_rate_arg = DeclareLaunchArgument(
        'pymycobot_baud_rate',
        default_value='1000000',
        description='pymycobot 직렬 통신 속도')
    pymycobot_move_speed_arg = DeclareLaunchArgument(
        'pymycobot_move_speed',
        default_value='40',
        description='pymycobot 좌표 명령 속도 (0~100)')
    pymycobot_arm_sides_arg = DeclareLaunchArgument(
        'pymycobot_arm_sides',
        default_value='left',
        description='사용할 팔 목록 (쉼표 구분, 예: left,right)')
    pymycobot_approach_offset_arg = DeclareLaunchArgument(
        'pymycobot_approach_offset',
        default_value='0.05',
        description='픽업/담기 접근 시 상승 높이 (m)')
    pymycobot_lift_offset_arg = DeclareLaunchArgument(
        'pymycobot_lift_offset',
        default_value='0.06',
        description='픽업 후 상승 높이 (m)')
    pymycobot_gripper_open_arg = DeclareLaunchArgument(
        'pymycobot_gripper_open_value',
        default_value='100',
        description='그리퍼 개방 값')
    pymycobot_gripper_close_arg = DeclareLaunchArgument(
        'pymycobot_gripper_close_value',
        default_value='20',
        description='그리퍼 파지 값')

    # Packee Arm Controller 노드 정의
    controller_node = Node(
        package='packee_arm',
        executable='packee_arm_controller',
        name='packee_arm_controller',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_pymycobot_controller')),
        parameters=[{
            'servo_gain_xy': LaunchConfiguration('servo_gain_xy'),
            'servo_gain_z': LaunchConfiguration('servo_gain_z'),
            'servo_gain_yaw': LaunchConfiguration('servo_gain_yaw'),
            'cnn_confidence_threshold': LaunchConfiguration('cnn_confidence_threshold'),
            'max_translation_speed': LaunchConfiguration('max_translation_speed'),
            'max_yaw_speed_deg': LaunchConfiguration('max_yaw_speed_deg'),
            'gripper_force_limit': LaunchConfiguration('gripper_force_limit'),
            'progress_publish_interval': LaunchConfiguration('progress_publish_interval'),
            'command_timeout_sec': LaunchConfiguration('command_timeout_sec'),
            'left_arm_velocity_topic': LaunchConfiguration('left_arm_velocity_topic'),
            'right_arm_velocity_topic': LaunchConfiguration('right_arm_velocity_topic'),
            'left_gripper_topic': LaunchConfiguration('left_gripper_topic'),
            'right_gripper_topic': LaunchConfiguration('right_gripper_topic'),
            'velocity_frame_id': LaunchConfiguration('velocity_frame_id')
        }])

    # Packee Main 모의 노드 정의 (옵션)
    mock_main_group = GroupAction([
        Node(
            package='packee_arm',
            executable='mock_packee_main',
            name='mock_packee_main',
            output='screen')
    ], condition=IfCondition(LaunchConfiguration('run_mock_main')))

    packee_main_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('packee_main'),
                'launch',
                'packee_main_server.launch.py')),
        condition=IfCondition(LaunchConfiguration('run_packee_main')))

    jetcobot_bridge_node = Node(
        package='packee_arm',
        executable='jetcobot_bridge.py',
        name='jetcobot_bridge',
        output='screen',
        parameters=[{
            'left_serial_port': LaunchConfiguration('left_serial_port'),
            'right_serial_port': LaunchConfiguration('right_serial_port'),
            'left_velocity_topic': LaunchConfiguration('left_arm_velocity_topic'),
            'right_velocity_topic': LaunchConfiguration('right_arm_velocity_topic'),
            'left_gripper_topic': LaunchConfiguration('left_gripper_topic'),
            'right_gripper_topic': LaunchConfiguration('right_gripper_topic'),
            'command_period_sec': LaunchConfiguration('jetcobot_command_period'),
            'move_speed': LaunchConfiguration('jetcobot_move_speed')
        }],
        condition=IfCondition(LaunchConfiguration('run_jetcobot_bridge')))

    pymycobot_node = Node(
        package='packee_arm',
        executable='pymycobot',
        name='pymycobot_arm_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('left_serial_port'),
            'baud_rate': LaunchConfiguration('pymycobot_baud_rate'),
            'move_speed': LaunchConfiguration('pymycobot_move_speed'),
            'arm_sides': LaunchConfiguration('pymycobot_arm_sides'),
            'approach_offset_m': LaunchConfiguration('pymycobot_approach_offset'),
            'lift_offset_m': LaunchConfiguration('pymycobot_lift_offset'),
            'preset_pose_cart_view': LaunchConfiguration('preset_pose_cart_view'),
            'preset_pose_standby': LaunchConfiguration('preset_pose_standby'),
            'gripper_open_value': LaunchConfiguration('pymycobot_gripper_open_value'),
            'gripper_close_value': LaunchConfiguration('pymycobot_gripper_close_value')
        }],
        condition=IfCondition(LaunchConfiguration('use_pymycobot_controller')))

    return LaunchDescription([
        servo_gain_xy_arg,
        servo_gain_z_arg,
        servo_gain_yaw_arg,
        cnn_confidence_threshold_arg,
        max_translation_speed_arg,
        max_yaw_speed_deg_arg,
        gripper_force_limit_arg,
        progress_publish_interval_arg,
        command_timeout_sec_arg,
        preset_pose_cart_view_arg,
        preset_pose_standby_arg,
        left_arm_velocity_topic_arg,
        right_arm_velocity_topic_arg,
        left_gripper_topic_arg,
        right_gripper_topic_arg,
        velocity_frame_id_arg,
        run_mock_main_arg,
        run_jetcobot_bridge_arg,
        run_packee_main_arg,
        use_pymycobot_controller_arg,
        left_serial_port_arg,
        right_serial_port_arg,
        jetcobot_move_speed_arg,
        jetcobot_command_period_arg,
        pymycobot_baud_rate_arg,
        pymycobot_move_speed_arg,
        pymycobot_arm_sides_arg,
        pymycobot_approach_offset_arg,
        pymycobot_lift_offset_arg,
        pymycobot_gripper_open_arg,
        pymycobot_gripper_close_arg,
        controller_node,
        mock_main_group,
        packee_main_launch,
        jetcobot_bridge_node,
        pymycobot_node
    ])
