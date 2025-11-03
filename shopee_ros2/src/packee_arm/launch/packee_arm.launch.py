from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # pymycobot 우측/좌측 노드 실행 여부
    run_pymycobot_dual_arg = DeclareLaunchArgument(
        'run_pymycobot_dual',
        default_value='true',
        description='듀얼 팔 pymycobot 노드 실행 여부')
    run_pymycobot_left_arg = DeclareLaunchArgument(
        'run_pymycobot_left',
        default_value='false',
        description='좌측 팔 pymycobot 노드 실행 여부 (듀얼 모드 비활성화 시 사용)')
    run_pymycobot_right_arg = DeclareLaunchArgument(
        'run_pymycobot_right',
        default_value='false',
        description='우측 팔 pymycobot 노드 실행 여부 (듀얼 모드 비활성화 시 사용)')

    # JetCobot 브릿지 실행 여부
    run_jetcobot_bridge_arg = DeclareLaunchArgument(
        'run_jetcobot_bridge',
        default_value='true',
        description='JetCobot 브릿지 노드 실행 여부')

    # 공통 팔 동작 파라미터
    preset_pose_cart_view_arg = DeclareLaunchArgument(
        'preset_pose_cart_view',
        default_value='[0.16, 0.0, 0.18, 0.0, 0.0, 0.0]',
        description='카트 확인 자세 (x,y,z,rx,ry,rz)')
    preset_pose_standby_arg = DeclareLaunchArgument(
        'preset_pose_standby',
        default_value='[0.10, 0.0, 0.14, 0.0, 0.0, 0.0]',
        description='대기 자세 (x,y,z,rx,ry,rz)')
    pymycobot_baud_rate_arg = DeclareLaunchArgument(
        'pymycobot_baud_rate',
        default_value='1000000',
        description='pymycobot 직렬 통신 속도')
    pymycobot_move_speed_arg = DeclareLaunchArgument(
        'pymycobot_move_speed',
        default_value='40',
        description='pymycobot 좌표 명령 속도 (0~100)')
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
        default_value='0',
        description='그리퍼 파지 값')
    pymycobot_enabled_arms_arg = DeclareLaunchArgument(
        'pymycobot_enabled_arms',
        default_value='left,right',
        description='듀얼 노드가 제어할 팔 목록 (쉼표 구분)')
    pose_status_topic_arg = DeclareLaunchArgument(
        'pose_status_topic',
        default_value='/packee/arm/pose_status',
        description='자세 상태 토픽 이름')
    pick_status_topic_arg = DeclareLaunchArgument(
        'pick_status_topic',
        default_value='/packee/arm/pick_status',
        description='픽업 상태 토픽 이름')
    place_status_topic_arg = DeclareLaunchArgument(
        'place_status_topic',
        default_value='/packee/arm/place_status',
        description='담기 상태 토픽 이름')

    # 좌/우 팔 별 시리얼 포트
    left_serial_port_arg = DeclareLaunchArgument(
        'left_serial_port',
        default_value='/dev/ttyUSB1',
        description='좌측 myCobot 시리얼 포트')
    right_serial_port_arg = DeclareLaunchArgument(
        'right_serial_port',
        default_value='/dev/ttyUSB0',
        description='우측 myCobot 시리얼 포트')

    # JetCobot 브릿지 토픽/파라미터
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
        description='좌측 그리퍼 명령 토픽')
    right_gripper_topic_arg = DeclareLaunchArgument(
        'right_gripper_topic',
        default_value='/packee/jetcobot/right/gripper_cmd',
        description='우측 그리퍼 명령 토픽')
    jetcobot_move_speed_arg = DeclareLaunchArgument(
        'jetcobot_move_speed',
        default_value='40',
        description='JetCobot sync_send_coords 속도 (0~100)')
    jetcobot_command_period_arg = DeclareLaunchArgument(
        'jetcobot_command_period',
        default_value='0.15',
        description='JetCobot 브릿지 적분 주기 (초)')
    jetcobot_workspace_radial_arg = DeclareLaunchArgument(
        'jetcobot_workspace_radial',
        default_value='0.28',
        description='허용 수평 반경 (m)')
    jetcobot_workspace_z_min_arg = DeclareLaunchArgument(
        'jetcobot_workspace_z_min',
        default_value='0.05',
        description='허용 Z 최소 높이 (m)')
    jetcobot_workspace_z_max_arg = DeclareLaunchArgument(
        'jetcobot_workspace_z_max',
        default_value='0.30',
        description='허용 Z 최대 높이 (m)')

    pymycobot_dual_node = Node(
        package='packee_arm',
        executable='pymycobot_dual',
        name='pymycobot_arm_node',
        output='screen',
        parameters=[{
            'enabled_arms': LaunchConfiguration('pymycobot_enabled_arms'),
            'serial_port_left': LaunchConfiguration('left_serial_port'),
            'serial_port_right': LaunchConfiguration('right_serial_port'),
            'baud_rate': LaunchConfiguration('pymycobot_baud_rate'),
            'move_speed': LaunchConfiguration('pymycobot_move_speed'),
            'approach_offset_m': LaunchConfiguration('pymycobot_approach_offset'),
            'lift_offset_m': LaunchConfiguration('pymycobot_lift_offset'),
            'preset_pose_cart_view': LaunchConfiguration('preset_pose_cart_view'),
            'preset_pose_standby': LaunchConfiguration('preset_pose_standby'),
            'gripper_open_value': LaunchConfiguration('pymycobot_gripper_open_value'),
            'gripper_close_value': LaunchConfiguration('pymycobot_gripper_close_value'),
            'pose_status_topic': LaunchConfiguration('pose_status_topic'),
            'pick_status_topic': LaunchConfiguration('pick_status_topic'),
            'place_status_topic': LaunchConfiguration('place_status_topic')
        }],
        condition=IfCondition(LaunchConfiguration('run_pymycobot_dual')))

    pymycobot_left_node = Node(
        package='packee_arm',
        executable='pymycobot_left',
        name='pymycobot_left_arm_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('left_serial_port'),
            'baud_rate': LaunchConfiguration('pymycobot_baud_rate'),
            'move_speed': LaunchConfiguration('pymycobot_move_speed'),
            'arm_sides': 'left',
            'approach_offset_m': LaunchConfiguration('pymycobot_approach_offset'),
            'lift_offset_m': LaunchConfiguration('pymycobot_lift_offset'),
            'preset_pose_cart_view': LaunchConfiguration('preset_pose_cart_view'),
            'preset_pose_standby': LaunchConfiguration('preset_pose_standby'),
            'gripper_open_value': LaunchConfiguration('pymycobot_gripper_open_value'),
            'gripper_close_value': LaunchConfiguration('pymycobot_gripper_close_value'),
            'pose_status_topic': LaunchConfiguration('pose_status_topic'),
            'pick_status_topic': LaunchConfiguration('pick_status_topic'),
            'place_status_topic': LaunchConfiguration('place_status_topic')
        }],
        condition=IfCondition(LaunchConfiguration('run_pymycobot_left')))

    pymycobot_right_node = Node(
        package='packee_arm',
        executable='pymycobot_right',
        name='pymycobot_right_arm_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('right_serial_port'),
            'baud_rate': LaunchConfiguration('pymycobot_baud_rate'),
            'move_speed': LaunchConfiguration('pymycobot_move_speed'),
            'arm_sides': 'right',
            'approach_offset_m': LaunchConfiguration('pymycobot_approach_offset'),
            'lift_offset_m': LaunchConfiguration('pymycobot_lift_offset'),
            'preset_pose_cart_view': LaunchConfiguration('preset_pose_cart_view'),
            'preset_pose_standby': LaunchConfiguration('preset_pose_standby'),
            'gripper_open_value': LaunchConfiguration('pymycobot_gripper_open_value'),
            'gripper_close_value': LaunchConfiguration('pymycobot_gripper_close_value'),
            'pose_status_topic': LaunchConfiguration('pose_status_topic'),
            'pick_status_topic': LaunchConfiguration('pick_status_topic'),
            'place_status_topic': LaunchConfiguration('place_status_topic')
        }],
        condition=IfCondition(LaunchConfiguration('run_pymycobot_right')))

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
            'move_speed': LaunchConfiguration('jetcobot_move_speed'),
            'workspace_radial': LaunchConfiguration('jetcobot_workspace_radial'),
            'workspace_z_min': LaunchConfiguration('jetcobot_workspace_z_min'),
            'workspace_z_max': LaunchConfiguration('jetcobot_workspace_z_max'),
            'default_pose_cart_view': LaunchConfiguration('preset_pose_cart_view'),
            'default_pose_standby': LaunchConfiguration('preset_pose_standby'),
            'gripper_open_value': LaunchConfiguration('pymycobot_gripper_open_value'),
            'gripper_close_value': LaunchConfiguration('pymycobot_gripper_close_value')
        }],
        condition=IfCondition(LaunchConfiguration('run_jetcobot_bridge')))

    return LaunchDescription([
        run_pymycobot_dual_arg,
        run_pymycobot_left_arg,
        run_pymycobot_right_arg,
        run_jetcobot_bridge_arg,
        preset_pose_cart_view_arg,
        preset_pose_standby_arg,
        pymycobot_baud_rate_arg,
        pymycobot_move_speed_arg,
        pymycobot_approach_offset_arg,
        pymycobot_lift_offset_arg,
        pymycobot_gripper_open_arg,
        pymycobot_gripper_close_arg,
        pymycobot_enabled_arms_arg,
        pose_status_topic_arg,
        pick_status_topic_arg,
        place_status_topic_arg,
        left_serial_port_arg,
        right_serial_port_arg,
        left_arm_velocity_topic_arg,
        right_arm_velocity_topic_arg,
        left_gripper_topic_arg,
        right_gripper_topic_arg,
        jetcobot_move_speed_arg,
        jetcobot_command_period_arg,
        jetcobot_workspace_radial_arg,
        jetcobot_workspace_z_min_arg,
        jetcobot_workspace_z_max_arg,
        pymycobot_dual_node,
        pymycobot_left_node,
        pymycobot_right_node,
        jetcobot_bridge_node
    ])
