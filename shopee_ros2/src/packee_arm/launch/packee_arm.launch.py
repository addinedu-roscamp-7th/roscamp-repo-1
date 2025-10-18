from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
    progress_publish_interval_arg = DeclareLaunchArgument(
        'progress_publish_interval',
        default_value='0.2',
        description='상태 토픽 발행 주기 (초)')
    command_timeout_sec_arg = DeclareLaunchArgument(
        'command_timeout_sec',
        default_value='5.0',
        description='명령 타임아웃 (초)')
    run_mock_main_arg = DeclareLaunchArgument(
        'run_mock_main',
        default_value='false',
        description='Packee Main 모의 노드를 함께 실행')

    # Packee Arm Controller 노드 정의
    controller_node = Node(
        package='packee_arm',
        executable='packee_arm_controller',
        name='packee_arm_controller',
        output='screen',
        parameters=[{
            'servo_gain_xy': LaunchConfiguration('servo_gain_xy'),
            'servo_gain_z': LaunchConfiguration('servo_gain_z'),
            'servo_gain_yaw': LaunchConfiguration('servo_gain_yaw'),
            'cnn_confidence_threshold': LaunchConfiguration('cnn_confidence_threshold'),
            'max_translation_speed': LaunchConfiguration('max_translation_speed'),
            'max_yaw_speed_deg': LaunchConfiguration('max_yaw_speed_deg'),
            'gripper_force_limit': LaunchConfiguration('gripper_force_limit'),
            'progress_publish_interval': LaunchConfiguration('progress_publish_interval'),
            'command_timeout_sec': LaunchConfiguration('command_timeout_sec')
        }])

    # Packee Main 모의 노드 정의 (옵션)
    mock_main_group = GroupAction([
        Node(
            package='packee_arm',
            executable='mock_packee_main',
            name='mock_packee_main',
            output='screen')
    ], condition=IfCondition(LaunchConfiguration('run_mock_main')))

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
        run_mock_main_arg,
        controller_node,
        mock_main_group
    ])
