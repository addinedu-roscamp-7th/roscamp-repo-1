from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """
    Pickee Mobile Controller 실행을 위한 Launch 파일
    """
    
    # 패키지 경로
    pkg_pickee_mobile = FindPackageShare('pickee_mobile_wonho')
    
    # Launch 인자 선언
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='1',
        description='로봇 ID'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='시뮬레이션 시간 사용 여부'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            pkg_pickee_mobile, 'config', 'pickee_mobile_params.yaml'
        ]),
        description='파라미터 설정 파일 경로'
    )
    
    # Pickee Mobile Controller 노드
    pickee_mobile_controller_node = Node(
        package='pickee_mobile_wonho',
        executable='pickee_mobile_wonho_controller',
        name='pickee_mobile_controller',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'robot_id': LaunchConfiguration('robot_id'),
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }
        ],
        remappings=[
            # 토픽 리매핑 (필요시)
            ('/cmd_vel', '/cmd_vel'),
            ('/scan', '/scan'),
            ('/odom', '/odom'),
            ('/imu', '/imu'),
        ]
    )
    
    # 시작 메시지
    start_message = LogInfo(
        msg='=== Pickee Mobile Controller 시작 ==='
    )
    
    return LaunchDescription([
        # Launch 인자들
        robot_id_arg,
        use_sim_time_arg,
        config_file_arg,
        
        # 시작 메시지
        start_message,
        
        # 노드들
        pickee_mobile_controller_node,
    ])