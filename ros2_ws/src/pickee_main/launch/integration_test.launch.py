#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    """통합 테스트용 Launch 파일"""
    
    # Launch arguments 선언
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='1',
        description='Robot ID for the test'
    )
    
    battery_threshold_available_arg = DeclareLaunchArgument(
        'battery_threshold_available',
        default_value='30.0',
        description='Battery threshold for available state'
    )
    
    battery_threshold_unavailable_arg = DeclareLaunchArgument(
        'battery_threshold_unavailable',
        default_value='10.0',
        description='Battery threshold for unavailable state'
    )
    
    return LaunchDescription([
        # Launch arguments
        robot_id_arg,
        battery_threshold_available_arg,
        battery_threshold_unavailable_arg,
        
        # Mock Mobile Node
        Node(
            package='pickee_main',
            executable='mock_mobile_node',
            name='mock_mobile_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': False,
            }]
        ),
        
        # Mock Arm Node
        Node(
            package='pickee_main',
            executable='mock_arm_node',
            name='mock_arm_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': False,
            }]
        ),
        
        # Mock Vision Node
        Node(
            package='pickee_main',
            executable='mock_vision_node',
            name='mock_vision_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': False,
            }]
        ),
        
        # Main Controller Node
        Node(
            package='pickee_main',
            executable='main_controller',
            name='pickee_main_controller',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'robot_id': LaunchConfiguration('robot_id'),
                'battery_threshold_available': LaunchConfiguration('battery_threshold_available'),
                'battery_threshold_unavailable': LaunchConfiguration('battery_threshold_unavailable'),
                'use_sim_time': False,
            }]
        ),
    ])


if __name__ == '__main__':
    generate_launch_description()