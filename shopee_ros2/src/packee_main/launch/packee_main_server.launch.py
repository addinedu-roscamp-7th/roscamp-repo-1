from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='packee_main',
            executable='packee_check_availability',
            name='packee_check_availability',
            output='screen'
        ),
        Node(
            package='packee_main',
            executable='packee_packing',
            name='packee_packing',
            output='screen'
        ),
        Node(
            package='packee_main',
            executable='packee_state_manager',
            name='packee_state_manager',
            output='screen'
        )
    ])