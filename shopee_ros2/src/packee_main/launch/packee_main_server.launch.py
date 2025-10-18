from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='packee_main',
            executable='packee_main_service',
            name='packee_main_service',
            output='screen'
        ),
        Node(
            package='packee_main',
            executable='packee_main_topic',
            name='packee_main_topic',
            output='screen'
        )
    ])