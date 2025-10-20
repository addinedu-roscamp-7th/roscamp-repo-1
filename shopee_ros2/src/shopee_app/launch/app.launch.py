from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package='shopee_app',
                executable='shopee_app_gui',
                name='shopee_app_gui',
                output='screen',
            ),
        ]
    )
