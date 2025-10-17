from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="packee_main_controller",
                executable="packee_main_service"
            ),
            Node(
                package="packee_main_controller",
                executable="packee_main_topic"
            )
        ]
    )