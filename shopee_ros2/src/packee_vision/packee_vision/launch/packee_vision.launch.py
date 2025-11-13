from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='packee_vision',
            executable='check_cart_presence',
            name='check_cart_presence',
            output='screen'
        ),
        Node(
            package='packee_vision',
            executable='detect_products_in_cart1',
            name='detect_products_in_cart1',
            output='screen'
        ),
        Node(
            package='packee_vision',
            executable='detect_products_in_cart2',
            name='detect_products_in_cart2',
            output='screen'
        ),
        Node(
            package='packee_vision',
            executable='verify_packing_complete',
            name='verify_packing_complete',
            output='screen'
        )
    ])