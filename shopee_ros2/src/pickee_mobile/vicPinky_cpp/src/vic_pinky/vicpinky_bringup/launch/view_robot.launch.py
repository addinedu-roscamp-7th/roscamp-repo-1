from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare arguments
    declare_namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for the robot'
    )
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # Get the path to the RViz config file
    # Using urdf_config.rviz from vicpinky_description as a base
    rviz_config_dir = os.path.join(get_package_share_directory('vicpinky_description'), 'rviz')
    rviz_config_file = PathJoinSubstitution([rviz_config_dir, 'urdf_config.rviz'])

    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        remappings=[
            ('/tf', [LaunchConfiguration('namespace'), '/tf']),
            ('/tf_static', [LaunchConfiguration('namespace'), '/tf_static'])
        ],
        namespace=LaunchConfiguration('namespace') # Using the 'namespace' argument of Node
    )

    return LaunchDescription([
        declare_namespace_arg,
        declare_use_sim_time_arg,
        rviz_node
    ])
