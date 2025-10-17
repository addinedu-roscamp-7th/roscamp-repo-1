from setuptools import find_packages, setup

package_name = 'pickee_mobile'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'shopee_interfaces', 'numpy', 'scipy', 'tf2_ros', 'geometry_msgs'],
    zip_safe=True,
    maintainer='lim',
    maintainer_email='f499314e@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mobile_controller = pickee_mobile.mobile_controller:main',
            'path_planning_component = pickee_mobile.path_planning_component:main',
            'mock_speed_control_publisher = pickee_mobile.mock.mock_speed_control_publisher:main',
            'mock_move_to_location_client = pickee_mobile.mock.mock_move_to_location_client:main',
            'mock_update_global_path_client = pickee_mobile.mock.mock_update_global_path_client:main',
            'mock_pose_subscriber = pickee_mobile.mock.mock_pose_subscriber:main',
            'mock_arrival_and_move_status_subscriber = pickee_mobile.mock.mock_arrival_and_move_status_subscriber:main',
            'test_pose_subscriber = pickee_mobile.topic_test.pose_sub:main',
            'test_velocity_controller = pickee_mobile.topic_test.control_vel:main',
        ],
    },
)
