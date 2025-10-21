from setuptools import find_packages, setup
import os, glob

package_name = 'pickee_mobile'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', '*launch.*'))),
        ('share/' + package_name + '/params', glob.glob(os.path.join('params', '*.yaml'))),
        ('share/' + package_name + '/map', glob.glob(os.path.join('map', '*.yaml')) + glob.glob(os.path.join('map', '*.pgm'))),
        ('share/' + package_name + '/models/desk/meshes', glob.glob(os.path.join('models', 'desk', 'meshes', '*'))),
        ('share/' + package_name + '/models/desk', [os.path.join('models', 'desk', 'model.config'), os.path.join('models', 'desk', 'model.sdf')]),
        ('share/' + package_name + '/models/factory_L1/meshes', glob.glob(os.path.join('models', 'factory_L1', 'meshes', '*'))),
        ('share/' + package_name + '/models/factory_L1', [os.path.join('models', 'factory_L1', 'model.config'), os.path.join('models', 'factory_L1', 'model.sdf')]),
        ('share/' + package_name + '/models/shelf/meshes', glob.glob(os.path.join('models', 'shelf', 'meshes', '*'))),
        ('share/' + package_name + '/models/shelf', [os.path.join('models', 'shelf', 'model.config'), os.path.join('models', 'shelf', 'model.sdf')]),
        ('share/' + package_name + '/models/shopee_map/meshes', glob.glob(os.path.join('models', 'shopee_map', 'meshes', '*'))),
        ('share/' + package_name + '/models/shopee_map', [os.path.join('models', 'shopee_map', 'model.config'), os.path.join('models', 'shopee_map', 'model.sdf')]),
        ('share/' + package_name + '/meshes/collision', glob.glob(os.path.join('meshes', 'collision', '*'))),
        ('share/' + package_name + '/meshes/visual', glob.glob(os.path.join('meshes', 'visual', '*'))),
        ('share/' + package_name + '/urdf', glob.glob(os.path.join('urdf', '*.xacro'))),
        ('share/' + package_name + '/worlds', glob.glob(os.path.join('worlds', '*.world'))),
        ('share/' + package_name + '/rviz', glob.glob(os.path.join('rviz', '*.rviz'))),
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
            'control_vel = pickee_mobile.topic_test.control_vel:main',
            'bringup = pickee_mobile.bringup:main',
            'get_clicked = pickee_mobile.goal_test.get_clicked:main',
            'get_clicked_move = pickee_mobile.goal_test.get_send_goal:main',
            'custom_goal_move = pickee_mobile.goal_test.custom_goal:main',
            'get_amcl_pose = pickee_mobile.topic_test.get_amcl_pose:main',
            'pub_pose = pickee_mobile.topic_test.pub_pose:main',
            'pub_cmd_vel = pickee_mobile.topic_test.pub_cmd_vel:main',
            ''
        ],
    },
)
