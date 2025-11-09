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
    entry_points = {
        'console_scripts': [
            # System
            'bringup = pickee_mobile.bringup:main',

            # Main Components
            'main_aruco_pub = pickee_mobile.main.main_aruco_pub:main', # ArUco 마커 퍼블리시 노드
            'main_aruco_pub_once = pickee_mobile.main.main_aruco_pub_once:main', # 도킹 1트
            'main_aruco_detect = pickee_mobile.main.main_aruco_detect:main', # ArUco 마커 인식 함수
            'mobile_controller = pickee_mobile.main.mobile_controller:main', # 모바일 로봇의 주요 제어 노드
            'mobile_vel_modifier = pickee_mobile.main.mobile_vel_modifier:main', # 속도 명령을 수정하는 노드
            'mobile_aruco_docking = pickee_mobile.main.mobile_aruco_docking:main', # ArUco 마커 도킹 노드
            'mobile_aruco_docking_once = pickee_mobile.main.mobile_aruco_docking_once:main', # 도킹 1트

            # Module Components 함수
            'module_go_strait = pickee_mobile.module.module_go_strait:main',
            'module_rotate = pickee_mobile.module.module_rotate:main',
            'module_aruco_detect = pickee_mobile.module.module_aruco_detect:main',
            'module_straight_rotate = pickee_mobile.module.module_straight_rotate:main',
            
            

            ####Test Nodes####
            # Goal / Navigation Test
            'get_clicked = pickee_mobile.test.goal_test.get_clicked:main',
            'get_clicked_move = pickee_mobile.test.goal_test.get_send_goal:main',
            'custom_goal_move = pickee_mobile.test.goal_test.custom_goal:main',
            'goal_send_client = pickee_mobile.test.goal_test.goal_send_client:main',
            'import_go_straight = pickee_mobile.test.goal_test.import_go_straight:main',

            # Topic Test
            'control_vel = pickee_mobile.test.topic_test.control_vel:main',
            'pub_pose = pickee_mobile.test.topic_test.pub_pose:main',
            'pub_cmd_vel = pickee_mobile.test.topic_test.pub_cmd_vel:main',
            'control_vel_teteop = pickee_mobile.test.topic_test.control_vel_teleop:main',
            'import_test_rotate_odom = pickee_mobile.test.topic_test.import_test_rotate_odom:main',
            'import_test_go_straight_odom = pickee_mobile.test.topic_test.import_test_go_straight_odom:main',

            # Mock / Simulation Test
            'mock_vel_modifier_publisher = pickee_mobile.test.mock_test.mock_vel_modifier_publisher:main',
            'mock_move_to_location_client = pickee_mobile.test.mock_test.mock_move_to_location_client:main',
            'mock_update_global_path_client = pickee_mobile.test.mock_test.mock_update_global_path_client:main',
            'mock_pose_subscriber = pickee_mobile.test.mock_test.mock_pose_subscriber:main',
            'mock_arrival_subscriber = pickee_mobile.test.mock_test.mock_arrival_subscriber:main',
            'mock_arrival_publisher = pickee_mobile.test.mock_test.mock_arrival_publisher:main',
            'mock_aruco_sub = pickee_mobile.test.mock_test.mock_aruco_sub:main',

            # Aruco Test
            'aruco_detect = pickee_mobile.test.aruco.aruco_detect:main',
            'aruco_follow_1 = pickee_mobile.test.aruco.aruco_follow_1:main',
            'aruco_follow_2 = pickee_mobile.test.aruco.aruco_follow_2:main',
            'aruco_follow_3 = pickee_mobile.test.aruco.aruco_follow_3:main',
            'aruco_follow_4 = pickee_mobile.test.aruco.aruco_follow_4:main',
            'aruco_pub = pickee_mobile.test.aruco.aruco_pub:main',
        ],
    }

)
