"""
Mock 환경 실행용 런치 파일

Main Service, Mock LLM, Mock Robot 노드를 한 번에 실행합니다.
환경 변수는 LaunchArgument로 제어하며 config.py와 정합성을 유지합니다.
"""
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """런치 설명을 생성한다."""
    api_host = LaunchConfiguration('api_host')
    api_port = LaunchConfiguration('api_port')
    llm_base_url = LaunchConfiguration('llm_base_url')
    db_url = LaunchConfiguration('db_url')
    robot_mode = LaunchConfiguration('mock_robot_mode')

    launch_arguments = [
        DeclareLaunchArgument(
            'api_host',
            default_value='0.0.0.0',
            description='Main Service TCP 서버 호스트',
        ),
        DeclareLaunchArgument(
            'api_port',
            default_value='5000',
            description='Main Service TCP 서버 포트',
        ),
        DeclareLaunchArgument(
            'llm_base_url',
            default_value='http://localhost:5001',
            description='Mock LLM 서버 기본 URL',
        ),
        DeclareLaunchArgument(
            'db_url',
            default_value='mysql+pymysql://shopee:shopee@localhost:3306/shopee',
            description='Main Service가 접속할 데이터베이스 URI',
        ),
        DeclareLaunchArgument(
            'mock_robot_mode',
            default_value='all',
            description="Mock Robot 실행 모드 (all | pickee | packee)",
        ),
    ]

    env_assignments = [
        SetEnvironmentVariable('SHOPEE_API_HOST', api_host),
        SetEnvironmentVariable('SHOPEE_API_PORT', api_port),
        SetEnvironmentVariable('SHOPEE_LLM_BASE_URL', llm_base_url),
        SetEnvironmentVariable('SHOPEE_DB_URL', db_url),
        SetEnvironmentVariable('SHOPEE_GUI_ENABLED', 'true'),
        SetEnvironmentVariable('SHOPEE_LOG_LEVEL', 'INFO'),
        SetEnvironmentVariable('PYTHONUNBUFFERED', '1'),
    ]

    # Mock LLM 노드
    mock_llm_node = Node(
        package='main_service',
        executable='mock_llm_server',
        name='mock_llm_server',
        output='screen',
    )
    mock_robot_node = Node(
        package='main_service',
        executable='mock_robot_node',
        name='mock_robot_node',
        output='screen',
        arguments=['--mode', robot_mode],
    )

    main_service_node = Node(
        package='main_service',
        executable='main_service_node',
        name='main_service_node',
        output='screen',
    )

    return LaunchDescription(
        launch_arguments
        + env_assignments
        + [
            mock_llm_node,
            mock_robot_node,
            main_service_node,
        ]
    )
