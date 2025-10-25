"""
Pickee Main 개발자를 위한 메인 서비스 런치 파일

Pickee Main에서 상품 검색 기능을 검증할 수 있도록 Main Service와 Mock LLM을 함께 실행합니다.
"""
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
)
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    """LaunchDescription을 생성한다."""
    api_host = LaunchConfiguration('api_host')
    api_port = LaunchConfiguration('api_port')
    llm_base_url = LaunchConfiguration('llm_base_url')
    db_url = LaunchConfiguration('db_url')
    gui_enabled = LaunchConfiguration('gui_enabled')

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
            description='LLM 서비스 URL',
        ),
        DeclareLaunchArgument(
            'db_url',
            default_value='mysql+pymysql://shopee:shopee@localhost:3306/shopee',
            description='Main Service가 사용할 데이터베이스 URL',
        ),
        DeclareLaunchArgument(
            'gui_enabled',
            default_value='true',
            description='대시보드 GUI 활성화 여부',
        ),
    ]

    env_assignments = [
        SetEnvironmentVariable('SHOPEE_API_HOST', api_host),
        SetEnvironmentVariable('SHOPEE_API_PORT', api_port),
        SetEnvironmentVariable('SHOPEE_LLM_BASE_URL', llm_base_url),
        SetEnvironmentVariable('SHOPEE_DB_URL', db_url),
        SetEnvironmentVariable('SHOPEE_GUI_ENABLED', gui_enabled),
        SetEnvironmentVariable('SHOPEE_LOG_LEVEL', 'INFO'),
        SetEnvironmentVariable('PYTHONUNBUFFERED', '1'),
    ]

    mock_llm_node = Node(
        package='main_service',
        executable='mock_llm_server',
        name='mock_llm_server',
        output='screen',
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
            main_service_node,
        ]
    )
