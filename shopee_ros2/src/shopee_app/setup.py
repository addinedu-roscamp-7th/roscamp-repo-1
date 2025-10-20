from setuptools import find_packages
from setuptools import setup


package_name = 'shopee_app'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=[package_name, f'{package_name}.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml', 'requirements.txt', 'README.md']),
        (f'share/{package_name}/launch', ['launch/app.launch.py']),
    ],
    install_requires=[
        'setuptools',
        'PyQt6',
        'watchdog',
        'PyYAML',
        'rclpy',
        'shopee_interfaces',
    ],
    zip_safe=False,
    maintainer='Shopee Robotics Team',
    maintainer_email='dev@shopee.app',
    description='Shopee 로봇 쇼핑 관리자 GUI의 ROS2 통합',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'shopee_app_gui = shopee_app.launcher:main',
        ],
    },
)
