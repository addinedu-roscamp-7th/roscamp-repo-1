from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'pickee_main'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wonho',
    maintainer_email='wonho9188@gmail.com',
    description='Main controller for Pickee robot',
    license='Apache License 2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'main_controller = pickee_main.main_controller:main',
            'mock_mobile_node = test.mock_nodes.mock_mobile_node:main',
            'mock_arm_node = test.mock_nodes.mock_arm_node:main',
            'mock_vision_node = test.mock_nodes.mock_vision_node:main',
            'integration_test_client = test.integration.integration_test_client:main',
            'dashboard = pickee_main.dashboard.launcher:main',
            'mock_shopee_main = test.mock_nodes.mock_shopee_main:main',
        ],
    },
)
