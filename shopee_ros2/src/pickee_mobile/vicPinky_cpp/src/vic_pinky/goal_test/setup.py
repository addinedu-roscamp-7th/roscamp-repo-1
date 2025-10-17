from setuptools import find_packages, setup

package_name = 'goal_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lim',
    maintainer_email='lim@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'goal_test_client = goal_test.goal_test_client:main',
            'goal_send_client = goal_test.goal_send_client:main',
            'get_pose = goal_test.get_pose:main',
            'get_send_goal = goal_test.get_send_goal:main',
        ],
    },
)
