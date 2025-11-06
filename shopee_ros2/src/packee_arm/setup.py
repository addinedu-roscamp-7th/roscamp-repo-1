from setuptools import setup

package_name = 'packee_arm'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],  # ✅ src/ 내부 구조에 맞춤
    package_dir={package_name: 'src'},  # ✅ src를 모듈 경로로 지정
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetcobot',
    maintainer_email='you@example.com',
    description='Packee myCobot arm control node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pymycobot_right_node = packee_arm.pymycobot_right:main',
            'pymycobot_left_node = packee_arm.pymycobot_left:main',
            'pymycobot_dual_node = packee_arm.pymycobot_dual:main',
            'packee_arm_left_node = packee_arm.packee_arm_left:main',
            'packee_arm_right_node = packee_arm.packee_arm_right:main',
        ],
    },
)
