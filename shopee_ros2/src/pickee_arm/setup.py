from setuptools import find_packages, setup

package_name = 'pickee_arm'

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
    maintainer='addinedu',
    maintainer_email='tst50087@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pickee_arm_node = pickee_arm.pickee_arm_node:main',
            'pickee_arm_node_for_basket = pickee_arm.pickee_arm_node_for_basket:main',
            #'test_1110_new_arm_node = pickee_arm.1110_new_arm_node_test:main',
        ],
    },
)
