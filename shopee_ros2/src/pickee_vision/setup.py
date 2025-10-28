from setuptools import find_packages, setup

package_name = 'pickee_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
                        ('share/' + package_name, ['package.xml', 'pickee_vision/20251015_1.pt', 'pickee_vision/cart_best_.pth', 
                                                   'pickee_vision/20251024_v8_last.pt', 'pickee_vision/20251024_v11_ver1.pt']),
                    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='addinedu',
    maintainer_email='dltmdgks062358@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'obstacle_detector = pickee_vision.obstacle_detector_node:main',
            'pickee_vision = pickee_vision.pickee_vision_node:main',
            'staff_tracker = pickee_vision.staff_tracker_node:main',
            'picvi_test = pickee_vision.picvi_test:main',
        ],
    },
)
