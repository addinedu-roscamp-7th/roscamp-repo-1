from setuptools import find_packages, setup

package_name = 'packee_vision'

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
    maintainer_email='leehansu0201@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'check_cart_presence = packee_vision.check_cart_presence:main',
            'detect_products_in_cart = packee_vision.detect_products_in_cart:main',
            'verify_packing_complete = packee_vision.verify_packing_complete:main'
        ],
    },
)
