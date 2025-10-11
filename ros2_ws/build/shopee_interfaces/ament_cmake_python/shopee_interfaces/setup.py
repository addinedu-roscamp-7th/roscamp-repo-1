from setuptools import find_packages
from setuptools import setup

setup(
    name='shopee_interfaces',
    version='0.0.1',
    packages=find_packages(
        include=('shopee_interfaces', 'shopee_interfaces.*')),
)
