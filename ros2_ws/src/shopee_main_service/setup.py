from setuptools import setup

package_name = "shopee_main_service"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "pydantic-settings",
        "python-dotenv",
        "SQLAlchemy",
        "PyMySQL",
        "passlib",
        "bcrypt",
        "httpx",
    ],
    zip_safe=True,
    author="Shopee Robotics",
    author_email="jinhyuk2me@example.com",
    maintainer="jinhyuk2me",
    maintainer_email="jinhyuk2me@example.com",
    description="Main Service ROS 2 entry point handling TCP API and robot orchestration.",
    license="Apache License 2.0",
    tests_require=["pytest", "pytest-asyncio"],
    entry_points={
        "console_scripts": [
            "main_service_node = shopee_main_service.main_service_node:main",
        ],
    },
)
