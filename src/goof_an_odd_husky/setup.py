from setuptools import find_packages, setup

package_name = "goof_an_odd_husky"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "scipy",
        "pyceres",
        "matplotlib",
    ],
    zip_safe=True,
    maintainer="Radomyr Husiev",
    maintainer_email="h.radomyr@proton.me",
    description="A husky controller to avoid obstacles",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "controller_node = goof_an_odd_husky.controller_node:main",
            "lidar_visualize = goof_an_odd_husky.lidar_visualize:main",
        ],
    },
)
