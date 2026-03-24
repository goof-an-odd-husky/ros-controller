from setuptools import find_packages, setup

package_name = "goof_an_odd_husky_viz"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Radomyr Husiev",
    maintainer_email="h.radomyr@proton.me",
    description="A goof_an_odd_husky visualizer",
    license="MIT",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "visualizer_node = goof_an_odd_husky_viz.visualizer_node:main",
        ],
    },
)
