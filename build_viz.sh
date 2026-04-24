#!/bin/bash
set -e

cd ros2_ws

export PIP_BREAK_SYSTEM_PACKAGES=1

rosdep install --from-paths /workspace/goof-an-odd-husky/src/goof_an_odd_husky_* --ignore-src -r -y

colcon build --symlink-install \
    --packages-select \
        goof_an_odd_husky_common \
        goof_an_odd_husky_msgs \
        goof_an_odd_husky_viz \
    --cmake-args -DROSIDL_TYPESUPPORT_FASTRTPS_CPP=OFF -DROSIDL_TYPESUPPORT_FASTRTPS_C=OFF

source install/setup.bash
