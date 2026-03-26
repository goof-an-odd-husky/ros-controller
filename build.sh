#!/bin/bash
set -e

cd ros2_ws

export PIP_BREAK_SYSTEM_PACKAGES=1

rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install

source install/setup.bash
