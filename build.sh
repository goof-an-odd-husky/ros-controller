#!/bin/bash
set -e

cd ros2_ws

rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install

source install/setup.bash
