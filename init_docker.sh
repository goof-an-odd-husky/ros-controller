#!/bin/bash

set -e

mkdir -p ros2_ws/src
ln -sf $(pwd)/src ros2_ws/src

source /opt/ros/jazzy/setup.bash
if [ -f ros2_ws/install/setup.bash ]; then
    source ros2_ws/install/setup.bash
fi

grep -qxF 'source /opt/ros/jazzy/setup.bash' ~/.bashrc || \
    echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
grep -qxF 'source /workspace/goof-an-odd-husky/ros2_ws/install/setup.bash' ~/.bashrc || \
    echo 'source /workspace/goof-an-odd-husky/ros2_ws/install/setup.bash' >> ~/.bashrc

echo "yaml file://$(pwd)/rosdep_custom.yaml" | sudo tee /etc/ros/rosdep/sources.list.d/99-custom.list
rosdep update
apt update
