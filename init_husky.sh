#!/bin/bash

set -e

mkdir -p ros2_ws/src
ln -sf $(pwd)/src ros2_ws/src

if [ -f ros2_ws/install/setup.bash ]; then
    source ros2_ws/install/setup.bash
fi

grep -qxF "source $PWD/ros2_ws/install/setup.bash" ~/.bashrc || \
    echo "source $PWD/ros2_ws/install/setup.bash" >> ~/.bashrc

echo "yaml file://$(pwd)/rosdep_custom.yaml" | sudo tee /etc/ros/rosdep/sources.list.d/99-custom.list
rosdep update
