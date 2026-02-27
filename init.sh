#!/bin/bash

source /workspace/husky-sim/ros2_ws/install/setup.bash

mkdir -p ros2_ws/src
ln -sf $(pwd)/src ros2_ws/src

if [ -f ros2_ws/install/setup.bash ]; then
    source ros2_ws/install/setup.bash
fi

grep -qxF 'source /workspace/husky-sim/ros2_ws/install/setup.bash' ~/.bashrc || \
    echo 'source /workspace/husky-sim/ros2_ws/install/setup.bash' >> ~/.bashrc
grep -qxF 'source /workspace/goof-an-odd-husky/ros2_ws/install/setup.bash' ~/.bashrc || \
    echo 'source /workspace/goof-an-odd-husky/ros2_ws/install/setup.bash' >> ~/.bashrc
