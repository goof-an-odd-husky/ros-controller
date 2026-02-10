#!/bin/bash

sim_path="${1:-../turtlebot4_sim_ws}"

docker compose -f ${sim_path}/docker-compose.yaml run -v $(pwd):/workspace/goof-an-odd-husky -d --name ros_dev ros_dev
