Clone `https://github.com/UCU-robotics-lab/turtlebot4_sim_ws`

Start: `./start.sh` or `./start.sh <path-to-turtlebot4_sim_ws>`

Then `docker exec -ti ros_dev bash`

After starting the container for the first time run `./init.sh` and `./build.sh` from `/workspace/turtlebot4_sim_ws`, then run `./init.sh` and `./build.sh` from `/workspace/got-an-odd-husky`

Relaunch the shell

Run the simulation of your choice (e.g. `ros2 launch clearpath_gz simulation.launch.py setup_path:=/workspace/turtlebot4_sim_ws/husky_config slam:=true nav2:=true`)

Lastly, `ros2 run got_an_odd_husky controller_node`

Re-run `./build.sh` in `/workspace/got-an-odd-husky` when changing the program

In the end `docker stop ros_dev` and `docker rm ros_dev`
