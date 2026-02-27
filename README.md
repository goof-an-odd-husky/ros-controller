Clone `git clone https://github.com/goof-an-odd-husky/simulation.git husky-sim`

Start: `./start.sh` or `./start.sh <path-to-husky-sim>`

Then `docker exec -ti husky_dev bash`

After starting the container for the first time run `./init.sh` and `./build.sh` from `/workspace/husky-sim`, then run `./init.sh` and `./build.sh` from `/workspace/goof-an-odd-husky`

Relaunch the shell

Run the simulation of your choice (e.g. `ros2 launch custom_simulation.launch.py setup_path:=/workspace/husky-sim/husky_config slam:=true nav2:=true world:=sonoma z:=35`)

Lastly, `ros2 run goof_an_odd_husky controller_node`

Re-run `./build.sh` in `/workspace/goof-an-odd-husky` when changing the program

In the end `docker stop husky_dev` and `docker rm husky_dev`
