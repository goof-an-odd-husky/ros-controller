All work after the submission deadline (2026-04-29 09:00) is done in the branch `continuation`

# Launch

Either:
- Run the simulation of your choice
    - E.g. `ros2 launch custom_simulation.launch.py setup_path:=/workspace/husky-sim/husky_config slam:=false nav2:=true world:=sonoma origin:=-22.986687,-43.202501,35.0 spawn:=-22.98787540402462,-43.19982241436838,38.107444597408175`
- Use the real Husky

Start the goof-an-odd-husky:
- Controller node - controls Husky to move to coordinates set by visualizer
    - `ros2 run goof_an_odd_husky controller_node`
    - Preferably should be run on the vehicle itself (or at least over something faster than Wi-Fi)
- Visualizer node - shows the obstacles, the global path and local trajectory, created by controller
    - `ros2 run goof_an_odd_husky_viz visualizer_node`
    - Can be run on a separate machine on the network. Visualizer Container can be used
    - Needs a display (tested on XWayland)

Re-run `./build.sh` in `goof-an-odd-husky/` when changing packages (e.g. when updating dependencies)

# Setup

Use if you want a ros2 environment, with everything setup

## If you want a simulator

Clone `git clone https://github.com/goof-an-odd-husky/simulation.git husky-sim`

Follow the guide to use nvidia inside the container https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Real Husky

Run `. init_husky.sh && ./build.sh`

Run `ros2 launch ekf_launch.py`

## Start the container

### With simulator support

Start: `./start.sh <path-to-husky-sim>` or `./start.sh` (if cloned into `../husky-sim`)

Then `docker exec -ti husky_dev bash`

After starting the container for the first time run `./init.sh` from `/workspace/husky-sim`, then run `. init_docker.sh && ./build.sh` from `/workspace/goof-an-odd-husky`

To stop the container run `docker stop husky_dev` and `docker rm husky_dev`

### Visualizer only

Start: `docker compose up -d --build`

Then `docker compose exec -ti husky_viz /bin/bash`

After starting the container for the first time run `. init_docker.sh && ./build_viz.sh` from `/workspace/goof-an-odd-husky`

To stop the container run `docker compose down`
