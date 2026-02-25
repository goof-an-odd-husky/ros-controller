Clone `https://github.com/UCU-robotics-lab/turtlebot4_sim_ws` | todo: change to husky-sim

Start: `./start.sh` or `./start.sh <path-to-husky-sim>`

Then `docker exec -ti husky_dev bash`

After starting the container for the first time run `./init.sh` and `./build.sh` from `/workspace/husky-sim`, then run `./init.sh` and `./build.sh` from `/workspace/goof-an-odd-husky`

Relaunch the shell

Run the simulation of your choice (e.g. `ros2 launch custom_simulation.launch.py setup_path:=/workspace/husky-sim/husky_config slam:=true nav2:=true world:=sonoma z:=35`)

Lastly, `ros2 run goof_an_odd_husky controller_node`

Re-run `./build.sh` in `/workspace/goof-an-odd-husky` when changing the program

In the end `docker stop husky_dev` and `docker rm husky_dev`

# Custom world

Under sdf/world/ add:

```sdf
    <plugin name="gz::sim::systems::Imu" filename="libgz-sim-imu-system.so"/>
    <plugin name="gz::sim::systems::NavSat" filename="libgz-sim-navsat-system.so"/>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>-22.986687</latitude_deg>
      <longitude_deg>-43.202501</longitude_deg>
      <elevation>35</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>
```
