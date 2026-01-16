from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # 1. Find the Clearpath package
    pkg_clearpath_gz = FindPackageShare('clearpath_gz')

    # 2. Define Arguments
    # We remove the 'choices' list here so it accepts ANY string
    arg_world = DeclareLaunchArgument(
        'world', 
        default_value='warehouse',
        description='Gazebo World name (filename without .sdf)'
    )
    
    # Standard Clearpath arguments
    arg_rviz = DeclareLaunchArgument('rviz', default_value='false',
                          choices=['true', 'false'], description='Start rviz.')
    arg_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true',
                          choices=['true', 'false'], description='use_sim_time')
    arg_setup_path = DeclareLaunchArgument('setup_path',
                          default_value=[EnvironmentVariable('HOME'), '/clearpath/'],
                          description='Clearpath setup path')

    # Robot Pose arguments
    pose_args = [
        DeclareLaunchArgument('x', default_value='0.0', description='x pose'),
        DeclareLaunchArgument('y', default_value='0.0', description='y pose'),
        DeclareLaunchArgument('z', default_value='0.3', description='z pose'),
        DeclareLaunchArgument('yaw', default_value='0.0', description='yaw pose')
    ]

    # 3. Define Paths to included launch files
    gz_sim_launch = PathJoinSubstitution(
        [pkg_clearpath_gz, 'launch', 'gz_sim.launch.py'])
    
    robot_spawn_launch = PathJoinSubstitution(
        [pkg_clearpath_gz, 'launch', 'robot_spawn.launch.py'])

    # 4. OPTIONAL: Add the folder containing your world to the Gazebo Resource Path
    # Change '/path/to/your/custom/worlds' to the actual folder containing your .sdf file
    # If you don't do this, you must copy your world to the clearpath_gz/worlds folder
    # or export the GZ_SIM_RESOURCE_PATH in your terminal.
    
    # custom_world_path = '/home/user/my_custom_worlds'
    # env_var = SetEnvironmentVariable(
    #     name='GZ_SIM_RESOURCE_PATH',
    #     value=[EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value=''), ':', custom_world_path]
    # )

    # 5. Include the Gazebo Simulation
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([gz_sim_launch]),
        launch_arguments=[
            ('world', LaunchConfiguration('world'))
        ]
    )

    # 6. Include the Robot Spawner
    robot_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([robot_spawn_launch]),
        launch_arguments=[
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ('setup_path', LaunchConfiguration('setup_path')),
            ('world', LaunchConfiguration('world')),
            ('rviz', LaunchConfiguration('rviz')),
            ('x', LaunchConfiguration('x')),
            ('y', LaunchConfiguration('y')),
            ('z', LaunchConfiguration('z')),
            ('yaw', LaunchConfiguration('yaw'))
        ]
    )

    # 7. Build Launch Description
    ld = LaunchDescription()
    ld.add_action(arg_world)
    ld.add_action(arg_rviz)
    ld.add_action(arg_sim_time)
    ld.add_action(arg_setup_path)
    for arg in pose_args:
        ld.add_action(arg)
        
    # ld.add_action(env_var) # Uncomment if using the path modification above
    ld.add_action(gz_sim)
    ld.add_action(robot_spawn)

    return ld
