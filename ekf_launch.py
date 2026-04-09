from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config_file_path = '/home/robot/goof-an-odd-husky/ekf.yaml'

    madgwick_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter_node',
        output='screen',
        parameters=[config_file_path],
        remappings=[
        ]
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[config_file_path],
        remappings=[
            ('odometry/filtered', '/platform/odom/filtered')
        ]
    )

    return LaunchDescription([
        madgwick_node,
        ekf_node
    ])
