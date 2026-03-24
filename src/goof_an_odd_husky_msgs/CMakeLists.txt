cmake_minimum_required(VERSION 3.8)
project(goof_an_odd_husky_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObstacleCircle.msg"
  "msg/ObstacleLine.msg"
  "msg/ObstacleArray.msg"
)

ament_package()
