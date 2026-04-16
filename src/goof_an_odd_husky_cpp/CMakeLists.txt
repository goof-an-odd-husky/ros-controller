cmake_minimum_required(VERSION 3.10)
project(goof_an_odd_husky_cpp)

find_package(ament_cmake REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)

pybind11_add_module(goof_costs
    src/bindings.cpp
)

target_link_libraries(goof_costs PRIVATE Ceres::ceres)
target_include_directories(goof_costs PRIVATE include)

install(TARGETS goof_costs
  LIBRARY DESTINATION lib/python3.12/site-packages
)

ament_package()
