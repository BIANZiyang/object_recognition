cmake_minimum_required(VERSION 2.8.3)
project(object_recognition)

## Find catkin macros and libraries

find_package(catkin REQUIRED COMPONENTS
    sensor_msgs
    roscpp
    std_msgs
    pcl_ros
    geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES Object_Recognition
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(object_detection src/object_detection.cpp)
target_link_libraries(object_detection ${catkin_LIBRARIES})
#add_executable(object_recognition src/object_recoginition.cpp)
#target_link_libraries(object_recognition ${catkin_LIBRARIES})
