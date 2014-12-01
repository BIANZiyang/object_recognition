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
# INCLUDE_DIRS include
# LIBRARIES Object_Recognition
# CATKIN_DEPENDS other_catkin_pkg
# DEPENDS system_lib
)
include_directories(
${catkin_INCLUDE_DIRS}
)
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
# message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
#list(APPEND catkin_LIBRARIES /opt/ros/hydro/lib/libopencv_core.so /opt/ros/hydro/lib/libopencv_imgproc.so)
add_executable(object_detection src/object_detection.cpp)
target_link_libraries(object_detection ${catkin_LIBRARIES} /opt/ros/hydro/lib/libopencv_core.so /opt/ros/hydro/lib/libopencv_imgproc.so /opt/ros/hydro/lib/libopencv_highgui.so /opt/ros/hydro/lib/libimage_transport.so /opt/ros/hydro/lib/libcv_bridge.so)
add_executable(object_recognition src/object_recognition.cpp)
target_link_libraries(object_recognition ${catkin_LIBRARIES} /opt/ros/hydro/lib/libopencv_ml.so /opt/ros/hydro/lib/libopencv_core.so /opt/ros/hydro/lib/libopencv_imgproc.so /opt/ros/hydro/lib/libopencv_highgui.so /opt/ros/hydro/lib/libimage_transport.so /opt/ros/hydro/lib/libcv_bridge.so)
add_executable(sample_image_creater src/sample_image_creater.cpp)
target_link_libraries(sample_image_creater ${catkin_LIBRARIES} /opt/ros/hydro/lib/libopencv_core.so /opt/ros/hydro/lib/libopencv_imgproc.so /opt/ros/hydro/lib/libopencv_highgui.so /opt/ros/hydro/lib/libimage_transport.so /opt/ros/hydro/lib/libcv_bridge.so)


