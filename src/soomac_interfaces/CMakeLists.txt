cmake_minimum_required(VERSION 3.8)
project(soomac_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/MissionEvent.msg"
  "msg/MissionResult.msg"
  "srv/DetectTarget.srv"
  "srv/PickPlace.srv"
  "srv/MoveScout.srv"
  "srv/ExecutePickPlace.srv"
  "srv/MoveToTicks.srv"
  "srv/DeliverMike.srv"
  "srv/MikeGripper.srv"
  "srv/StartMission2.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
