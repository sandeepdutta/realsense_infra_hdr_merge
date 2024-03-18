cmake_minimum_required(VERSION 3.8)
project(realsense_infra_hdr_merge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(realsense2_camera_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

############
# LIBRARIES #
#############
add_library(realsense_infra_hdr_merge
  src/realsense_infra_hdr_merge.cpp
)
target_compile_definitions(realsense_infra_hdr_merge
  PRIVATE "COMPOSITION_BUILDING_DLL")

ament_target_dependencies(realsense_infra_hdr_merge
  rclcpp
  rclcpp_components
  sensor_msgs
  message_filters
  realsense2_camera_msgs
  OpenCV
  cv_bridge
)
target_include_directories(realsense_infra_hdr_merge PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

rclcpp_components_register_nodes(realsense_infra_hdr_merge "rshdr::realsenseInfraHdrMerge")
set(node_plugins "${node_plugins}rshdr::realsenseInfraHdrMerge;$<TARGET_FILE:realsense_infra_hdr_merge>\n")


############
# BINARIES #
############
add_executable(realsense_infra_hdr_merge_node
  src/realsense_infra_hdr_merge_main.cpp
)
target_link_libraries(realsense_infra_hdr_merge_node realsense_infra_hdr_merge)

###########
# INSTALL #
###########

# includes
install(
  DIRECTORY include/
  DESTINATION include
)

# libs
install(
  TARGETS realsense_infra_hdr_merge
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# nodes
install(TARGETS realsense_infra_hdr_merge_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


##########
# EXPORT #
##########
ament_export_include_directories(
  include
)
ament_export_libraries(
  realsense_infra_hdr_merge
)
ament_export_targets(
  ${PROJECT_NAME}Targets
)
ament_export_dependencies(
  rclcpp
  sensor_msgs
)
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME})
ament_package()
