cmake_minimum_required(VERSION 3.8)
project(vo_baseline)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Export compile commands for IDE IntelliSense (removes red lines in VS Code)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++17 required for std::filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Find ROS2 and external dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

# 2. Define the C++ executables
add_executable(thermal_preprocessor_node src/thermal_preprocessor_node.cpp)
add_executable(orb_tracker_node src/orb_tracker_node.cpp)
add_executable(orb_vo_node src/orb_vo_node.cpp)
add_executable(image_publisher_node src/image_publisher_node.cpp)

# 3. Link ROS2 dependencies to the executables
ament_target_dependencies(thermal_preprocessor_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)

ament_target_dependencies(orb_tracker_node
  rclcpp
  sensor_msgs
  cv_bridge
)

ament_target_dependencies(orb_vo_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  cv_bridge
)

ament_target_dependencies(image_publisher_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
)

# 4. Link OpenCV specifically
target_link_libraries(thermal_preprocessor_node ${OpenCV_LIBRARIES})
target_link_libraries(orb_tracker_node ${OpenCV_LIBRARIES})
target_link_libraries(orb_vo_node ${OpenCV_LIBRARIES})
target_link_libraries(image_publisher_node ${OpenCV_LIBRARIES})

# 5. Define installation rules
install(TARGETS
  thermal_preprocessor_node
  orb_tracker_node
  orb_vo_node
  image_publisher_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()