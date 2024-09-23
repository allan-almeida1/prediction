cmake_minimum_required(VERSION 3.26)
project(prediction)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Define custom messages
add_message_files(
  FILES
  Limits.msg
  Path.msg
  Resolution.msg
  States.msg
)

# Generate the messages
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs image_transport roscpp rospy sensor_msgs std_msgs message_runtime
)

include_directories(
  ${PROJECT_SOURCE_DIR}/src/processing/include
  ${PROJECT_SOURCE_DIR}/src/state_estimation/include
  ${PROJECT_SOURCE_DIR}/include/prediction
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${GTEST_INCLUDE_DIRS}
)

# Define processing executable
set(SOURCES_processing
  src/processing/Processing.cpp
  src/processing/ProcessingNode.cpp
  src/util/exec_time_cpp/ExecTime.cpp
  src/processing/main.cpp
)

set(HEADERS_processing
  src/processing/include/Processing.hpp
  include/prediction/ExecTime.hpp
  src/processing/include/ProcessingNode.hpp
)

add_executable(processing ${SOURCES_processing} ${HEADERS_processing})
target_link_libraries(processing
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

# Ensure processing depends on message generation
add_dependencies(processing ${PROJECT_NAME}_generate_messages_cpp)

# Define state estimation executable
set(SOURCES_state_estimation
  src/state_estimation/StateEstimation.cpp
  src/state_estimation/StateEstimationNode.cpp
  src/state_estimation/main.cpp
)

set(HEADERS_state_estimation
  src/state_estimation/include/StateEstimation.hpp
  src/state_estimation/include/StateEstimationNode.hpp
)

add_executable(state_estimation ${SOURCES_state_estimation} ${HEADERS_state_estimation})
target_link_libraries(state_estimation
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

# Ensure state estimation depends on message generation
add_dependencies(state_estimation ${PROJECT_NAME}_generate_messages_cpp)

# Define unit tests
find_package(GTest REQUIRED)

set(SOURCES_tests
  tests/processing_test.cpp
  tests/state_estimation_test.cpp
  tests/main.cpp
  src/processing/Processing.cpp
  src/state_estimation/StateEstimation.cpp
)

set(HEADERS_tests
  src/processing/include/Processing.hpp
  src/state_estimation/include/StateEstimation.hpp
)

add_executable(unit_tests ${SOURCES_tests} ${HEADERS_tests})
target_link_libraries(unit_tests
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${GTEST_LIBRARY}
  ${GTEST_MAIN_LIBRARY}
  pthread
)
add_dependencies(unit_tests ${PROJECT_NAME}_generate_messages_cpp)
