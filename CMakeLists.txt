cmake_minimum_required(VERSION 3.22)

set(PROJECT_NAME AR-Gesture-Interaction-System)
project(${PROJECT_NAME})

#set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/cmake-build-debug)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(include)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(AR-Gesture-Interaction-System
        src/main.cpp
        include/fsm/fsm.hpp
        include/interact/interact-ui.hpp
        src/interact/interact-ui.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})