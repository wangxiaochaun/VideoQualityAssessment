cmake_minimum_required(VERSION 3.8)
project(hole_filling)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES

        demo.cpp
        hole_filling.h
        hole_filling.cpp
        A6.h
        A6.cpp
        utils.h
        utils.cpp sprite.cpp sprite.h)

add_executable(hole_filling ${SOURCE_FILES})
target_link_libraries(hole_filling ${OpenCV_LIBS})