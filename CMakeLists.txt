cmake_minimum_required(VERSION 3.22)
project(fm_smoke_cpp)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Add the executable for your project
add_executable(fm_smoke_cpp
        main.cpp
        smoke_merged_detect_malf.cpp
)

# Link OpenCV libraries to the executable
target_link_libraries(fm_smoke_cpp ${OpenCV_LIBS})

# Optionally, specify the directories where OpenCV headers are located
# This is typically not needed because find_package handles it
# target_include_directories(fm_smoke_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})

# Include directories required for your project, if any
target_include_directories(fm_smoke_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
