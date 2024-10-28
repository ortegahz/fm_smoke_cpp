cmake_minimum_required(VERSION 3.22)
project(fm_smoke_cpp)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Add the executable for your project
add_executable(fm_smoke_cpp main.cpp)

# Link OpenCV libraries to the executable
target_link_libraries(fm_smoke_cpp ${OpenCV_LIBS})

# Optionally, you can include OpenCV headers directory explicitly
# target_include_directories(fm_smoke_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
