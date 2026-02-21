cmake_minimum_required(VERSION 3.15)
project(FlightPath VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main.cpp
    src/VideoProcessor.cpp
    src/ObjectDetector.cpp
    src/PathPlanner.cpp
    src/Visualizer.cpp
)

# Header files
set(HEADERS
    src/VideoProcessor.h
    src/ObjectDetector.h
    src/PathPlanner.h
    src/Visualizer.h
    src/Config.h
    src/utility/DroppingSafeQueue.h
)

# Create executable
add_executable(FlightPath ${SOURCES} ${HEADERS})

# Link OpenCV libraries
target_link_libraries(FlightPath ${OpenCV_LIBS})

# Set output directory
set_target_properties(FlightPath PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Copy models directory to build output (if it exists)
if(EXISTS "${CMAKE_SOURCE_DIR}/models")
    file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR}/bin)
endif()

# Installation rules
install(TARGETS FlightPath DESTINATION bin)
