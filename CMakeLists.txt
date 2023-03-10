# CMake version check.
cmake_minimum_required(VERSION 3.16)

# Define the project name.
project(EVSVesuvio)

# Policy settings
if(POLICY CMP0022)
  cmake_policy(SET CMP0022 NEW)
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# Add the path to our custom 'find' modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/buildconfig/CMake")

# Bootstrap any dependencies
include(Bootstrap)

# Cross-platform setup
include(CommonSetup)