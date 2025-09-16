# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.12)

# 项目信息
set(PROJECT_NAME "tts_g2p")
project(${PROJECT_NAME} C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

set(root_dir ${PROJECT_SOURCE_DIR})
message(STATUS "Source folder - ${root_dir}")


########################################################
# Properties
########################################################
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# enable code analysis by generating compile_commands.json (sonarlint, intellisense...)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# disable cxx11::basic_string since centos7 does not support this 
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# Determine the platform.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  set(OS_MACOSX 1)
  set(OS_POSIX 1)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(OS_LINUX 1)
  set(OS_POSIX 1)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(OS_WINDOWS 1)
endif()

# Determine the project architecture.
EXECUTE_PROCESS( COMMAND uname -p COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )

########################################################
# Building options
########################################################
if (NOT CMAKE_BUILD_TYPE)
   set (CMAKE_BUILD_TYPE "Release")
   message (
      STATUS "No CMAKE_BUILD_TYPE selected, defaulting to ${CMAKE_BUILD_TYPE}"
   )
endif ()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${root_dir}/include)
include_directories(${root_dir}/include/common)

find_library(TN_LIB NAMES tn HINTS ${root_dir})
# find_library(JSON_LIB NAMES jsoncpp HINTS ${root_dir}/jsoncpp/lib)

message(STATUS "detected tn lib: ${TN_LIB}")

execute_process(COMMAND patchelf --replace-needed libjsoncpp.so.25 ${root_dir}/jsoncpp/lib/libjsoncpp.so.25 ${TN_LIB})
execute_process(COMMAND patchelf --replace-needed libonnxruntime.so.1.10.0 ${root_dir}/onnxruntime/lib/libonnxruntime.so.1.10.0 ${TN_LIB})
execute_process(COMMAND python3 ${root_dir}/get_includes.py OUTPUT_VARIABLE pybind_inc)
execute_process(COMMAND python3 ${root_dir}/get_python.py OUTPUT_VARIABLE python_inc)
execute_process(COMMAND python3-config --extension-suffix OUTPUT_VARIABLE python_suffix)
string(STRIP ${python_suffix} python_suffix)
message(STATUS "python suffix: ${python_suffix}")
include_directories(${pybind_inc})
include_directories(${python_inc})

# json
include_directories(${root_dir}/jsoncpp/include)
set(JSON_LIB ${root_dir}/jsoncpp/lib/libjsoncpp.so.25)

message(STATUS "detected json lib: ${JSON_LIB}")

# onnx
include_directories(${root_dir}/onnxruntime/include)
set(ONNX_LIB ${root_dir}/onnxruntime/lib/libonnxruntime.so.1.10.0)

message(STATUS "detected onnx lib: ${ONNX_LIB}")
########################################################
# Main part
########################################################

# tn python library
add_library(redtn SHARED tn.cpp)
set_target_properties(redtn PROPERTIES SUFFIX "${python_suffix}")
set_target_properties(redtn PROPERTIES PREFIX "")

target_link_libraries(redtn PRIVATE
  ${TN_LIB}
  ${JSON_LIB}
  ${ONNX_LIB}
)

