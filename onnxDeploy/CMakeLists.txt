# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.26)

# Project
project(ort_model_test C CXX)

option(BUILD_x64 "make for x86_64" OFF)
option(BUILD_7020 "make for xilinx 7020" OFF)
option(BUILD_LICHEE "make for lichee pi" OFF)
option(BUILD_MPSOC "make for mpsoc" OFF)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ")

#* Build for x86
if (BUILD_x64)
  message("Make for x86_64...")
  set(ONNXRUNTIME_ROOTDIR "./ortbuild_x86/")
endif()

#* FOR CROSS COMPILING - 7020
if (BUILD_7020)
  message("Make for 7020...")
  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_SYSTEM_PROCESSOR arm)
  set(tools "/opt/gcc-arm-8.2-2018.08-x86_64-arm-linux-gnueabihf")
  set(TOOLCHAIN_PATH ${tools}/gcc-arm-linux-gnueabihf)
  set(CMAKE_C_COMPILER ${tools}/bin/arm-linux-gnueabihf-gcc)
  set(CMAKE_CXX_COMPILER ${tools}/bin/arm-linux-gnueabihf-g++)
  set(ONNXRUNTIME_ROOTDIR "./ortbuild_7020/")
endif()

#* FOR CROSS COMPILING - ROCKPI
if (BUILD_ROCKPI)
  message("Make for ROCKPI...")
  set(CMAKE_SYSTEM_NAME Linux)
  SET(CMAKE_SYSTEM_PROCESSOR aarch64)
  set(tools "/opt/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu")
  set(CMAKE_C_COMPILER ${tools}/bin/aarch64-none-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER ${tools}/bin/aarch64-none-linux-gnu-g++)
  set(ONNXRUNTIME_ROOTDIR "./ortbuild_aarch64/")
endif()

#* FOR CROSS COMPILING - MPSOC
if (BUILD_MPSOC)
  message("Make for MPSOC...")
  set(CMAKE_SYSTEM_NAME Linux)
  SET(CMAKE_SYSTEM_PROCESSOR aarch64)
  set(tools "/opt/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu")
  set(CMAKE_C_COMPILER ${tools}/bin/aarch64-none-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER ${tools}/bin/aarch64-none-linux-gnu-g++)
  set(ONNXRUNTIME_ROOTDIR "./ortbuild_mpsoc/")
endif()

#* FOR CROSS COMPILING - lichee
if (BUILD_LICHEE)
  message("Make for LicheePi...")
  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_SYSTEM_PROCESSOR riscv64)
  set(tools "/opt/host-tools/gcc/riscv64-linux-musl-x86_64")
  set(CMAKE_C_COMPILER ${tools}/bin/riscv64-unknown-linux-musl-gcc)
  set(CMAKE_CXX_COMPILER ${tools}/bin/riscv64-unknown-linux-musl-g++)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d")
  set(ONNXRUNTIME_ROOTDIR "./ortbuild_lichee/")
endif()

# The ORT package has a different include directory structure to a local install via cmake.
# We added the path for the pre-built package above. Add the path for a local install to support either usage.
# TODO: If we want to support additional EPs being loadable from a local install we also need to add EP specific
# directories under /include/onnxruntime/core/providers

# Headers
include_directories(
  "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"
  "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"
  )
# Static Libraries
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

message(STATUS "ONNXRuntime libraries: ${ONNXRUNTIME_ROOTDIR}")  
message(STATUS "ONNXRuntime include dirs: ${ONNXRUNTIME_ROOTDIR}")  
 

add_executable(mtest model-test.cpp)
# add_executable(neontest neon-test.cpp)
# Dynamic Libraries
target_link_libraries(mtest PRIVATE onnxruntime) #libonnxruntime.so