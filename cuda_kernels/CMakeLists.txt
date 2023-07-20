cmake_minimum_required(VERSION 3.11)

project(cuda_kernels LANGUAGES CUDA CXX)

# Cuda Code
file(GLOB_RECURSE SRC
    "src/*.cu"
    "src/*.cuh"
)
add_library(cuda_kernels STATIC ${SRC})

# Configure
find_package(CUDA REQUIRED)
set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda_kernels PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link libraries
target_link_libraries(
  cuda_kernels
  ${CUDA_LIBRARIES}
  )

# Post Build
add_custom_command(TARGET cuda_kernels POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cuda_kernels> ${CMAKE_SOURCE_DIR})
