#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# Permit the experimental NCCL device code used by perf tests.
add_compile_definitions(NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1)

# Tests that need a specific NCCL are left out of the build below the version
# they require, so an older NCCL still builds everything else. Keep these in
# step with the equivalents in Makefile. An undetermined version is not treated
# as too old, so a missing nccl.h fails at compile time like every other test
# rather than quietly skipping.
file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" nccl_version_code_line
     REGEX "^#define[ \t]+NCCL_VERSION_CODE[ \t]+[0-9]+")
string(REGEX MATCH "[0-9]+" NCCL_HEADER_VERSION "${nccl_version_code_line}")

# Sets out_var to TRUE when the headers are older than the given version code,
# and to FALSE when they are new enough or the version could not be determined.
function(nccl_older_than out_var min_version)
  set(${out_var} FALSE PARENT_SCOPE)
  if(NCCL_HEADER_VERSION AND NCCL_HEADER_VERSION LESS min_version)
    set(${out_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

# Communicator operations need ncclCommGrow/ncclCommGetUniqueId (2.29) and MPI.
set(COMM_OPS_MIN_NCCL_VERSION 22900)
nccl_older_than(COMM_OPS_NCCL_TOO_OLD ${COMM_OPS_MIN_NCCL_VERSION})

set(PERF_COMMON_SOURCES
  common.cu
  util.cu
  timer.cc
)

set(COLLS
  all_reduce
  all_gather
  broadcast
  reduce_scatter
  reduce
  alltoall
  alltoallv
  gather
  scatter
  sendrecv
  hypercube
)

foreach(COLL IN LISTS COLLS)
  add_executable(${COLL}_perf ${COLL}.cu ${PERF_COMMON_SOURCES})
  add_dependencies(${COLL}_perf generate_git_version)
  set_target_properties(${COLL}_perf PROPERTIES
    ENABLE_EXPORTS TRUE
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  target_include_directories(${COLL}_perf
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/os
      ${CMAKE_SOURCE_DIR}/verifiable
  )

  target_link_libraries(${COLL}_perf
    PRIVATE
      nccl_tests_options
      nccl
      CUDA::cudart
      CUDA::cuda_driver
      verifiable
      nccl_test_os
  )

  if(MPI_FOUND)
    target_compile_definitions(${COLL}_perf PRIVATE MPI_SUPPORT)
    target_include_directories(${COLL}_perf PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(${COLL}_perf PRIVATE MPI::MPI_CXX)
  endif()
endforeach()

if(MPI_FOUND)
  add_subdirectory(device_api/gin)
endif()

if(MPI_FOUND)
  if(COMM_OPS_NCCL_TOO_OLD)
    message(STATUS "Skipping communicator operations test: NCCL version code "
                   "${NCCL_HEADER_VERSION} is older than ${COMM_OPS_MIN_NCCL_VERSION}")
  else()
    add_executable(comm_ops_perf comm_ops.cu)
    set_target_properties(comm_ops_perf PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    target_include_directories(comm_ops_perf
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/os
        ${MPI_CXX_INCLUDE_DIRS}
    )

    target_link_libraries(comm_ops_perf
      PRIVATE
        nccl_tests_options
        nccl
        CUDA::cudart
        CUDA::cuda_driver
        nccl_test_os
        MPI::MPI_CXX
    )

    target_compile_definitions(comm_ops_perf PRIVATE MPI_SUPPORT)
  endif()
endif()
