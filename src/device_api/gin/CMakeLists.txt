if(NCCL_OS_WINDOWS)
    return()
endif()

if(NOT MPI_FOUND)
    return()
endif()

# The GIN device API requirements these benchmarks fill in are only available
# from NCCL 2.30.7 onwards, so anything older is left out of the build the same
# way a missing MPI leaves it out. Keep this in step with src/Makefile. An
# undetermined version is not treated as too old, so a missing nccl.h fails at
# compile time like every other test rather than quietly skipping.
set(DEVICE_API_GIN_MIN_NCCL_VERSION 23007)
set(DEVICE_API_GIN_NCCL_HEADER "${NCCL_INCLUDE_DIR}/nccl.h")
file(STRINGS "${DEVICE_API_GIN_NCCL_HEADER}" device_api_gin_nccl_version_line
     REGEX "^#define[ \t]+NCCL_VERSION_CODE[ \t]+[0-9]+")
string(REGEX MATCH "[0-9]+" DEVICE_API_GIN_NCCL_VERSION "${device_api_gin_nccl_version_line}")

set(DEVICE_API_GIN_NCCL_TOO_OLD FALSE)
if(DEVICE_API_GIN_NCCL_VERSION AND
   DEVICE_API_GIN_NCCL_VERSION LESS DEVICE_API_GIN_MIN_NCCL_VERSION)
    set(DEVICE_API_GIN_NCCL_TOO_OLD TRUE)
endif()

if(DEVICE_API_GIN_NCCL_TOO_OLD)
    message(STATUS "Skipping GIN device API performance tests: NCCL version code "
                   "${DEVICE_API_GIN_NCCL_VERSION} is older than ${DEVICE_API_GIN_MIN_NCCL_VERSION}")
    return()
endif()

function(gin_perf_target_includes target_name)
    target_include_directories(${target_name}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
    )
endfunction()

function(add_gin_perf_category category_dir)
    set(category_path ${CMAKE_CURRENT_SOURCE_DIR}/${category_dir})
    if(NOT EXISTS ${category_path})
        return()
    endif()

    file(GLOB runner_sources CONFIGURE_DEPENDS ${category_path}/*_main.cu)
    file(GLOB test_sources CONFIGURE_DEPENDS ${category_path}/*.cu)
    if(runner_sources)
        list(REMOVE_ITEM test_sources ${runner_sources})
    endif()
    if(NOT test_sources OR NOT runner_sources)
        return()
    endif()

    set(tests_lib ginPerf${category_dir})
    add_library(${tests_lib} STATIC ${test_sources})
    gin_perf_target_includes(${tests_lib})
    target_link_libraries(${tests_lib}
        PRIVATE
        nccl_tests_options
        nccl
        MPI::MPI_CXX
        CUDA::cudart
    )

    foreach(runner_file ${runner_sources})
        get_filename_component(runner_name ${runner_file} NAME_WE)
        string(REGEX REPLACE "_main$" "" test_name ${runner_name})
        set(target_name ${test_name}_perf)

        add_executable(${target_name}
            ${runner_file}
            ${CMAKE_CURRENT_SOURCE_DIR}/utils/common.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/utils/args.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/utils/gin_context.cc
        )

        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/device_api/gin/${category_dir}
        )

        gin_perf_target_includes(${target_name})

        target_link_libraries(${target_name}
            PRIVATE
            ${tests_lib}
            nccl_tests_options
            nccl
            MPI::MPI_CXX
            CUDA::cudart
            m
            stdc++
        )
    endforeach()
endfunction()

add_gin_perf_category(latency)
add_gin_perf_category(throughput)
