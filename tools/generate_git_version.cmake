#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

execute_process(
  COMMAND git describe --dirty --always --exclude *
  WORKING_DIRECTORY "${SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE GIT_RESULT
)
if(GIT_RESULT)
  set(GIT_VERSION unknown)
endif()

set(CONTENT "#define NCCL_TESTS_GIT_VERSION \"${GIT_VERSION}\"\n")
if(EXISTS "${OUTPUT_FILE}")
  file(READ "${OUTPUT_FILE}" OLD_CONTENT)
endif()
if(NOT CONTENT STREQUAL OLD_CONTENT)
  get_filename_component(OUTPUT_DIR "${OUTPUT_FILE}" DIRECTORY)
  file(MAKE_DIRECTORY "${OUTPUT_DIR}")
  file(WRITE "${OUTPUT_FILE}" "${CONTENT}")
endif()
