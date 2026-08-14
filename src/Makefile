#
# Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
include common.mk

# Permit the experimental NCCL device code used by perf tests.
CXXFLAGS += -DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1
NVCUFLAGS += -DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1

# Set to 1 to enable MPI support (multi-process/multi-node)
MPI ?= 0
# e.g. Set to _mpi when using MPI=1
NAME_SUFFIX ?=
# Set to 1 to create and use libverifiable.so to reduce binary size
DSO ?= 0

.PHONY: build clean

BUILDDIR ?= ../build
ifneq ($(NCCL_HOME), "")
NVCUFLAGS += -I$(NCCL_HOME)/include/
NVLDFLAGS += -L$(NCCL_HOME)/lib
endif

ifeq ($(MPI), 1)
MPI_INCLUDE ?= $(MPI_HOME)/include
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_INCLUDE)
NVLDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/lib64 -lmpi
endif
ifeq ($(MPI_IBM),1)
NVCUFLAGS += -DMPI_SUPPORT
NVLDFLAGS += -lmpi_ibm
endif
LIBRARIES += nccl
NVLDFLAGS += $(LIBRARIES:%=-l%)

DST_DIR := $(BUILDDIR)
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := all_reduce all_gather broadcast reduce_scatter reduce alltoall alltoallv scatter gather sendrecv hypercube
EXTRA_BIN_FILES := comm_ops
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf${NAME_SUFFIX}) $(EXTRA_BIN_FILES:%=${DST_DIR}/%_perf${NAME_SUFFIX})

TEST_OS_SRCDIR := ../os
TEST_OS_BUILDDIR := $(BUILDDIR)/os
TEST_OS_LIB := $(TEST_OS_BUILDDIR)/libnccl_test_os.a
NVCUFLAGS += -I$(TEST_OS_SRCDIR)
NVLDFLAGS += -L$(TEST_OS_BUILDDIR) -lnccl_test_os

build: os.build ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}

TEST_VERIFIABLE_SRCDIR := ../verifiable
TEST_VERIFIABLE_BUILDDIR := $(BUILDDIR)/verifiable
include ../verifiable/verifiable.mk

$(TEST_OS_LIB): os.build
	@:

os.%:
	${MAKE} -C $(TEST_OS_SRCDIR) $* BUILDDIR=$(BUILDDIR)

.PRECIOUS: ${DST_DIR}/%.o

${DST_DIR}/%.o: %.cu common.h util.h $(TEST_VERIFIABLE_HDRS)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/%$(NAME_SUFFIX).o: %.cu common.h util.h $(TEST_VERIFIABLE_HDRS)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/timer.o: timer.cc timer.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(CXXFLAGS) -o $@ -c $<

ifeq ($(DSO), 1)
${DST_DIR}/%_perf$(NAME_SUFFIX): ${DST_DIR}/%.o ${DST_DIR}/common$(NAME_SUFFIX).o ${DST_DIR}/util$(NAME_SUFFIX).o ${DST_DIR}/timer.o $(TEST_VERIFIABLE_LIBS) $(TEST_OS_LIB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ -L$(TEST_VERIFIABLE_BUILDDIR) -lverifiable ${NVLDFLAGS} -Xlinker "--enable-new-dtags" -Xlinker "-rpath,\$$ORIGIN:\$$ORIGIN/verifiable"
else
${DST_DIR}/%_perf$(NAME_SUFFIX):${DST_DIR}/%.o ${DST_DIR}/common$(NAME_SUFFIX).o ${DST_DIR}/util$(NAME_SUFFIX).o ${DST_DIR}/timer.o $(TEST_VERIFIABLE_OBJS) $(TEST_OS_LIB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}
endif

${DST_DIR}/comm_ops_perf$(NAME_SUFFIX): ${DST_DIR}/comm_ops.o $(TEST_OS_LIB)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

clean_intermediates:
	rm -f ${DST_DIR}/*.o $(TEST_VERIFIABLE_OBJS)

# Point-to-point GIN perf sub-framework. Each benchmark category (latency and
# throughput) has runner sources and operation sources; the operation objects
# are archived once per category and linked into every category runner.
DEVICE_API_GIN_SRCDIR       := $(abspath device_api/gin)
DEVICE_API_GIN_UTILS_SRCDIR := $(DEVICE_API_GIN_SRCDIR)/utils
DEVICE_API_GIN_UTILS_DST    := $(BUILDDIR)/device_api/gin/utils
DEVICE_API_GIN_MPI_ENABLED  := $(filter 1,$(MPI))
DEVICE_API_GIN_EXTRA_INC := -I$(DEVICE_API_GIN_UTILS_SRCDIR) -I$(NCCL_HOME)/include -I$(MPI_HOME)/include \
                            -I$(CUDA_INC) -I$(CUDA_INC)/cccl
DEVICE_API_GIN_NVLDFLAGS := $(filter-out -lnccl_test_os -L$(TEST_OS_BUILDDIR),$(NVLDFLAGS))

DEVICE_API_GIN_UTILS_CC_SRCS := $(wildcard $(DEVICE_API_GIN_UTILS_SRCDIR)/*.cc)
DEVICE_API_GIN_UTILS_CU_SRCS := $(wildcard $(DEVICE_API_GIN_UTILS_SRCDIR)/*.cu)
DEVICE_API_GIN_UTILS_OBJS := $(patsubst $(DEVICE_API_GIN_UTILS_SRCDIR)/%.cc,$(DEVICE_API_GIN_UTILS_DST)/%.o,$(DEVICE_API_GIN_UTILS_CC_SRCS)) \
                             $(patsubst $(DEVICE_API_GIN_UTILS_SRCDIR)/%.cu,$(DEVICE_API_GIN_UTILS_DST)/%.o,$(DEVICE_API_GIN_UTILS_CU_SRCS))

$(DEVICE_API_GIN_UTILS_DST)/%.o: $(DEVICE_API_GIN_UTILS_SRCDIR)/%.cc
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(DEVICE_API_GIN_UTILS_DST)
	$(CXX) $(CXXFLAGS) $(DEVICE_API_GIN_EXTRA_INC) -o $@ -c $<

$(DEVICE_API_GIN_UTILS_DST)/%.o: $(DEVICE_API_GIN_UTILS_SRCDIR)/%.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(DEVICE_API_GIN_UTILS_DST)
	$(NVCC) -o $@ $(DEVICE_API_GIN_EXTRA_INC) $(NVCUFLAGS) -c $<

# Allow a single benchmark to be requested with the repository-relative path.
build/device_api/gin/%: $(BUILDDIR)/device_api/gin/%
	@:

define DEVICE_API_GIN_CATEGORY
DEVICE_API_GIN_$(1)_SRCDIR := $$(DEVICE_API_GIN_SRCDIR)/$(1)
DEVICE_API_GIN_$(1)_DST := $$(BUILDDIR)/device_api/gin/$(1)
DEVICE_API_GIN_$(1)_MAIN_SRCS := $$(wildcard $$(DEVICE_API_GIN_$(1)_SRCDIR)/*_main.cu)
DEVICE_API_GIN_$(1)_TEST_SRCS := $$(filter-out $$(DEVICE_API_GIN_$(1)_MAIN_SRCS),$$(wildcard $$(DEVICE_API_GIN_$(1)_SRCDIR)/*.cu))
DEVICE_API_GIN_$(1)_TEST_OBJS := $$(patsubst $$(DEVICE_API_GIN_$(1)_SRCDIR)/%.cu,$$(DEVICE_API_GIN_$(1)_DST)/%.o,$$(DEVICE_API_GIN_$(1)_TEST_SRCS))
DEVICE_API_GIN_$(1)_MAIN_OBJS := $$(patsubst $$(DEVICE_API_GIN_$(1)_SRCDIR)/%.cu,$$(DEVICE_API_GIN_$(1)_DST)/%.o,$$(DEVICE_API_GIN_$(1)_MAIN_SRCS))
DEVICE_API_GIN_$(1)_LIB := $$(DEVICE_API_GIN_$(1)_DST)/libginPerf$(1).a
DEVICE_API_GIN_$(1)_BINS := $$(patsubst $$(DEVICE_API_GIN_$(1)_SRCDIR)/%_main.cu,$$(DEVICE_API_GIN_$(1)_DST)/%_perf,$$(DEVICE_API_GIN_$(1)_MAIN_SRCS))

ifneq ($$(DEVICE_API_GIN_MPI_ENABLED),)
build: $$(DEVICE_API_GIN_$(1)_BINS)
endif

$$(DEVICE_API_GIN_$(1)_TEST_OBJS) $$(DEVICE_API_GIN_$(1)_MAIN_OBJS): $$(DEVICE_API_GIN_$(1)_DST)/%.o: $$(DEVICE_API_GIN_$(1)_SRCDIR)/%.cu
	@printf "Compiling  %-35s > %s\n" $$< $$@
	@mkdir -p $$(DEVICE_API_GIN_$(1)_DST)
	$$(NVCC) -o $$@ $$(DEVICE_API_GIN_EXTRA_INC) $$(NVCUFLAGS) -c $$<

$$(DEVICE_API_GIN_$(1)_LIB): $$(DEVICE_API_GIN_$(1)_TEST_OBJS)
	@printf "Archiving  %-35s > %s\n" $(1) $$@
	@mkdir -p $$(DEVICE_API_GIN_$(1)_DST)
	$$(AR) rcs $$@ $$^

ifeq ($$(DEVICE_API_GIN_MPI_ENABLED),)
$$(DEVICE_API_GIN_$(1)_BINS):
	$$(error GIN device API performance tests require MPI=1)
else
$$(DEVICE_API_GIN_$(1)_BINS): $$(DEVICE_API_GIN_$(1)_DST)/%_perf: $$(DEVICE_API_GIN_$(1)_DST)/%_main.o $$(DEVICE_API_GIN_$(1)_LIB) $$(DEVICE_API_GIN_UTILS_OBJS)
	@printf "Linking  %-35s > %s\n" $$< $$@
	@mkdir -p $$(DEVICE_API_GIN_$(1)_DST)
	$$(NVCC) -o $$@ $$(NVCUFLAGS) $$< $$(DEVICE_API_GIN_UTILS_OBJS) $$(DEVICE_API_GIN_$(1)_LIB) $$(DEVICE_API_GIN_NVLDFLAGS)
endif
endef

$(eval $(call DEVICE_API_GIN_CATEGORY,latency))
$(eval $(call DEVICE_API_GIN_CATEGORY,throughput))
