#include "gin_context.h"
#include "common.h"

/*
 * The build only produces these benchmarks against headers new enough for the
 * GIN requirements below, so this guards the remaining case the build system
 * cannot see: a library older than those headers loaded at run time.
 */
static void check_nccl_version(void) {
  int runtimeVersion = 0;
  NCCLCHECK_FATAL(ncclGetVersion(&runtimeVersion));
  if (runtimeVersion < NCCL_VERSION(2, 30, 7)) {
    fprintf(stderr,
            "Incompatible NCCL versions. nccl-tests was compiled with NCCL %d, but is "
            "running with NCCL %d. The GIN device API is not compatible with versions "
            "before 2.30.7.\n",
            NCCL_VERSION_CODE, runtimeVersion);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

size_t gin_max_buffer_bytes(void) {
  return 20ULL * (1ULL << 31);
}

void gin_devComm_create(ncclComm_t comm, const args_t* args, gin_context_t* ctx) {
  check_nccl_version();

  ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginContextCount = args->num_ctas;
  reqs.ginQueueDepth = args->queue_depth;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginSignalCount = 1;
  reqs.ginCounterCount = 1;

  reqs.ginStrongSignalsRequired = args->gin_strong_signal;
  reqs.ginVaSignalsRequired = false;

  ncclDevResourceRequirements bufReq = {};
  bufReq.bufferSize = args->maxbytes;
  bufReq.outBufferHandle = &ctx->hBuf;
  bufReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &bufReq;

  NCCLCHECK_FATAL(ncclDevCommCreate(comm, &reqs, &ctx->dcomm));
}

void gin_devComm_destroy(ncclComm_t comm, gin_context_t* ctx) {
    NCCLCHECK_FATAL(ncclDevCommDestroy(comm, &ctx->dcomm));
}
