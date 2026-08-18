#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, ncclGinResourceSharingMode rsm>
__global__ void ginGet_ping_kernel(ncclDevComm comm, ncclDevResourceHandle hBuf, int iters, size_t bytes) {
#if __CUDA_ARCH__ >= 700
  ncclTeam team = ncclTeamWorld(comm);
  ncclGin gin(comm, 0, rsm);
  ncclSymPtr<int> sbuf = (ncclSymPtr<int>)ncclGetResourceBuffer(comm, hBuf);

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  for (int i = 0; i < iters; i++) {
    gin.get(team, 1, sbuf.window, sbuf.offset, sbuf.window, sbuf.offset, bytes, ncclCoopThread{},
            ncclGin_None{}, optFlags);
    gin.flush(ncclCoopThread{});
  }
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginGetLatency_ping_launch_rsm(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                          const args_t* args, size_t numElems, int iters) {
  size_t bytes = numElems * sizeof(int);
#define LAUNCH_PING_GET(SKIP) \
  ginGet_ping_kernel<SKIP, rsm><<<1, 1, 0, stream>>>(dcomm, hBuf, iters, bytes)

  if (args->gin_skip_credit_check) {
    LAUNCH_PING_GET(true);
  } else {
    LAUNCH_PING_GET(false);
  }

#undef LAUNCH_PING_GET
  CUDACHECK_FATAL(cudaGetLastError());
}

void ginGetLatency_ping_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                               const args_t* args, size_t numElems, int iters) {
  if (args->gin_rsm == GIN_RSM_THREAD)
    ginGetLatency_ping_launch_rsm<NCCL_GIN_RESOURCE_SHARING_THREAD>(dcomm, hBuf, stream, args, numElems, iters);
  else if (args->gin_rsm == GIN_RSM_CTA)
    ginGetLatency_ping_launch_rsm<NCCL_GIN_RESOURCE_SHARING_CTA>(dcomm, hBuf, stream, args, numElems, iters);
  else
    ginGetLatency_ping_launch_rsm<NCCL_GIN_RESOURCE_SHARING_GPU>(dcomm, hBuf, stream, args, numElems, iters);
}
