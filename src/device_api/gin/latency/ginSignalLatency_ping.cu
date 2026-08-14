#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, bool strongSignal, ncclGinResourceSharingMode rsm>
__global__ void ginSignal_ping_kernel(ncclDevComm comm, ncclDevResourceHandle hBuf, int iters) {
#if __CUDA_ARCH__ >= 700
  ncclTeam team = ncclTeamWorld(comm);
  ncclGin gin(comm, 0, rsm);

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  constexpr ncclGinSignal_t signalId = 0;

  for (int i = 0; i < iters; i++) {
    if (strongSignal) {
      gin.signal(team, 1, ncclGin_StrongSignalInc{signalId}, ncclCoopThread{}, ncclGin_None{},
                 cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
    } else {
      gin.signal(team, 1, ncclGin_WeakSignalInc{signalId}, ncclCoopThread{}, ncclGin_None{},
                 cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
    }
    gin.flush(ncclCoopThread{});
  }
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginSignalLatency_ping_launch_rsm(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                             const args_t* args, int iters) {
#define LAUNCH_PING_SIGNAL(SKIP, STRONG) \
  ginSignal_ping_kernel<SKIP, STRONG, rsm><<<1, 1, 0, stream>>>(dcomm, hBuf, iters)

  if (args->gin_skip_credit_check) {
    if (args->gin_strong_signal) LAUNCH_PING_SIGNAL(true, true);
    else                         LAUNCH_PING_SIGNAL(true, false);
  } else {
    if (args->gin_strong_signal) LAUNCH_PING_SIGNAL(false, true);
    else                         LAUNCH_PING_SIGNAL(false, false);
  }

#undef LAUNCH_PING_SIGNAL
  CUDACHECK_FATAL(cudaGetLastError());
}

void ginSignalLatency_ping_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                  const args_t* args, size_t numElems, int iters) {
  if (args->gin_rsm == GIN_RSM_THREAD)
    ginSignalLatency_ping_launch_rsm<NCCL_GIN_RESOURCE_SHARING_THREAD>(dcomm, hBuf, stream, args, iters);
  else if (args->gin_rsm == GIN_RSM_CTA)
    ginSignalLatency_ping_launch_rsm<NCCL_GIN_RESOURCE_SHARING_CTA>(dcomm, hBuf, stream, args, iters);
  else
    ginSignalLatency_ping_launch_rsm<NCCL_GIN_RESOURCE_SHARING_GPU>(dcomm, hBuf, stream, args, iters);
}
