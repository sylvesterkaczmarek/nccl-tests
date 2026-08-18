#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, ncclGinResourceSharingMode rsm>
__global__ void ginPutCounter_ping_kernel(ncclDevComm comm, ncclDevResourceHandle hBuf, int iters, size_t numElems,
                                  int queueDepth) {
#if __CUDA_ARCH__ >= 700
  ncclTeam team = ncclTeamWorld(comm);
  ncclGin gin(comm, 0, rsm);
  ncclSymPtr<int> sbuf = (ncclSymPtr<int>)ncclGetResourceBuffer(comm, hBuf);
  ncclSymPtr<int> dbuf = sbuf;

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  constexpr ncclGinCounter_t counterId = 0;
  uint64_t counterShadow = 0;
  const int flushEvery = queueDepth / 2 < 1 ? 1 : queueDepth / 2;

  for (int i = 0; i < iters; i++) {
    gin.put(team, 1, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_WeakCounterInc{counterId}, ncclCoopThread{},
            ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
    counterShadow++;
    if (skipCreditCheck && (i % flushEvery == 0)) gin.flush(ncclCoopThread{});
    gin.waitCounter(ncclCoopThread{}, counterId, counterShadow);
  }

  gin.flush(ncclCoopThread{});
  gin.resetCounter(counterId);
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginPutCounterLatency_ping_launch_rsm(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                                 const args_t* args, size_t numElems, int iters) {
  const int queueDepth = args->queue_depth;
#define LAUNCH_PING_PUT_COUNTER(SKIP) \
  ginPutCounter_ping_kernel<SKIP, rsm><<<1, 1, 0, stream>>>(dcomm, hBuf, iters, numElems, queueDepth)

  if (args->gin_skip_credit_check) {
    LAUNCH_PING_PUT_COUNTER(true);
  } else {
    LAUNCH_PING_PUT_COUNTER(false);
  }

#undef LAUNCH_PING_PUT_COUNTER
  CUDACHECK_FATAL(cudaGetLastError());
}

void ginPutCounterLatency_ping_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                      const args_t* args, size_t numElems, int iters) {
  if (args->gin_rsm == GIN_RSM_THREAD)
    ginPutCounterLatency_ping_launch_rsm<NCCL_GIN_RESOURCE_SHARING_THREAD>(dcomm, hBuf, stream, args, numElems, iters);
  else if (args->gin_rsm == GIN_RSM_CTA)
    ginPutCounterLatency_ping_launch_rsm<NCCL_GIN_RESOURCE_SHARING_CTA>(dcomm, hBuf, stream, args, numElems, iters);
  else
    ginPutCounterLatency_ping_launch_rsm<NCCL_GIN_RESOURCE_SHARING_GPU>(dcomm, hBuf, stream, args, numElems, iters);
}
