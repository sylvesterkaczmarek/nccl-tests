#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, bool aggregateRequests, ncclGinResourceSharingMode rsm>
__global__ void ginPutCounter_bw_kernel(ncclDevComm comm, ncclDevResourceHandle hBuf, int iters, size_t numElems,
                                        int queueDepth, size_t maxElems) {
#if __CUDA_ARCH__ >= 700
  const int tag = blockIdx.x;
  ncclTeam team = ncclTeamWorld(comm);
  const int peer = team.rank ^ 1;
  ncclGin gin(comm, tag, rsm);

  const size_t slots = maxElems / numElems;
  ncclSymPtr<int> buf = (ncclSymPtr<int>)ncclGetResourceBuffer(comm, hBuf);
  buf += (size_t)(threadIdx.x % slots) * numElems;
  ncclSymPtr<int> sbuf = buf;
  ncclSymPtr<int> dbuf = buf;

  const int activeThreads = (int)blockDim.x;
  const int lastActive = activeThreads - 1;

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  const int wqesPerOp = 2;
  const int wqesPerIter = blockDim.x * wqesPerOp;
  const int flushEvery = (queueDepth / 2) / wqesPerIter < 1 ? 1 : (queueDepth / 2) / wqesPerIter;

  constexpr ncclGinCounter_t counterId = 0;

  for (int i = 0; i < iters; i++) {
    if (aggregateRequests) {
      if (threadIdx.x != lastActive) {
        gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_WeakCounterInc{counterId}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread,
                optFlags | ncclGinOptFlagsAggregateRequests);
      }
      __syncthreads();
      if (threadIdx.x == lastActive) {
        gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_WeakCounterInc{counterId}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      }
    } else {
      gin.put(team, peer, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_WeakCounterInc{counterId}, ncclCoopThread{},
              ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
    }
    if (skipCreditCheck) {
      if (i % flushEvery == 0) gin.flush(ncclCoopCta{});
    } else {
      __syncthreads();
    }
  }
  gin.flush(ncclCoopCta{});
  gin.resetCounter(counterId);
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginPutCounterBW_launch_rsm(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                       const args_t* args, size_t numElems, int iters) {
  const int queueDepth = args->queue_depth;
  const size_t maxElems = args->maxbytes / sizeof(int);
#define LAUNCH_BW_PUT_COUNTER(SKIP, AG) \
  ginPutCounter_bw_kernel<SKIP, AG, rsm><<<args->num_ctas, args->num_threads, 0, stream>>>(dcomm, hBuf, iters, numElems, queueDepth, maxElems)

  if (args->gin_skip_credit_check) {
    if (args->gin_aggregate_requests) LAUNCH_BW_PUT_COUNTER(true, true);
    else                              LAUNCH_BW_PUT_COUNTER(true, false);
  } else {
    if (args->gin_aggregate_requests) LAUNCH_BW_PUT_COUNTER(false, true);
    else                              LAUNCH_BW_PUT_COUNTER(false, false);
  }

#undef LAUNCH_BW_PUT_COUNTER
  CUDACHECK_FATAL(cudaGetLastError());
}

void ginPutCounterBW_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                            const args_t* args, size_t numElems, int iters) {
  if (args->gin_rsm == GIN_RSM_CTA)
    ginPutCounterBW_launch_rsm<NCCL_GIN_RESOURCE_SHARING_CTA>(dcomm, hBuf, stream, args, numElems, iters);
  else
    ginPutCounterBW_launch_rsm<NCCL_GIN_RESOURCE_SHARING_GPU>(dcomm, hBuf, stream, args, numElems, iters);
}
