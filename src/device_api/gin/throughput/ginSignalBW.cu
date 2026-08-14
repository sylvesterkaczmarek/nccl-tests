#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, bool aggregateRequests, bool strongSignal,
          ncclGinResourceSharingMode rsm>
__global__ void ginSignal_bw_kernel(ncclDevComm comm, ncclDevResourceHandle hBuf, int iters, int queueDepth) {
#if __CUDA_ARCH__ >= 700
  const int tag = blockIdx.x;
  ncclTeam team = ncclTeamWorld(comm);
  const int peer = team.rank ^ 1;
  ncclGin gin(comm, tag, rsm);

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  const int wqesPerOp = 1;
  const int wqesPerIter = blockDim.x * wqesPerOp;
  const int flushEvery = (queueDepth / 2) / wqesPerIter < 1 ? 1 : (queueDepth / 2) / wqesPerIter;

  constexpr ncclGinSignal_t signalId = 0;

  for (int i = 0; i < iters; i++) {
    if (aggregateRequests) {
      if (threadIdx.x < blockDim.x - 1) {
        if (strongSignal) {
          gin.signal(team, peer, ncclGin_StrongSignalInc{signalId}, ncclCoopThread{}, ncclGin_None{},
                     cuda::thread_scope_thread, cuda::thread_scope_thread,
                     optFlags | ncclGinOptFlagsAggregateRequests);
        } else {
          gin.signal(team, peer, ncclGin_WeakSignalInc{signalId}, ncclCoopThread{}, ncclGin_None{},
                     cuda::thread_scope_thread, cuda::thread_scope_thread,
                     optFlags | ncclGinOptFlagsAggregateRequests);
        }
      }
      __syncthreads();
      if (threadIdx.x == blockDim.x - 1) {
        if (strongSignal) {
          gin.signal(team, peer, ncclGin_StrongSignalInc{signalId}, ncclCoopThread{}, ncclGin_None{},
                     cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
        } else {
          gin.signal(team, peer, ncclGin_WeakSignalInc{signalId}, ncclCoopThread{}, ncclGin_None{},
                     cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
        }
      }
    } else {
      if (strongSignal) {
        gin.signal(team, peer, ncclGin_StrongSignalInc{signalId}, ncclCoopThread{}, ncclGin_None{},
                   cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      } else {
        gin.signal(team, peer, ncclGin_WeakSignalInc{signalId}, ncclCoopThread{}, ncclGin_None{},
                   cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      }
    }
    if (skipCreditCheck) {
      if (i % flushEvery == 0) gin.flush(ncclCoopCta{});
    } else {
      __syncthreads();
    }
  }
  gin.flush(ncclCoopCta{});
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginSignalBW_launch_rsm(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                   const args_t* args, int iters) {
  const int queueDepth = args->queue_depth;
#define LAUNCH_BW_SIGNAL(SKIP, AG, STRONG) \
  ginSignal_bw_kernel<SKIP, AG, STRONG, rsm><<<args->num_ctas, args->num_threads, 0, stream>>>(dcomm, hBuf, iters, queueDepth)

  if (args->gin_skip_credit_check) {
    if (args->gin_aggregate_requests) {
      if (args->gin_strong_signal) LAUNCH_BW_SIGNAL(true, true, true);
      else                         LAUNCH_BW_SIGNAL(true, true, false);
    } else {
      if (args->gin_strong_signal) LAUNCH_BW_SIGNAL(true, false, true);
      else                         LAUNCH_BW_SIGNAL(true, false, false);
    }
  } else {
    if (args->gin_aggregate_requests) {
      if (args->gin_strong_signal) LAUNCH_BW_SIGNAL(false, true, true);
      else                         LAUNCH_BW_SIGNAL(false, true, false);
    } else {
      if (args->gin_strong_signal) LAUNCH_BW_SIGNAL(false, false, true);
      else                         LAUNCH_BW_SIGNAL(false, false, false);
    }
  }

#undef LAUNCH_BW_SIGNAL
  CUDACHECK_FATAL(cudaGetLastError());
}

void ginSignalBW_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                        const args_t* args, size_t numElems, int iters) {
  if (args->gin_rsm == GIN_RSM_CTA)
    ginSignalBW_launch_rsm<NCCL_GIN_RESOURCE_SHARING_CTA>(dcomm, hBuf, stream, args, iters);
  else
    ginSignalBW_launch_rsm<NCCL_GIN_RESOURCE_SHARING_GPU>(dcomm, hBuf, stream, args, iters);
}
