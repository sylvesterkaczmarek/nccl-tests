#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "args.h"
#include "gin_context.h"

template <bool skipCreditCheck, ncclGinResourceSharingMode rsm>
__global__ void ginPut_pingPong_kernel(ncclDevComm comm, ncclDevResourceHandle hBuf, int iters, size_t numElems,
                                int queueDepth) {
#if __CUDA_ARCH__ >= 700
  ncclTeam team = ncclTeamWorld(comm);
  ncclGin gin(comm, 0, rsm);
  ncclSymPtr<int> sbuf = (ncclSymPtr<int>)ncclGetResourceBuffer(comm, hBuf);
  ncclSymPtr<int> dbuf = sbuf;

  constexpr uint32_t optFlags =
    skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  const int flushEvery = queueDepth / 2 < 1 ? 1 : queueDepth / 2;

  int* lastElem = (int*)ncclGetResourceBufferLocalPointer(comm, hBuf) + (numElems - 1);

  int serverSign = 1;
  int clientSign = -1;
  for (int i = 0; i < iters; i++) {
    if (comm.rank == 0) {
      while (*(volatile int*)lastElem != serverSign) continue;
      cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
      *lastElem = clientSign;
      gin.put(team, 1, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
              ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      if (skipCreditCheck && (i % flushEvery == 0)) gin.flush(ncclCoopThread{});
    } else {
      *lastElem = serverSign;
      gin.put(team, 0, dbuf, sbuf, numElems, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
              ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread, optFlags);
      if (skipCreditCheck && (i % flushEvery == 0)) gin.flush(ncclCoopThread{});
      while (*(volatile int*)lastElem != clientSign) continue;
      cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
    }
    clientSign--;
    serverSign++;
  }
  gin.flush(ncclCoopThread{});
#endif
}

template <ncclGinResourceSharingMode rsm>
static void ginPutLatency_pingPong_launch_rsm(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                              const args_t* args, size_t numElems, int iters) {
  const int queueDepth = args->queue_depth;
#define LAUNCH_PING_PONG_PUT(SKIP) \
  ginPut_pingPong_kernel<SKIP, rsm><<<1, 1, 0, stream>>>(dcomm, hBuf, iters, numElems, queueDepth)

  if (args->gin_skip_credit_check) {
    LAUNCH_PING_PONG_PUT(true);
  } else {
    LAUNCH_PING_PONG_PUT(false);
  }

#undef LAUNCH_PING_PONG_PUT
  CUDACHECK_FATAL(cudaGetLastError());
}

void ginPutLatency_pingPong_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                   const args_t* args, size_t numElems, int iters) {
  if (args->gin_rsm == GIN_RSM_THREAD)
    ginPutLatency_pingPong_launch_rsm<NCCL_GIN_RESOURCE_SHARING_THREAD>(dcomm, hBuf, stream, args, numElems, iters);
  else if (args->gin_rsm == GIN_RSM_CTA)
    ginPutLatency_pingPong_launch_rsm<NCCL_GIN_RESOURCE_SHARING_CTA>(dcomm, hBuf, stream, args, numElems, iters);
  else
    ginPutLatency_pingPong_launch_rsm<NCCL_GIN_RESOURCE_SHARING_GPU>(dcomm, hBuf, stream, args, numElems, iters);
}
