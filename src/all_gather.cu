/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

void AllGatherGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  size_t base = (count/nranks) & ~(16/eltSize - 1);
  *sendcount = base;
  *recvcount = base*nranks;
  *sendInplaceOffset = base;
  *recvInplaceOffset = 0;
  *paramcount = base;
}

testResult_t AllGatherInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? ((char*)args->recvbuffs[i])+rank*args->sendBytes : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep + rank, 1, 0));
    for (int j=0; j<nranks; j++) {
      TESTCHECK(InitData((char*)args->expected[i] + args->sendBytes*j, sendcount, 0, type, ncclSum, 33*rep + j, 1, 0));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

void AllGatherGetBw(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1))/((double)nranks);
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
testResult_t AllGatherRmaPut(void* sendWindow, size_t sendoffset, void* recvWindow, size_t recvoffset,
                             size_t count, ncclDataType_t type, ncclComm_t comm, cudaStream_t stream) {
  int rank, nranks;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  NCCLCHECK(ncclCommCount(comm, &nranks));

  ncclWindow_t sendWin = (ncclWindow_t)sendWindow;
  ncclWindow_t recvWin = (ncclWindow_t)recvWindow;

  void* sendPtr = NULL;
  void* recvPtr = NULL;
  NCCLCHECK(ncclWinGetUserPtr(comm, sendWin, &sendPtr));
  NCCLCHECK(ncclWinGetUserPtr(comm, recvWin, &recvPtr));

  size_t eltSize = wordSize(type);
  size_t bytes = count * eltSize;
  const int nctx = rmaCtxCount;

  bool isInPlace = ((char*)sendPtr + sendoffset == (char*)recvPtr + recvoffset + rank * bytes);
  size_t peerWinOffset = recvoffset + rank * bytes;

  ncclWaitSignalDesc_t* waitDescs = (ncclWaitSignalDesc_t*)malloc(sizeof(ncclWaitSignalDesc_t) * nranks);
  if (waitDescs == NULL) {
    return testInternalError;
  }

  int descIdx = 0;
  for (int i = 0; i < nranks; i++) {
    if (isInPlace && i == rank) {
      continue;
    }
    waitDescs[descIdx].opCnt = 1;
    waitDescs[descIdx].peer = i;
    waitDescs[descIdx].sigIdx = i % NUM_RMA_SIG;
    waitDescs[descIdx].ctx = (i + rank) % nctx;
    descIdx++;
  }

  NCCLCHECK(ncclGroupStart());
  for (int peer = 0; peer < nranks; peer++) {
    int targetRank = (rank + peer) % nranks;
    if (isInPlace && targetRank == rank) {
      continue;
    }
    NCCLCHECK(ncclPutSignal((char*)sendPtr + sendoffset, count, type, targetRank,
                      recvWin, peerWinOffset, rank % NUM_RMA_SIG, (rank + targetRank) % nctx, 0, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());

  NCCLCHECK(ncclWaitSignal(descIdx, waitDescs, comm, stream));
  free(waitDescs);
  return testSuccess;
}
#endif

testResult_t AllGatherRunColl(void* sendbuff,  size_t sendoffset,void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    NCCLCHECK(ncclAllGather(sptr, rptr, count, type, comm, stream));
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
  } else if (deviceImpl == HOST_RMA_IMPL) {
    TESTCHECK(AllGatherRmaPut(sendbuff, sendoffset, recvbuff, recvoffset, count, type, comm, stream));
#endif
  } else {
    return testNotImplemented;
  }
  return testSuccess;
}

struct testColl allGatherTest = {
  "AllGather",
  AllGatherGetCollByteCount,
  AllGatherInitData,
  AllGatherGetBw,
  AllGatherRunColl
};

void AllGatherGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllGatherGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AllGatherRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &allGatherTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) {
    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

NCCL_WEAK struct testEngine ncclTestEngine = {
  /* .getBuffSize = */ AllGatherGetBuffSize,
  /* .runTest = */ AllGatherRunTest
};
