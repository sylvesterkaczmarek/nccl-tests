/*************************************************************************
 * Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PROFILER_H_
#define NCCL_PROFILER_H_

#include "nccl.h"
#include <stdint.h>
#include <stdlib.h>

#if defined(NCCL_OS_WINDOWS)
typedef unsigned long pid_t;
#else
#include <sys/types.h>
#endif

typedef enum {NCCL_LOG_NONE=0, NCCL_LOG_VERSION=1, NCCL_LOG_WARN=2, NCCL_LOG_INFO=3, NCCL_LOG_ABORT=4, NCCL_LOG_TRACE=5} ncclDebugLogLevel;
typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

enum {
  ncclProfileGroup          = (1 << 0),  // group event type
  ncclProfileColl           = (1 << 1),  // host collective call event type
  ncclProfileP2p            = (1 << 2),  // host point-to-point call event type
  ncclProfileProxyOp        = (1 << 3),  // proxy operation event type
  ncclProfileProxyStep      = (1 << 4),  // proxy step event type
  ncclProfileProxyCtrl      = (1 << 5),  // proxy control event type
  ncclProfileKernelCh       = (1 << 6),  // kernel channel event type
  ncclProfileNetPlugin      = (1 << 7),  // network plugin-defined events
  ncclProfileGroupApi       = (1 << 8),  // group API events
  ncclProfileCollApi        = (1 << 9),  // collective API events
  ncclProfileP2pApi         = (1 << 10), // point-to-point API events
  ncclProfileKernelLaunch   = (1 << 11), // kernel launch events
  ncclProfileCeColl         = (1 << 12), // CE collective operation
  ncclProfileCeSync         = (1 << 13), // CE synchronization operation
  ncclProfileCeBatch        = (1 << 14), // CE batch operation
  ncclProfileKernelPhase    = (1 << 15), // kernel barrier phase sub-event
};

typedef enum {
  ncclProfilerProxyOpSendPosted        = 0,
  ncclProfilerProxyOpSendRemFifoWait   = 1,
  ncclProfilerProxyOpSendTransmitted   = 2,
  ncclProfilerProxyOpSendDone          = 3,
  ncclProfilerProxyOpRecvPosted        = 4,
  ncclProfilerProxyOpRecvReceived      = 5,
  ncclProfilerProxyOpRecvTransmitted   = 6,
  ncclProfilerProxyOpRecvDone          = 7,
  ncclProfilerProxyStepSendGPUWait     = 8,
  ncclProfilerProxyStepSendWait        = 9,
  ncclProfilerProxyStepRecvWait        = 10,
  ncclProfilerProxyStepRecvFlushWait   = 11,
  ncclProfilerProxyStepRecvGPUWait     = 12,
  ncclProfilerProxyCtrlIdle            = 13,
  ncclProfilerProxyCtrlActive          = 14,
  ncclProfilerProxyCtrlSleep           = 15,
  ncclProfilerProxyCtrlWakeup          = 16,
  ncclProfilerProxyCtrlAppend          = 17,
  ncclProfilerProxyCtrlAppendEnd       = 18,
  ncclProfilerProxyOpInProgress_v4     = 19,
  ncclProfilerProxyStepSendPeerWait_v4 = 20,
  ncclProfilerNetPluginUpdate          = 21,
  ncclProfilerKernelChStop             = 22,
  ncclProfilerGroupStartApiStop        = 23,
  ncclProfilerGroupEndApiStart         = 24,
  ncclProfilerCeCollStart               = 25,
  ncclProfilerCeCollComplete            = 26,
  ncclProfilerCeSyncStart               = 27,
  ncclProfilerCeSyncComplete            = 28,
  ncclProfilerCeBatchStart              = 29,
  ncclProfilerCeBatchComplete           = 30,
  ncclProfilerKernelPhaseStop           = 31,
} ncclProfilerEventState_t;

typedef ncclProfilerEventState_t ncclProfilerEventState_v5_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v6_t;
typedef ncclProfilerEventState_t ncclProfilerEventState_v7_t;

// Copied verbatim from the published NCCL profiler example headers.
#include "profiler_v5.h"
#include "profiler_v6.h"
#include "profiler_v7.h"

#endif
