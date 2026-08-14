#pragma once

#include "common.h"

/*
 * Launch entry points for the latency tests, one per GIN operation, each
 * implemented by the like-named .cu file in this directory. The *_main.cu
 * runners select one of these according to --gin_op; the linker then pulls only
 * the referenced implementations out of the category archive.
 *
 * Every entry point matches gin_run_fn_t so it can be assigned to
 * gin_benchmark_t::run directly.
 */

void ginPutLatency_ping_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                               const args_t* args, size_t numElems, int iters);
void ginPutSignalLatency_ping_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                     const args_t* args, size_t numElems, int iters);
void ginPutCounterLatency_ping_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                      const args_t* args, size_t numElems, int iters);

void ginGetLatency_ping_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                               const args_t* args, size_t numElems, int iters);

void ginSignalLatency_ping_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                  const args_t* args, size_t numElems, int iters);

void ginPutLatency_pingPong_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                   const args_t* args, size_t numElems, int iters);
void ginPutSignalLatency_pingPong_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                         const args_t* args, size_t numElems, int iters);

void ginSignalLatency_pingPong_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                                      const args_t* args, size_t numElems, int iters);
