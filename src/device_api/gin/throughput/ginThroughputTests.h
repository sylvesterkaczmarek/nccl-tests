#pragma once

#include "common.h"

/*
 * Launch entry points for the throughput (bandwidth / message rate) tests, one
 * per GIN operation, each implemented by the like-named .cu file in this
 * directory. The *_main.cu runners select one of these according to --gin_op;
 * the linker then pulls only the referenced implementations out of the category
 * archive.
 *
 * Every entry point matches gin_run_fn_t so it can be assigned to
 * gin_benchmark_t::run directly.
 */

void ginPutBW_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                     const args_t* args, size_t numElems, int iters);
void ginPutSignalBW_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                           const args_t* args, size_t numElems, int iters);
void ginPutCounterBW_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                            const args_t* args, size_t numElems, int iters);

void ginGetBW_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                     const args_t* args, size_t numElems, int iters);

void ginSignalBW_launch(ncclDevComm dcomm, ncclDevResourceHandle hBuf, cudaStream_t stream,
                        const args_t* args, size_t numElems, int iters);
