#pragma once

#include "nccl.h"
#include "nccl_device.h"
#include "args.h"

typedef struct {
  ncclDevComm dcomm;
  ncclDevResourceHandle hBuf;
} gin_context_t;

size_t gin_max_buffer_bytes(void);

void gin_devComm_create(ncclComm_t comm, const args_t* args, gin_context_t* ctx);
void gin_devComm_destroy(ncclComm_t comm, gin_context_t* ctx);