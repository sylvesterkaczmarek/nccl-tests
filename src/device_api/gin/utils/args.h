#pragma once

#include <stddef.h>

typedef enum {
  GIN_RSM_UNSET = 0,
  GIN_RSM_THREAD,
  GIN_RSM_CTA,
  GIN_RSM_GPU,
} gin_rsm_t;

/*
 * GIN operation under test, selected by --gin_op. A benchmark binary links one
 * test implementation per operation it supports and dispatches to the matching
 * one, so e.g. the put binaries cover plain put, put+signal and put+counter.
 */
typedef enum {
  GIN_OP_UNSET = 0,
  GIN_OP_PUT,
  GIN_OP_PUT_SIGNAL,
  GIN_OP_PUT_COUNTER,
} gin_op_t;

#define GIN_OP_BIT(op) (1u << (unsigned)(op))

typedef struct {
  size_t minbytes;
  size_t maxbytes;
  int stepfactor;
  size_t stepbytes;
  int warmup_iters;
  int iters;
  int num_ctas;
  int num_threads;
  int gin_skip_credit_check;
  bool gin_strong_signal;
  bool gin_aggregate_requests;
  bool gin_bidirectional;
  int queue_depth;
  gin_rsm_t gin_rsm;
  gin_op_t gin_op;
} args_t;

/*
 * What a benchmark binary actually implements. Anything a binary does not
 * declare here is rejected by configure_test_args with a diagnostic rather than
 * being silently ignored.
 *
 *   default_rsm / allow_thread_rsm – --gin_rsm default (gpu) and whether thread
 *                                    mode is usable (latency yes, throughput no).
 *   op_mask                        – OR of GIN_OP_BIT() values --gin_op accepts;
 *                                    0 for binaries with a single operation
 *                                    (the get and signal tests).
 *   default_op                     – used when --gin_op is omitted.
 *   is_signal_op                   – the binary's operation is itself a signal,
 *                                    so --gin_strong_signal applies regardless
 *                                    of --gin_op.
 *   allow_aggregate_requests       – --gin_ag (throughput only).
 *   allow_bidirectional            – --gin_bd (throughput only).
 *   allow_multi_cta_threads        – -c/-t other than 1 (throughput only;
 *                                    latency requires a single CTA and thread).
 */
typedef struct {
  gin_rsm_t default_rsm;
  bool allow_thread_rsm;
  unsigned op_mask;
  gin_op_t default_op;
  bool is_signal_op;
  bool allow_aggregate_requests;
  bool allow_bidirectional;
  bool allow_multi_cta_threads;
} gin_test_caps_t;

void parse_args(int argc, char** argv, args_t* args);
void configure_test_args(args_t* args, const gin_test_caps_t* caps);
const char* gin_op_name(gin_op_t op);
const char* gin_rsm_name(gin_rsm_t rsm);
