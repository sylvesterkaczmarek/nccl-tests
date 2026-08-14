#include "common.h"
#include "args.h"
#include "ginLatencyTests.h"

/*
 * Ping-pong latency of the put family. There is no put+counter variant here: a
 * counter reports local completion, which cannot drive the peer's turn.
 */
int main(int argc, char** argv) {
  args_t args;
  parse_args(argc, argv, &args);

  gin_test_caps_t caps = {};
  caps.default_rsm = GIN_RSM_GPU;
  caps.allow_thread_rsm = true;
  caps.op_mask = GIN_OP_BIT(GIN_OP_PUT) | GIN_OP_BIT(GIN_OP_PUT_SIGNAL);
  caps.default_op = GIN_OP_PUT;
  caps.is_signal_op = false;
  caps.allow_aggregate_requests = false;
  caps.allow_bidirectional = false;
  caps.allow_multi_cta_threads = false;
  configure_test_args(&args, &caps);

  gin_benchmark_t bench = {};
  bench.type = GIN_BENCHMARK_TYPE_PING_PONG;
  switch (args.gin_op) {
  case GIN_OP_PUT_SIGNAL:
    bench.run = ginPutSignalLatency_pingPong_launch;
    bench.name = "GIN PUT_SIGNAL PING-PONG LATENCY";
    break;
  default:
    bench.run = ginPutLatency_pingPong_launch;
    bench.name = "GIN PUT PING-PONG LATENCY";
    break;
  }

  gin_perf_run(argc, argv, &args, &bench);
  return 0;
}
