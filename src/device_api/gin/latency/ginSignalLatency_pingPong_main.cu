#include "common.h"
#include "args.h"
#include "ginLatencyTests.h"

/* Ping-pong latency of signal. Single operation, so --gin_op does not apply. */
int main(int argc, char** argv) {
  args_t args;
  parse_args(argc, argv, &args);

  gin_test_caps_t caps = {};
  caps.default_rsm = GIN_RSM_GPU;
  caps.allow_thread_rsm = true;
  caps.op_mask = 0;
  caps.default_op = GIN_OP_UNSET;
  caps.is_signal_op = true;
  caps.allow_aggregate_requests = false;
  caps.allow_bidirectional = false;
  caps.allow_multi_cta_threads = false;
  configure_test_args(&args, &caps);

  gin_benchmark_t bench = {};
  bench.run = ginSignalLatency_pingPong_launch;
  bench.name = "GIN SIGNAL PING-PONG LATENCY";
  bench.type = GIN_BENCHMARK_TYPE_PING_PONG;

  gin_perf_run(argc, argv, &args, &bench);
  return 0;
}
