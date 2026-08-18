#include "common.h"
#include "args.h"
#include "ginThroughputTests.h"

/* Bandwidth / message rate of get. Single operation, so --gin_op does not apply. */
int main(int argc, char** argv) {
  args_t args;
  parse_args(argc, argv, &args);

  gin_test_caps_t caps = {};
  caps.default_rsm = GIN_RSM_GPU;
  caps.allow_thread_rsm = false;
  caps.op_mask = 0;
  caps.default_op = GIN_OP_UNSET;
  caps.is_signal_op = false;
  caps.allow_aggregate_requests = true;
  caps.allow_bidirectional = true;
  caps.allow_multi_cta_threads = true;
  configure_test_args(&args, &caps);

  gin_benchmark_t bench = {};
  bench.run = ginGetBW_launch;
  bench.name = "GIN GET BANDWIDTH/MESSAGE RATE";
  bench.type = GIN_BENCHMARK_TYPE_THROUGHPUT;

  gin_perf_run(argc, argv, &args, &bench);
  return 0;
}
