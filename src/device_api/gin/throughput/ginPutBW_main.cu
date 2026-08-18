#include "common.h"
#include "args.h"
#include "ginThroughputTests.h"

/* Bandwidth / message rate of the put family: put, put+signal or put+counter. */
int main(int argc, char** argv) {
  args_t args;
  parse_args(argc, argv, &args);

  gin_test_caps_t caps = {};
  caps.default_rsm = GIN_RSM_GPU;
  caps.allow_thread_rsm = false;
  caps.op_mask = GIN_OP_BIT(GIN_OP_PUT) | GIN_OP_BIT(GIN_OP_PUT_SIGNAL) | GIN_OP_BIT(GIN_OP_PUT_COUNTER);
  caps.default_op = GIN_OP_PUT;
  caps.is_signal_op = false;
  caps.allow_aggregate_requests = true;
  caps.allow_bidirectional = true;
  caps.allow_multi_cta_threads = true;
  configure_test_args(&args, &caps);

  gin_benchmark_t bench = {};
  bench.type = GIN_BENCHMARK_TYPE_THROUGHPUT;
  switch (args.gin_op) {
  case GIN_OP_PUT_SIGNAL:
    bench.run = ginPutSignalBW_launch;
    bench.name = "GIN PUT_SIGNAL BANDWIDTH/MESSAGE RATE";
    break;
  case GIN_OP_PUT_COUNTER:
    bench.run = ginPutCounterBW_launch;
    bench.name = "GIN PUT_COUNTER BANDWIDTH/MESSAGE RATE";
    break;
  default:
    bench.run = ginPutBW_launch;
    bench.name = "GIN PUT BANDWIDTH/MESSAGE RATE";
    break;
  }

  gin_perf_run(argc, argv, &args, &bench);
  return 0;
}
