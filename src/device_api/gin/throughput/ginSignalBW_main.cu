#include "common.h"
#include "args.h"
#include "ginThroughputTests.h"

/*
 * Message rate of signal. Signals carry no payload, so bandwidth is reported as
 * zero and only the message rate column is meaningful.
 */
int main(int argc, char** argv) {
  args_t args;
  parse_args(argc, argv, &args);

  gin_test_caps_t caps = {};
  caps.default_rsm = GIN_RSM_GPU;
  caps.allow_thread_rsm = false;
  caps.op_mask = 0;
  caps.default_op = GIN_OP_UNSET;
  caps.is_signal_op = true;
  caps.allow_aggregate_requests = true;
  caps.allow_bidirectional = true;
  caps.allow_multi_cta_threads = true;
  configure_test_args(&args, &caps);

  gin_benchmark_t bench = {};
  bench.run = ginSignalBW_launch;
  bench.name = "GIN SIGNAL MESSAGE RATE";
  bench.type = GIN_BENCHMARK_TYPE_THROUGHPUT;
  bench.payloadMode = GIN_THROUGHPUT_PAYLOAD_NONE;

  gin_perf_run(argc, argv, &args, &bench);
  return 0;
}
