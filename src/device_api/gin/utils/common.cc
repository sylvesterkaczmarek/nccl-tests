#include "common.h"
#include "gin_context.h"


/*
 * A binary can cover several operations (the put benchmarks select between put,
 * put+signal and put+counter via --gin_op), so name the resolved configuration
 * before the table.
 */
static void print_configuration(const args_t* args, const gin_benchmark_t* bench) {
  printf("# %s: rsm=%s", bench->name ? bench->name : "gin benchmark", gin_rsm_name(args->gin_rsm));
  if (args->gin_strong_signal) printf(", strong_signal");
  if (args->gin_skip_credit_check) printf(", skip_credit_check");
  if (args->gin_aggregate_requests) printf(", aggregate_requests");
  if (args->gin_bidirectional) printf(", bidirectional");
  if (bench->type == GIN_BENCHMARK_TYPE_THROUGHPUT) {
    printf(", ctas=%d, threads=%d", args->num_ctas, args->num_threads);
  }
  printf("\n");
}

void print_header(const gin_benchmark_t* bench) {
  if (bench->type == GIN_BENCHMARK_TYPE_THROUGHPUT) {
    printf("%12s  %14s  %18s  %20s\n", "Size(B)", "Num_Messages", "Bandwidth(MiB/s)",
           "Message_Rate(MPPS)");
  } else {
    printf("%12s  %8s  %14s\n", "Size(B)", "Iters", "Latency(us)");
  }
}

static void print_throughput_information(size_t size, double num_messages, double bandwidth_MiBps,
                                         double message_rate_MPPS) {
  printf("%12zu  %14.0f  %18.3f  %20.3f\n", size, num_messages, bandwidth_MiBps, message_rate_MPPS);
}

static void print_latency_information(size_t size, int iters, double latency_us){
  printf("%12zu  %8d  %14.3f\n", size, iters, latency_us);
}

static uint64_t getHostHash(const char* string) {
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  memset(hostname, 0, maxlen);
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen && hostname[i] != '\0'; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

static size_t next_size(const args_t* args, size_t size) {
  if (args->stepbytes != 0) return size + args->stepbytes;
  size_t factor = args->stepfactor;
  if (factor <= 1) factor = 2;
  return size * factor;
}

static double throughput_bytes_per_message(const gin_benchmark_t* bench, size_t size) {
  switch (bench->payloadMode) {
  case GIN_THROUGHPUT_PAYLOAD_NONE:
    return 0.0;
  case GIN_THROUGHPUT_PAYLOAD_FIXED:
    return (double)bench->payloadFixedBytes;
  case GIN_THROUGHPUT_PAYLOAD_SIZE:
  default:
    return (double)size;
  }
}

void gin_perf_run(int argc, char** argv, const args_t* args, const gin_benchmark_t* bench) {

  int rank = 0, nRanks = 0;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK_FATAL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK_FATAL(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  if (nRanks != 2) {
    if (rank == 0) {
      fprintf(stderr, "Error: GIN perf benchmarks require exactly 2 MPI ranks (got %d)\n", nRanks);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  uint64_t* hosts = new uint64_t[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hosts[rank] = getHostHash(hostname);
  MPICHECK_FATAL(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                               hosts, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  int dev = 0;
  for (int r = 0; r < rank; r++) {
    if (hosts[r] == hosts[rank]) dev++;
  }
  delete[] hosts;

  CUDACHECK_FATAL(cudaSetDevice(dev));

  ncclUniqueId id;
  if (rank == 0) NCCLCHECK_FATAL(ncclGetUniqueId(&id));
  MPICHECK_FATAL(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  cudaStream_t stream;
  CUDACHECK_FATAL(cudaStreamCreate(&stream));

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 1;
  ncclComm_t comm;
  NCCLCHECK_FATAL(ncclCommInitRankConfig(&comm, nRanks, id, rank, &config));

  gin_context_t ctx;
  gin_devComm_create(comm, args, &ctx);

  const bool bidir = args->gin_bidirectional;
  const bool participates = (bench->type == GIN_BENCHMARK_TYPE_PING_PONG || bidir) || rank == 0;
  const bool measures = (rank == 0) || bidir;

  cudaEvent_t start, stop;
  if (measures) {
    CUDACHECK_FATAL(cudaEventCreate(&start));
    CUDACHECK_FATAL(cudaEventCreate(&stop));
  }

  if (rank == 0) {
    print_configuration(args, bench);
    print_header(bench);
  }

  for (size_t size = args->minbytes; size <= args->maxbytes;) {
    size_t numElems = size / sizeof(int);

    //-----Warmup-----
    if (args->warmup_iters > 0 && participates) {
      bench->run(ctx.dcomm, ctx.hBuf, stream, args, numElems, args->warmup_iters);
      CUDACHECK_FATAL(cudaStreamSynchronize(stream));
    }

    //-----Benchmark-----
    MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));

    if (measures) CUDACHECK_FATAL(cudaEventRecord(start, stream));
    if (participates) {
      bench->run(ctx.dcomm, ctx.hBuf, stream, args, numElems, args->iters);
    }
    if (measures) CUDACHECK_FATAL(cudaEventRecord(stop, stream));
    if (participates) {
      CUDACHECK_FATAL(cudaStreamSynchronize(stream));
    }

    if (bench->type == GIN_BENCHMARK_TYPE_PING || bench->type == GIN_BENCHMARK_TYPE_PING_PONG) {
      // Latency: rank 0 reports (bidirectional is not supported for latency).
      if (rank == 0) {
        float milliseconds = 0.0f;
        CUDACHECK_FATAL(cudaEventElapsedTime(&milliseconds, start, stop));
        double RTT_us = (double)milliseconds * 1000.0 / (double)args->iters;
        double latency_us = RTT_us / 2.0;
        print_latency_information(size, args->iters, latency_us);
      }
    } else {
      double num_messages = 0.0, bandwidth_MiBps = 0.0, message_rate_MPPS = 0.0;
      if (measures) {
        float milliseconds = 0.0f;
        CUDACHECK_FATAL(cudaEventElapsedTime(&milliseconds, start, stop));
        double seconds = (double)milliseconds / 1e3;
        num_messages = (double)args->iters * (double)args->num_ctas * (double)args->num_threads;
        double bytes = num_messages * throughput_bytes_per_message(bench, size);
        bandwidth_MiBps = bytes / (double)(1ULL << 20) / seconds;
        message_rate_MPPS = num_messages / 1e6 / seconds;
      }

      if (bidir) {
        double local_metrics[3] = {num_messages, bandwidth_MiBps, message_rate_MPPS};
        double summed_metrics[3] = {0.0, 0.0, 0.0};
        MPICHECK_FATAL(MPI_Reduce(local_metrics, summed_metrics, 3, MPI_DOUBLE, MPI_SUM, 0,
                                  MPI_COMM_WORLD));
        num_messages = summed_metrics[0];
        bandwidth_MiBps = summed_metrics[1];
        message_rate_MPPS = summed_metrics[2];
      }

      if (rank == 0) {
        print_throughput_information(size, num_messages, bandwidth_MiBps, message_rate_MPPS);
      }
    }

    MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));

    size_t next = next_size(args, size);
    if (next <= size) break;
    size = next;
  }

  MPICHECK_FATAL(MPI_Barrier(MPI_COMM_WORLD));
  gin_devComm_destroy(comm, &ctx);
  NCCLCHECK_FATAL(ncclCommDestroy(comm));
  if (measures) {
    CUDACHECK_FATAL(cudaEventDestroy(start));
    CUDACHECK_FATAL(cudaEventDestroy(stop));
  }
  CUDACHECK_FATAL(cudaStreamDestroy(stream));
  MPICHECK_FATAL(MPI_Finalize());
}
