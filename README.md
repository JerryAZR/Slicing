# Slicing

The cuda codes are in [cuda](./cuda/) and [AlgoGPU](./AlgoGPU/).

To build, type:

``make [fps|pps|new|fps-test|pps-test]``

The output binary will be in [out](./out/).

To get a summary for kernel/API execution time, run

``sudo nvprof [executable] [path_to_stl_file] 2> [output file]``

To get detailed profiling results, run

``sudo nvprof --events [comma separated list of events, or "all"] --metrics [comma separated list of metrics, or "all"] [executable] [path_to_stl_file] 2> [output file]``
