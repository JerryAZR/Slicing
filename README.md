# Slicing

The cuda codes are in [cuda](./cuda/) and [AlgoGPU](./AlgoGPU/).

To build, type:

``make [fps|pps|new|fps-test|pps-test]``

The output binary will be in [out](./out/).

To profile or test the program, type

``./run.py [options]``

To get detailed profiling results, run

``sudo nvprof --events [comma separated list of events] --metrics [comma separated list of metrics] [executable] [path_to_stl_file] 2> [output file]``

A list of available events and metrics can be found in [events.txt](./performance/events.txt) and [metrics.txt](./performance/metrics.txt)
