# Slicing

The cuda codes are in [cuda](./cuda/) and [AlgoGPU](./AlgoGPU/).

In any of the folders, use the following command to compile:

``nvcc *.cu -o <executable>``

To get a summary for kernel/API execution time, run

``sudo nvprof [path_to_stl_file] 2> [output file]``

To get detailed profiling results, run

``sudo nvprof --events [comma separated list of events, or "all"] --metrics [comma separated list of metrics, or "all"] [path_to_stl_file] 2> [output file]``

There is a function in [cuda/golden.cuh](./cuda/golden.cuh) that can be used for testing. The function

``checkOutput(triangle* triangles_dev, size_t num_triangles, bool* in)`` 

compares the actual output (bool* in) to the expected output, and returns the number of pixels that are different.
(More work to be done on this part to make testing easier)
