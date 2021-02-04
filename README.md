# Slicing

The cuda codes are in [cuda](./cuda/) and [AlgoGPU](./AlgoGPU/).

To build, type:

``make [fps|pps|new|second|mfps|bbox]``

| Algorithm |      Details                                         |
|:----------|:-----------------------------------------------------|
| pps       | Pixelwise parallel slicing                           |
| fps       | Fully parallel slicing                               |
| new       | Map-based triangle sorting and slicing               |
| second    | Processing large and small triangles separately ([details](./docs/second.txt))     |
| mfps      | Modified fps. Each thread handles multiple triangles |
| bbox      | Bounding-box based slicing algorithm. ([details](./docs/bbox.txt))    |
| ofps/opps | Optimized fps/pps implementation                     |

The output binary will be in [out](./out/).

To profile or test the program, type

``./run.py [options]``

To get detailed profiling results of selected events or metrics, run

``sudo nvprof --events [comma separated list of events] --metrics [comma separated list of metrics] [executable] [path_to_stl_file] 2> [output file]``

A list of available events and metrics can be found in [events.txt](./performance/events.txt) and [metrics.txt](./performance/metrics.txt)
