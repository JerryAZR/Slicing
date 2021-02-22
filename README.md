# Slicing

A GPU based slicing algorithm for continuous 3D printing.

The cuda codes are in [cuda](./cuda/).

Some old algorithms have been moved to [old](./cuda/old/)

To build, type:

``make [Name_of_Algo]``

Note that only ``pps, bbox, pps-large, bbox-large, bbox-rle`` are currently available

| Algorithm |      Details                                         |
|:----------|:-----------------------------------------------------|
| pps       | Pixelwise parallel slicing                           |
| fps       | Fully parallel slicing                               |
| new       | Map-based triangle sorting and slicing               |
| second    | Processing large and small triangles separately ([details](./docs/second.txt))     |
| mfps      | Modified fps. Each thread handles multiple triangles |
| bbox      | Bounding-box based slicing algorithm. ([details](./docs/bbox.txt))    |
| ofps/opps | Optimized fps/pps implementation                     |
| *-large   | The output is generated several layers at a time     |

The output binary will be in [out](./out/).

To profile or test the program, type

``./run.py [options]``

To get detailed profiling results of selected events or metrics, run

``sudo nvprof --events [comma separated list of events] --metrics [comma separated list of metrics] [executable] [path_to_stl_file] 2> [output file]``

A list of available events and metrics can be found in [events.txt](./performance/events.txt) and [metrics.txt](./performance/metrics.txt)
