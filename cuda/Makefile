
fps: fps-main.cu slicer.cu slicer.cuh triangle.cu triangle.cuh
	nvcc fps-main.cu slicer.cu triangle.cu -o fps

pps: pps-main.cu slicer.cu slicer.cuh triangle.cu triangle.cuh
	nvcc pps-main.cu slicer.cu triangle.cu -o pps

fps-test: fps-test.cu slicer.cu slicer.cuh triangle.cu triangle.cuh golden.cu golden.cuh
	nvcc fps-test.cu slicer.cu triangle.cu golden.cu -o fps-test

pps-test: fps-test.cu slicer.cu slicer.cuh triangle.cu triangle.cuh golden.cu golden.cuh
	nvcc pps-test.cu slicer.cu triangle.cu golden.cu -o pps-test
