ODIR=./obj


fps: fps-main.o slicer.o triangle.o
	nvcc fps-main.cu slicer.cu triangle.cu -o fps

pps: pps-main.o slicer.o triangle.o
	nvcc pps-main.cu slicer.cu triangle.cu -o pps

fps-test: fps-test.o slicer.o triangle.o golden.o
	nvcc fps-test.cu slicer.cu triangle.cu golden.cu -o fps-test

pps-test: pps-test.o slicer.o triangle.o golden.o
	nvcc pps-test.o slicer.o triangle.o golden.o -o pps-test

pps-main.o: pps-main.cu slicer.cuh triangle.cuh
	nvcc -c pps-main.cu -o pps-main.o

fps-main.o: fps-main.cu slicer.cuh triangle.cuh
	nvcc -c fps-main.cu -o fps-main.o

pps-test.o: pps-test.cu slicer.cuh triangle.cuh golden.cuh
	nvcc -c pps-test.cu -o pps-test.o

fps-test.o: fps-test.cu slicer.cuh triangle.cuh golden.cuh
	nvcc -c fps-test.cu -o fps-test.o

golden.o: golden.cu golden.cuh slicer.cuh triangle.cuh
	nvcc -c golden.cu -o golden.o

slicer.o: slicer.cu slicer.cuh triangle.cuh
	nvcc -c slicer.cu -o slicer.o

triangle.o: triangle.cu triangle.cuh
	nvcc -c triangle.cu -o triangle.o
