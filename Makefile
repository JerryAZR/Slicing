SRCDIR=cuda
OBJDIR=obj
OUTDIR=out
CXX=nvcc

DEPS = $(SRCDIR)/slicer.cuh $(SRCDIR)/triangle.cuh $(SRCDIR)/golden.cuh
NEW_DEPS = AlgoGPU/slicer.cu AlgoGPU/triangle.cu AlgoGPU/slicer.cuh AlgoGPU/triangle.cuh

all: fps pps new second mfps

fps: fps-main fps-test
pps: pps-main pps-test
new: new-main new-test
second: second-main second-test
mfps: mfps-main mfps-test

second-main: $(OBJDIR)/triangle.o $(OBJDIR)/second-slicer.o $(OBJDIR)/second-main.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/second-main

second-test: $(OBJDIR)/triangle.o $(OBJDIR)/second-slicer.o $(OBJDIR)/second-test.o $(OBJDIR)/golden.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/second-test

new-main: AlgoGPU/new-main.cu $(NEW_DEPS)
	mkdir -p $(OUTDIR)
	nvcc AlgoGPU/new-main.cu AlgoGPU/slicer.cu AlgoGPU/triangle.cu -o $(OUTDIR)/new

fps-main: $(OBJDIR)/triangle.o $(OBJDIR)/fps-slicer.o $(OBJDIR)/fps-main.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/fps

pps-main: $(OBJDIR)/triangle.o $(OBJDIR)/pps-slicer.o $(OBJDIR)/pps-main.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/pps

mfps-main: $(OBJDIR)/triangle.o $(OBJDIR)/mfps-slicer.o $(OBJDIR)/mfps-main.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/mfps

new-test: AlgoGPU/new-test.cu $(NEW_DEPS) AlgoGPU/golden.cu AlgoGPU/golden.cuh
	mkdir -p $(OUTDIR)
	nvcc AlgoGPU/new-test.cu AlgoGPU/slicer.cu AlgoGPU/triangle.cu AlgoGPU/golden.cu -o $(OUTDIR)/new-test
	
fps-test: $(OBJDIR)/triangle.o $(OBJDIR)/fps-slicer.o $(OBJDIR)/golden.o $(OBJDIR)/fps-test.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/fps-test

mfps-test: $(OBJDIR)/triangle.o $(OBJDIR)/mfps-slicer.o $(OBJDIR)/golden.o $(OBJDIR)/mfps-test.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/mfps-test

pps-test: $(OBJDIR)/triangle.o $(OBJDIR)/pps-slicer.o $(OBJDIR)/golden.o $(OBJDIR)/pps-test.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/pps-test

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	$(CXX) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(OUTDIR)/*
