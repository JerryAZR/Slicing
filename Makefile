SRCDIR=cuda
OBJDIR=obj
OUTDIR=out
CXX=nvcc

DEPS = $(SRCDIR)/slicer.cuh $(SRCDIR)/triangle.cuh $(SRCDIR)/golden.cuh
NEW_DEPS = AlgoGPU/slicer.cu AlgoGPU/triangle.cu AlgoGPU/slicer.cuh AlgoGPU/triangle.cuh

all: fps

second: $(SRCDIR)/triangle.cu $(SRCDIR)/sort.cu $(SRCDIR)/second-test.cu
	$(CXX) $(SRCDIR)/triangle.cu $(SRCDIR)/sort.cu $(SRCDIR)/second-test.cu -o second

new: AlgoGPU/new-main.cu $(NEW_DEPS)
	mkdir -p $(OUTDIR)
	nvcc AlgoGPU/new-main.cu AlgoGPU/slicer.cu AlgoGPU/triangle.cu -o $(OUTDIR)/new

fps: $(OBJDIR)/triangle.o $(OBJDIR)/fps-slicer.o $(OBJDIR)/fps-main.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/fps

pps: $(OBJDIR)/triangle.o $(OBJDIR)/pps-slicer.o $(OBJDIR)/pps-main.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/pps

new-test: AlgoGPU/new-test.cu $(NEW_DEPS) AlgoGPU/golden.cu AlgoGPU/golden.cuh
	mkdir -p $(OUTDIR)
	nvcc AlgoGPU/new-test.cu AlgoGPU/slicer.cu AlgoGPU/triangle.cu AlgoGPU/golden.cu -o $(OUTDIR)/new-test
	
fps-test: $(OBJDIR)/triangle.o $(OBJDIR)/fps-slicer.o $(OBJDIR)/golden.o $(OBJDIR)/fps-test.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/fps-test

pps-test: $(OBJDIR)/triangle.o $(OBJDIR)/pps-slicer.o $(OBJDIR)/golden.o $(OBJDIR)/pps-test.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/pps-test

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	$(CXX) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(OUTDIR)/*
