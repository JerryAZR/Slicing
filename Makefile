SRCDIR=cuda
OBJDIR=obj
OUTDIR=out
CXX=nvcc

DEPS = $(SRCDIR)/slicer.cuh $(SRCDIR)/triangle.cuh $(SRCDIR)/golden.cuh

all: fps

new:
	mkdir -p $(OUTDIR)
	nvcc AlgoGPU/new-main.cu AlgoGPU/slicer.cu AlgoGPU/triangle.cu -o $(OUTDIR)/new

fps: $(OBJDIR)/triangle.o $(OBJDIR)/slicer.o $(OBJDIR)/fps-main.o
	mkdir -p $(OUTDIR)
	$(CXX) $(OBJDIR)/fps-main.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o -o $(OUTDIR)/fps

pps: $(OBJDIR)/triangle.o $(OBJDIR)/slicer.o $(OBJDIR)/pps-main.o
	mkdir -p $(OUTDIR)
	$(CXX) $(OBJDIR)/pps-main.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o -o $(OUTDIR)/pps

new-test: 
	mkdir -p $(OUTDIR)
	nvcc AlgoGPU/new-test.cu AlgoGPU/slicer.cu AlgoGPU/triangle.cu AlgoGPU/golden.cu -o $(OUTDIR)/new-test
	
fps-test: $(OBJDIR)/triangle.o $(OBJDIR)/slicer.o $(OBJDIR)/golden.o $(OBJDIR)/fps-test.o
	mkdir -p $(OUTDIR)
	$(CXX) $(OBJDIR)/fps-test.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o $(OBJDIR)/golden.o -o $(OUTDIR)/fps-test

pps-test: $(OBJDIR)/triangle.o $(OBJDIR)/slicer.o $(OBJDIR)/golden.o $(OBJDIR)/pps-test.o
	mkdir -p $(OUTDIR)
	$(CXX) $(OBJDIR)/pps-test.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o $(OBJDIR)/golden.o -o $(OUTDIR)/pps-test

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	$(CXX) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(OUTDIR)/*
