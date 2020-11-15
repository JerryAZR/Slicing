SRCDIR=cuda
OBJDIR=obj
OUTDIR=out
CXX=nvcc

DEPS = $(SRCDIR)/slicer.cuh $(SRCDIR)/triangle.cuh $(SRCDIR)/golden.cuh

all: fps

new:
	$(CXX) AlgoGPU/*.cu -o $(OUTDIR)/newAlgo

fps: $(OBJDIR)/triangle.o $(OBJDIR)/slicer.o $(OBJDIR)/fps-main.o
	$(CXX) $(OBJDIR)/fps-main.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o -o $(OUTDIR)/fps

pps: $(OBJDIR)/triangle.o $(OBJDIR)/slicer.o $(OBJDIR)/pps-main.o
	$(CXX) $(OBJDIR)/pps-main.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o -o $(OUTDIR)/pps

fps-test: $(OBJDIR)/triangle.o $(OBJDIR)/slicer.o $(OBJDIR)/golden.o $(OBJDIR)/fps-test.o
	$(CXX) $(OBJDIR)/fps-test.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o $(OBJDIR)/golden.o -o $(OUTDIR)/fps-test

pps-test: $(OBJDIR)/triangle.o $(OBJDIR)/slicer.o $(OBJDIR)/golden.o $(OBJDIR)/pps-test.o
	$(CXX) $(OBJDIR)/pps-test.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o $(OBJDIR)/golden.o -o $(OUTDIR)/pps-test

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	$(CXX) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(OUTDIR)/*
