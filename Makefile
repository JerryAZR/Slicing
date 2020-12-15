SRCDIR=cuda
OBJDIR=obj
OUTDIR=out
CXX=nvcc

DEPS = $(SRCDIR)/slicer.cuh $(SRCDIR)/triangle.cuh $(SRCDIR)/golden.cuh
NEW_DEPS = AlgoGPU/slicer.cu AlgoGPU/triangle.cu AlgoGPU/slicer.cuh AlgoGPU/triangle.cuh

all: fps pps second mfps ofps

fps: fps-main fps-test
pps: pps-main pps-test
new: new-main new-test
second: second-main second-test
mfps: mfps-main mfps-test
ofps: ofps-main ofps-test

new-main: AlgoGPU/new-main.cu $(NEW_DEPS)
	mkdir -p $(OUTDIR)
	nvcc AlgoGPU/new-main.cu AlgoGPU/slicer.cu AlgoGPU/triangle.cu -o $(OUTDIR)/$@

ofps-main: $(OBJDIR)/triangle.o $(OBJDIR)/ofps-slicer.o $(OBJDIR)/fps-main.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/$@

new-test: AlgoGPU/new-test.cu $(NEW_DEPS) AlgoGPU/golden.cu AlgoGPU/golden.cuh
	mkdir -p $(OUTDIR)
	nvcc AlgoGPU/new-test.cu AlgoGPU/slicer.cu AlgoGPU/triangle.cu AlgoGPU/golden.cu -o $(OUTDIR)/$@

ofps-test: $(OBJDIR)/triangle.o $(OBJDIR)/ofps-slicer.o $(OBJDIR)/golden.o $(OBJDIR)/fps-test.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/$@

opps-test: $(OBJDIR)/triangle.o $(OBJDIR)/opps-slicer.o $(OBJDIR)/golden.o $(OBJDIR)/pps-test.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/$@

%-main: $(OBJDIR)/triangle.o $(OBJDIR)/%-slicer.o $(OBJDIR)/%-main.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/$@

%-test: $(OBJDIR)/triangle.o $(OBJDIR)/%-slicer.o $(OBJDIR)/golden.o $(OBJDIR)/%-test.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/$@

$(OBJDIR)/triangle.o: $(SRCDIR)/triangle.cu $(SRCDIR)/triangle.cuh
	$(CXX) -c -o $@ $<

.PRECIOUS: $(OBJDIR)/%.o
$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	$(CXX) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(OUTDIR)/*
