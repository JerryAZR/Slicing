SRCDIR=cuda
OBJDIR=obj
OUTDIR=out
CXX=nvcc

DEPS = $(SRCDIR)/slicer.cuh $(SRCDIR)/triangle.cuh $(SRCDIR)/golden.cuh
NEW_DEPS = AlgoGPU/slicer.cu AlgoGPU/triangle.cu AlgoGPU/slicer.cuh AlgoGPU/triangle.cuh

all: pps bbox bbox-large pps-large bbox-rle
main: pps-large-main bbox-large-main bbox-rle-main
test: pps-test bbox-test pps-large-test bbox-large-test bbox-rle-test
rle: bbox-rle pps-rle

fps: fps-main fps-test
pps: pps-main pps-test
pps-large: pps-large-main pps-large-test
opps: opps-main opps-test
new: new-main new-test
second: second-main second-test
mfps: mfps-main mfps-test
ofps: ofps-main ofps-test
bbox: bbox-main bbox-test
bbox-large: bbox-large-main bbox-large-test
bbox-rle: bbox-rle-main bbox-rle-test
pps-rle: pps-rle-main pps-rle-test

new-main: AlgoGPU/new-main.cu $(NEW_DEPS)
	mkdir -p $(OUTDIR)
	nvcc AlgoGPU/new-main.cu AlgoGPU/slicer.cu AlgoGPU/triangle.cu -o $(OUTDIR)/$@

new-test: AlgoGPU/new-test.cu $(NEW_DEPS) AlgoGPU/golden.cu AlgoGPU/golden.cuh
	mkdir -p $(OUTDIR)
	nvcc AlgoGPU/new-test.cu AlgoGPU/slicer.cu AlgoGPU/triangle.cu AlgoGPU/golden.cu -o $(OUTDIR)/$@

%-main: $(OBJDIR)/triangle.o $(OBJDIR)/%-slicer.o $(OBJDIR)/%.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/$@

%-test: $(OBJDIR)/triangle.o $(OBJDIR)/%-slicer.o $(OBJDIR)/golden.o $(OBJDIR)/%-test.o
	mkdir -p $(OUTDIR)
	$(CXX) $^ -o $(OUTDIR)/$@

$(OBJDIR)/triangle.o: $(SRCDIR)/triangle.cu $(SRCDIR)/triangle.cuh
	$(CXX) -c -o $@ $<

.PRECIOUS: $(OBJDIR)/%-test.o
$(OBJDIR)/%-test.o: $(SRCDIR)/%.cu $(DEPS)
	$(CXX) -c -DTEST -o $@ $<

.PRECIOUS: $(OBJDIR)/%.o
$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	$(CXX) -c -o $@ $<

# .PHONY: all
%:: %-main %-test

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(OUTDIR)/*
