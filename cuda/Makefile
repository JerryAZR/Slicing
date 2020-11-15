OBJDIR=./obj
SRCDIR=./src
OUTDIR=.
CXX=nvcc

all: fps

fps: $(OBJDIR)/fps-main.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o
	$(CXX) $(OBJDIR)/fps-main.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o -o $(OUTDIR)/fps

pps: $(OBJDIR)/pps-main.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o
	$(CXX) $(OBJDIR)/pps-main.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o -o $(OUTDIR)/pps

fps-test: $(OBJDIR)/fps-test.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o $(OBJDIR)/golden.o
	$(CXX) fps-test.cu slicer.cu triangle.cu golden.cu -o $(OUTDIR)/fps-test

pps-test: $(OBJDIR)/pps-test.o $(OBJDIR)/slicer.o $(OBJDIR)/triangle.o $(OBJDIR)/golden.o
	$(CXX) pps-test.o slicer.o triangle.o golden.o -o $(OUTDIR)/pps-test

$(OBJDIR)/pps-main.o: pps-main.cu slicer.cuh triangle.cuh
	$(CXX) -c pps-main.cu -o $(OBJDIR)/pps-main.o

$(OBJDIR)/fps-main.o: fps-main.cu slicer.cuh triangle.cuh
	$(CXX) -c fps-main.cu -o $(OBJDIR)/fps-main.o

$(OBJDIR)/pps-test.o: pps-test.cu slicer.cuh triangle.cuh golden.cuh
	$(CXX) -c pps-test.cu -o $(OBJDIR)/pps-test.o

$(OBJDIR)/fps-test.o: fps-test.cu slicer.cuh triangle.cuh golden.cuh
	$(CXX) -c fps-test.cu -o $(OBJDIR)/fps-test.o

$(OBJDIR)/golden.o: golden.cu golden.cuh slicer.cuh triangle.cuh
	$(CXX) -c golden.cu -o $(OBJDIR)/golden.o

$(OBJDIR)/slicer.o: slicer.cu slicer.cuh triangle.cuh
	$(CXX) -c slicer.cu -o $(OBJDIR)/slicer.o

$(OBJDIR)/triangle.o: triangle.cu triangle.cuh
	$(CXX) -c triangle.cu -o $(OBJDIR)/triangle.o

.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o
