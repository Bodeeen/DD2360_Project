CPP     = g++  -std=c++11 -Wall --pedantic
NVCC    = nvcc -std=c++11
INCDIR  = -I./ -I/usr/local/cuda/include -I/pdc/vol/cuda/cuda-8.0/include
LIBDIR  = -L./ -L/usr/local/cuda/lib64 -L/pdc/vol/cuda/cuda-8.0/lib64
NVLIBS  = -lcuda -lcudart
CFLAGS  = $(INCDIR) $(LIBDIR)
NVFLAGS = $(INCDIR) $(LIBDIR) -arch=sm_30

all: main.out

main.out: helper.o simulation.o
        @$(CPP) $(CFLAGS) planet.hpp main.cpp -o main.out *.o $(NVLIBS)

simulation.o:
        @$(NVCC) $(NVFLAGS) -c simulation.cu -o simulation.o

helper.o:
        @$(CPP) $(CFLAGS) -c helper.c -o helper.o

clean:
        @$(RM) -rf *.o *.out

rebuild: clean all
