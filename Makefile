CPP     = g++  -std=c++11 -Wall --pedantic
NVCC    = /usr/local/cuda-9.0/bin/nvcc -std=c++11
INCDIR  = -I./modules/build/include -I./ -I./util -I/usr/local/cuda-9.0/include
LIBDIR  = -L./modules/build/lib -L./ -L/usr/local/cuda-9.0/lib64
NVLIBS  = -lcuda -lcudart -lglad -lGL -lEGL -lXrandr -lXext -lX11 -lrt -ldl -lglut -lpthread
CFLAGS  = $(INCDIR) $(LIBDIR)
NVFLAGS = $(INCDIR) $(LIBDIR) -arch=sm_30

all: main.out

main.out: helper.o simulation.o util.o
	@$(CPP) $(CFLAGS) planet.hpp main.cpp -o main.out *.o $(NVLIBS)

simulation.o:
	@$(NVCC) $(NVFLAGS) -c simulation.cu -o simulation.o

helper.o:
	@$(CPP) $(CFLAGS) -c helper.c -o helper.o

util.o:
	@$(CPP) $(CFLAGS) -c util/util.cpp -o util.o

clean:
	@$(RM) -rf *.o *.out *.a

rebuild: clean all
