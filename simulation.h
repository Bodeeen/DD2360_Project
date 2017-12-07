#ifndef SIMULATION_H
#define SIMULATION_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include "common.hpp"


double getElapsed(struct timeval t0, struct timeval t1);

bool operator!=(const float3 &a, const float3 &b);


__host__ __device__ float3 operator+(const float3 &a, const float3 &b);

__host__ __device__ float3 operator*(const float &a, const float3 &b);

__host__ __device__ void update_particle(Particle *p, Particle *particles, float dt, int num_particles);

__global__ void simulation_GPU(Particle *particles, int n, float dt);

class Simulation
{
public:
        virtual void init() = 0;
        virtual void update() = 0;
        virtual void display() = 0;
        virtual void release() = 0;
};

class GPUSimulation : public Simulation
{
private:
        float dt;
        int NUM_PARTICLES;
        int NUM_ITERATIONS;
        int BLOCK_SIZE;
        Particle        *particles;
        float3          *fake_random;
        Particle        *particles_GPU;
        Particle        *d_p;
        float3          *d_r;
        double gpu_calculating_time;
        struct timeval t0, t1, t2, t3, t5, t6;
        int count;
public:

        GPUSimulation(int n, int it, int size);
        // virtual void init(int argc, char const *argv[])
        virtual void init();

        virtual void update();

        virtual void display();

        virtual void release();

        // void preset_values(int n, int it, int size);
        // void initial_particles();

        /*Sergio Rivas Gómez*/
        void checkCUDAError();

        // void update_random(int it);

};

class CPUSimulation : public Simulation
{
private:
        float dt;
        int NUM_PARTICLES;
        int NUM_ITERATIONS;
        int BLOCK_SIZE;
        Particle        *particles;
        float3          *fake_random;

        struct timeval t0, t1;
        int count;

public:
        CPUSimulation(int n, int it, int size);

        virtual void init();

        virtual void update();

        virtual void display();

        virtual void release();

        void simulation_CPU();

        // void update_random(int it);
};

void initial_particles(Particle *particles, int n);

void update_random(int it, float3 *fake_random, int n);

void print_particle(Particle *particles, int c);

#endif
