#include "simulation.h"


double getElapsed(struct timeval t0, struct timeval t1)// millisecond
{
    return ((t1.tv_sec-t0.tv_sec) * 1000000LL + (t1.tv_usec-t0.tv_usec)) / 1000.0;
}

bool operator!=(const float3 &a, const float3 &b)
{
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}


__host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__host__ __device__ float3 operator*(const float &a, const float3 &b)
{
  return make_float3(a*b.x, a*b.y, a*b.z);
}

__host__ __device__ bool isIncreasing(Particle p1, Particle p2)
{
  return (glm::dot((p2.velocity - p1.velocity), (p2.position - p1.position)) > 0);
}

__host__ __device__ float calculate_force(Particle p1, Particle p2)
{
  float r = glm::distance(p1.position, p2.position);

  if (r < eps){
    r = eps;
  }
  if (r<Core_D_Fe) {
    if (isIncreasing(p1, p2)) {
      return GMM/(r*r)-(K_Si*KRP_Si+K_Fe*KRP_Fe)*(D*D-r*r);
    } else {
      return GMM/(r*r)-(K_Si+K_Fe)*(D*D-r*r);
    }
  } else if (r < Core_D_Si) {
    if (isIncreasing(p1, p2)) {
      return GMM/(r*r)-(K_Si*KRP_Si+K_Fe)*(D*D-r*r);
    } else {
      return GMM/(r*r)-(K_Si+K_Fe)*(D*D-r*r);
    }
  } else if(r < D){
    return GMM/(r*r)-(K_Si+K_Fe)*(D*D-r*r);
  } else{
    return GMM/(r*r);
  }
}

__host__ __device__ void update_particle(Particle *p, Particle *particles, float dt, int num_particles)
{

  glm::vec3 f = glm::vec3(0.0);
  glm::vec3 a = glm::vec3(0.0);

  for(int i = 0; i<num_particles; i++)
  {
    f += glm::normalize(particles[i].position - p->position) * calculate_force(*p, particles[i]);
  }

  if(p->material)
    a = (1/(float)M_Fe)*f;
  else
    a = (1/(float)M_Si)*f;

  p->velocity += a*dt;
  p->position += dt * p->velocity;
  //return p.position + dt * p.velocity;
}

__global__ void simulation_GPU(Particle *particles, int n, float dt)
{
        const int i = blockIdx.x*blockDim.x + threadIdx.x;

        if(i<n){
          update_particle(&particles[i], particles, dt, n);
        }
}

GPUSimulation::GPUSimulation(int n, int it, int size)
{
        NUM_PARTICLES = n;
        NUM_ITERATIONS = it;
        BLOCK_SIZE = size;
        dt = 1.f;
        gpu_calculating_time =0.;
        count =0;
        printf("NUM_PARTICLES: %d, NUM_ITERATIONS: %d, BLOCK_SIZE: %d, dt: %.5f\n", NUM_PARTICLES, NUM_ITERATIONS, BLOCK_SIZE, dt);
}

void GPUSimulation::init()
{
  particles       = (Particle*)   malloc(sizeof(Particle) * NUM_PARTICLES);
	//fake_random     = (float3*)     malloc(sizeof(float3) * NUM_PARTICLES);
	particles_GPU   = (Particle*)   malloc(sizeof(Particle) * NUM_PARTICLES);//GPU result

  initial_particles(particles, NUM_PARTICLES);

  cudaMalloc( &d_p, NUM_PARTICLES * sizeof(Particle));
  //cudaMalloc( &d_r, NUM_PARTICLES * sizeof(float3) );

  printf("Start computing particles on the GPU... ");
  gettimeofday(&t0, NULL);

  cudaMemcpy( d_p, particles, NUM_PARTICLES * sizeof(Particle) , cudaMemcpyHostToDevice );
  checkCUDAError();
}

void GPUSimulation::update()
{
  //update_random(count, fake_random, NUM_PARTICLES);

  //cudaMemcpy( d_r, fake_random, NUM_PARTICLES * sizeof(float3) , cudaMemcpyHostToDevice );
  //checkCUDAError();

  gettimeofday(&t2, NULL);
  simulation_GPU<<<(NUM_PARTICLES + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_p, NUM_PARTICLES, dt);
  gettimeofday(&t3, NULL);

  gpu_calculating_time += getElapsed(t2,t3);
  count++;
}

void GPUSimulation::display()
{
  cudaMemcpy( particles_GPU, d_p, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost );
  checkCUDAError();

  gettimeofday(&t1, NULL);

  printf("Done\n");
  printf("Time\ncalculating: %lfms \t all: %lfms\n\n", gpu_calculating_time, getElapsed(t0, t1));
  //print_particle(particles_GPU, 5);
}

void GPUSimulation::release()
{
  cudaFree( d_p );
  //cudaFree( d_r );

	free(particles);
	//free(fake_random);
	free(particles_GPU);
}

/*Sergio Rivas Gómez*/
void GPUSimulation::checkCUDAError()
{
    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        printf("CUDA Error: Returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(-1);
    }
}

// void MockupSimulation::update_random(int ithttps://github.com/Bodeeen/DD2360_Project.git)
// {
//   float t;
//   for (int i = 0; i < NUM_PARTICLES; ++i)
//   {
//     t = it * 0.01 + 0.1*i;
//     fake_random[i]  = make_float3(t, t, t);
//   }
// }
// void MockupSimulation::preset_values(int n, int it, int size)
// {
// 	NUM_PARTICLES = n;
// 	NUM_ITERATIONS = it;
// 	BLOCK_SIZE = size;
// 	dt = 1.f;https://github.com/Bodeeen/DD2360_Project.git
// 	printf("NUM_PARTICLES: %d, NUM_ITERATIONS: %d, BLOCK_SIZE: %d, dt: %.5f\n", NUM_PARTICLES, NUM_ITERATIONS, BLOCK_SIZE, dt);
// }



CPUSimulation::CPUSimulation(int n, int it, int size)
{
        NUM_PARTICLES = n;
        NUM_ITERATIONS = it;
        BLOCK_SIZE = size;
        dt = 1.f;
        count =0;
        printf("NUM_PARTICLES: %d, NUM_ITERATIONS: %d, BLOCK_SIZE: %d, dt: %.5f\n", NUM_PARTICLES, NUM_ITERATIONS, BLOCK_SIZE, dt);
}

void CPUSimulation::init()
{
  particles       = (Particle*)   malloc(sizeof(Particle) * NUM_PARTICLES);
  // fake_random     = (float3*)     malloc(sizeof(float3) * NUM_PARTICLES);

  initial_particles(particles, NUM_PARTICLES);
  printf("Start computing particles on the CPU... ");
  gettimeofday(&t0, NULL);
}

void CPUSimulation::update()
{
  update_random(count, fake_random, NUM_PARTICLES);

  simulation_CPU();

  count++;
}

void CPUSimulation::simulation_CPU()
{
  for (int i = 0; i < NUM_PARTICLES; ++i)
  {
    // particles[i].velocity = fake_random[i];
    update_particle(&particles[i], particles, dt, NUM_PARTICLES);
  }
}

void CPUSimulation::display()
{

  gettimeofday(&t1, NULL);

  printf("Done\n");
  printf("Time: %lfms\n\n", getElapsed(t0, t1));
  print_particle(particles, 5);
}

void CPUSimulation::release()
{
  free(particles);
  free(fake_random);
}

void initial_particles(Particle *particles, int n)
{
  float t;
  for ( int i = 0; i < n; ++i)
  {
    t = 0.01*i;
    particles[i].position = glm::vec3(t, t, t);
    particles[i].velocity = glm::vec3(0.0, 0.0, 0.0);
  }
}

void update_random(int it, float3 *fake_random, int n)
{
  float t;
  for (int i = 0; i < n; ++i)
  {
    t = it * 0.01 + 0.1*i;
    fake_random[i]  = make_float3(t, t, t);
  }
}

void print_particle(Particle *particles, int c){

  for (int i = 0; i < c; ++i)
  {
    printf("position: %f, %f, %f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
    printf("velocity: %f, %f, %f\n", particles[i].velocity.x, particles[i].velocity.y, particles[i].velocity.z);
  }
}
