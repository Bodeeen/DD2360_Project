#include "simulation.h"

#include <cuda_gl_interop.h>


double getElapsed(struct timeval t0, struct timeval t1)// millisecond
{
    return ((t1.tv_sec-t0.tv_sec) * 1000000LL + (t1.tv_usec-t0.tv_usec)) / 1000.0;
}

bool operator!=(const float3 &a, const float3 &b)
{
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}

bool operator!=(const float4 &a, const float4 &b)
{
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z) || (a.w != b.w);
}

__host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__host__ __device__ float4 operator+(const float4 &a, const float4 &b)
{
  return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, 1.0);
}

__host__ __device__ float3 operator*(const float &a, const float3 &b)
{
  return make_float3(a*b.x, a*b.y, a*b.z);
}

__host__ __device__ float4 operator*(const float &a, const float4 &b)
{
  return make_float4(a*b.x, a*b.y, a*b.z, 1.0);
}

__host__ __device__ bool isIncreasing(Particle p1, Particle p2)
{
  return (glm::dot((p2.velocity - p1.velocity), (p2.position - p1.position)) > 0);
}

__host__ __device__ float calculate_force(Particle p1, Particle p2)
{
  //const float D = 376.78;
	float r = glm::distance(p1.position, p2.position);
	if (r < eps){
		r = eps;
  	}
	if(p1.material && p2.material)//Fe-Fe
	{
		if (r<Core_D_Fe)
		{
			if (isIncreasing(p1, p2))
			{
				return GMfeMfe/(r*r)-(K_Fe*KRP_Fe)*(Dia*Dia-r*r);
			}
			else
			{
				return GMfeMfe/(r*r)-(K_Fe)*(Dia*Dia-r*r);
			}
		}
		else if(r < Dia)
		{
    			return GMfeMfe/(r*r)-(K_Fe)*(Dia*Dia-r*r);
		}
		else
		{
    			return GMfeMfe/(r*r);
		}
	}
	else if(!p1.material && !p2.material)//Si-Si
	{
		if (r<Core_D_Si)
		{
			if (isIncreasing(p1, p2))
			{
				return GMsiMsi/(r*r)-(K_Si*KRP_Si)*(Dia*Dia-r*r);
			}
			else
			{
				return GMsiMsi/(r*r)-(K_Si)*(Dia*Dia-r*r);
			}
		}
		else if(r < Dia)
		{
    			return GMsiMsi/(r*r)-(K_Si)*(Dia*Dia-r*r);
		}
		else
		{
    			return GMsiMsi/(r*r);
		}
	}
	else//Fe-Si
	{
		if (r<Core_D_Fe)
		{
			if (isIncreasing(p1, p2))
			{
				return GMsiMfe/(r*r)-0.5*(K_Si*KRP_Si+K_Fe*KRP_Fe)*(Dia*Dia-r*r);
			}
			else
			{
				return GMsiMfe/(r*r)-0.5*(K_Si+K_Fe)*(Dia*Dia-r*r);
			}
		}
		else if (r < Core_D_Si) {
			if (isIncreasing(p1, p2))
			{
				return GMsiMfe/(r*r)-0.5*(K_Si*KRP_Si+K_Fe)*(Dia*Dia-r*r);
			}
			else
			{
				return GMsiMfe/(r*r)-0.5*(K_Si+K_Fe)*(Dia*Dia-r*r);
			}
		}
		else if(r < Dia)
		{
    			return GMsiMfe/(r*r)-0.5*(K_Si+K_Fe)*(Dia*Dia-r*r);
		}
		else
		{
    			return GMsiMfe/(r*r);
		}
	}
}

__host__ __device__ void update_particle(int idx, Particle *particles, float dt, int num_particles)
{

  //printf("before update %f %f\n", p->position.x, p->position.y);
  glm::vec4 f = glm::vec4(0.0);
  glm::vec4 a = glm::vec4(0.0);

  float fscale;
  glm::vec4 direction;
  for(int i = 0; i<num_particles; i++)
  {
    if(idx != i){
//printf("|f|: %f %f %f\n", f.x, f.y, f.z);
    fscale  = calculate_force(particles[idx], particles[i]);
    direction = glm::normalize(particles[i].position - particles[idx].position);
    f += direction * fscale;
    }
  }
//printf("f: %f %f %f\n", f.x, f.y, f.z);
  if(particles[idx].material)
    a = (1/(float)M_Fe)*f;
  else
    a = (1/(float)M_Si)*f;

  particles[idx].velocity += a*dt;
  particles[idx].position += dt * particles[idx].velocity;
  //return p.position + dt * p.velocity;
}

__global__ void simulation_GPU(Particle *particles, int n, float dt)
{
        const int i = blockIdx.x*blockDim.x + threadIdx.x;

        if(i<n){
          update_particle(i, particles, dt, n);
        }
}

GPUSimulation::GPUSimulation()
{
//  std::cout << NUM_IRON_PARTICLES << std::endl;
//  std::cout << NUM_SILICATE_PARTICLES << std::endl;
}

void GPUSimulation::init()
{
  NUM_PARTICLES = (NUM_SILICATE_PARTICLES+NUM_IRON_PARTICLES) * 2 ;
    std::shared_ptr<Planet> planetA = std::make_shared<Planet>(glm::vec4(Xcenter, 0.0, Zxenter, 1.0), glm::vec3(Vinitx, 0, 0), glm::vec3(0, 3.0973, 0), NUM_IRON_PARTICLES, NUM_SILICATE_PARTICLES);
    std::shared_ptr<Planet> planetB = std::make_shared<Planet>(glm::vec4(-Xcenter, 0.0, -Zxenter, 1.0), glm::vec3(-Vinitx, 0, 0), glm::vec3(0, -3.0973, 0), NUM_IRON_PARTICLES, NUM_SILICATE_PARTICLES);

    planets.push_back(planetA);
    planets.push_back(planetB);
    //Planet planetA = new Planet(glm::vec3(23925.0, 0.0, 9042.7), glm::vec3(3.2416, 0, 0), glm::vec3(0, 3.0973, 0), NUM_IRON_PARTICLES, NUM_SILICATE_PARTICLES);
    //Planet planetB = new Planet(glm::vec3(-23925.0, 0.0, -9042.7), glm::vec3(-3.2416, 0, 0), glm::vec3(0, -3.0973, 0), NUM_IRON_PARTICLES, NUM_SILICATE_PARTICLES);


    dt = 0.001f;
    gpu_calculating_time =0.;
    count =0;
    printf("NUM_PARTICLES: %d, NUM_ITERATIONS: %d, BLOCK_SIZE: %d, dt: %.5f\n", NUM_PARTICLES, NUM_ITERATIONS, BLOCK_SIZE, dt);

  //NUM_PARTICLES = (NUM_SILICATE_PARTICLES+NUM_IRON_PARTICLES) * 2 ;
  //cudaMalloc( &d_p, (NUM_SILICATE_PARTICLES+NUM_IRON_PARTICLES)*2 * sizeof(Particle));
  // cudaMalloc( &d_pB, (NUM_SILICATE_PARTICLES+NUM_IRON_PARTICLES) * sizeof(Particle));
  cudaMalloc( &d_p, NUM_PARTICLES * sizeof(Particle) );

  std::vector<Particle> ironA = planets[0]->getIronParticles();
  std::vector<Particle> silA = planets[0]->getSilicateParticles();
  std::vector<Particle> ironB = planets[1]->getIronParticles();
  std::vector<Particle> silB = planets[1]->getSilicateParticles();

  all = ironA;

  all.insert(all.end(), silA.begin(), silA.end());
  all.insert(all.end(), ironB.begin(), ironB.end());
  all.insert(all.end(), silB.begin(), silB.end());

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //allDevice = all;

  // cudaMemcpy( d_pB, planets[1], NUM_PARTICLES * sizeof(Particle) , cudaMemcpyHostToDevice );
  //checkCUDAError();
}

void GPUSimulation::update(cudaGraphicsResource_t &ssbo_handle)
{
  //checkCUDAError();
  //cudaMemcpy( d_p, &all[0], (NUM_SILICATE_PARTICLES+NUM_IRON_PARTICLES)*2 * sizeof(Particle) , cudaMemcpyHostToDevice );
  cudaGraphicsMapResources(1, &ssbo_handle);
  size_t t = NUM_PARTICLES;
  cudaGraphicsResourceGetMappedPointer((void **)&d_p, &t, ssbo_handle);
  cudaEventRecord(start);
  //allDevice = all;
  simulation_GPU<<<(NUM_PARTICLES + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_p, NUM_PARTICLES, dt);
  //all = allDevice;
  // for(auto &particle : all) {
  //   std::cout << particle.position.x << " " << particle.position.y << " " << particle.position.z << std::endl;
  // }
  //exit(1);
  cudaEventRecord(stop);
  cudaGraphicsUnmapResources(1, &ssbo_handle);

  cudaEventSynchronize(stop);

  //cudaMemcpy( &all[0], d_p, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost );
  checkCUDAError();

  float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "CUDA " << milliseconds << " ms" << std::endl;
  count++;
}

void GPUSimulation::display()
{
  //cudaMemcpy( particles_GPU, d_p, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost );
  //checkCUDAError();
  //all = allDevice;

  gettimeofday(&t1, NULL);

  printf("Done\n");
  printf("Time\ncalculating: %lfms \t all: %lfms\n\n", gpu_calculating_time, getElapsed(t0, t1));
  //print_particle(particles_GPU, 5);
}

void GPUSimulation::release()
{
  cudaFree( d_p );
  //cudaFree( d_r );

	//free(particles);
	//free(fake_random);
	//free(particles_GPU);
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

//
//
// CPUSimulation::CPUSimulation(int n, int it, int size)
// {
//         NUM_PARTICLES = n;
//         NUM_ITERATIONS = it;
//         BLOCK_SIZE = size;
//         dt = 1.f;
//         count =0;
//         printf("NUM_PARTICLES: %d, NUM_ITERATIONS: %d, BLOCK_SIZE: %d, dt: %.5f\n", NUM_PARTICLES, NUM_ITERATIONS, BLOCK_SIZE, dt);
// }
//
// void CPUSimulation::init()
// {
//   particles       = (Particle*)   malloc(sizeof(Particle) * NUM_PARTICLES);
//   // fake_random     = (float3*)     malloc(sizeof(float3) * NUM_PARTICLES);
//
//   initial_particles(particles, NUM_PARTICLES);
//   printf("Start computing particles on the CPU... ");
//   gettimeofday(&t0, NULL);
// }
//
// void CPUSimulation::update()
// {
//   update_random(count, fake_random, NUM_PARTICLES);
//
//   simulation_CPU();
//
//   count++;
// }
//
// void CPUSimulation::simulation_CPU()
// {
//   for (int i = 0; i < NUM_PARTICLES; ++i)
//   {
//     // particles[i].velocity = fake_random[i];
//     update_particle(&particles[i], particles, dt, NUM_PARTICLES);
//   }
// }
//
// void CPUSimulation::display()
// {
//
//   gettimeofday(&t1, NULL);
//
//   printf("Done\n");
//   printf("Time: %lfms\n\n", getElapsed(t0, t1));
//   print_particle(particles, 5);
// }
//
// void CPUSimulation::release()
// {
//   free(particles);
//   free(fake_random);
// }
//
// void initial_particles(Particle *particles, int n)
// {
//   float t;
//   for ( int i = 0; i < n; ++i)
//   {
//     t = 0.01*i;
//     particles[i].position = glm::vec3(t, t, t);
//     particles[i].velocity = glm::vec3(0.0, 0.0, 0.0);
//   }
// }
//
// void update_random(int it, float3 *fake_random, int n)
// {
//   float t;
//   for (int i = 0; i < n; ++i)
//   {
//     t = it * 0.01 + 0.1*i;
//     fake_random[i]  = make_float3(t, t, t);
//   }
// }
//
// void print_particle(Particle *particles, int c){
//
//   for (int i = 0; i < c; ++i)
//   {
//     printf("position: %f, %f, %f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
//     printf("velocity: %f, %f, %f\n", particles[i].velocity.x, particles[i].velocity.y, particles[i].velocity.z);
//   }
// }
