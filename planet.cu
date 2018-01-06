#include "planet.hpp"
#include <iostream>

Planet::Planet(glm::vec4 origin, glm::vec3 lin_vel, glm::vec3 ang_vel, int num_Fe_particles, int num_Si_particles)
{


  setOrigin(origin);
  //setOrigin(glm::vec3(0,0,0));

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> uniform01(0.0, 1.0);
  std::uniform_real_distribution<double> innerShell(0.0, RInnershell);
  std::uniform_real_distribution<double> outerShell(RInnershell, RImpactor);

  std::cout << num_Si_particles << std::endl;
  std::cout << num_Fe_particles << std::endl;

  for(int i=0; i<num_Si_particles; i++) {
    double rOuter = outerShell(generator);
    double theta = 2 * M_PI * uniform01(generator);
    double phi = acos(1 - 2 * uniform01(generator));
    double x = rOuter * sin(phi) * cos(theta);
    double y = rOuter * sin(phi) * sin(theta);
    double z = rOuter * cos(phi);

    addSilicateParticle(x, y, z);
  }

  for(int i=0; i<num_Fe_particles; i++) {
    double rInner = innerShell(generator);
    double theta = 2 * M_PI * uniform01(generator);
    double phi = acos(1 - 2 * uniform01(generator));
    double x = rInner * sin(phi) * cos(theta);
    double y = rInner * sin(phi) * sin(theta);
    double z = rInner * cos(phi);

    addIronParticle(x, y, z);
  }

  setLinearVelocity(lin_vel, ang_vel);
}

void Planet::setLinearVelocity(glm::vec3 linearVelocity, glm::vec3 angularVelocity)
{
  for(auto &particle : silicateParticles) {
    glm::vec4 arg2 = particle.position - origin;
    glm::vec3 arg = glm::vec3(arg2.x, arg2.y, arg2.z);
    particle.velocity = glm::vec4(glm::cross(angularVelocity / 3600.0f, arg) + linearVelocity, 1.0f);
  }

  for(auto &particle : ironParticles) {
    glm::vec4 arg2 = particle.position - origin;
    glm::vec3 arg = glm::vec3(arg2.x, arg2.y, arg2.z);
    particle.velocity = glm::vec4(glm::cross(angularVelocity / 3600.0f, arg) + linearVelocity, 1.0f);
  }
}

void Planet::setOrigin(glm::vec4 origin)
{
  this->origin = origin;
}

void Planet::setOrigin(double x, double y, double z)
{
  origin.x = x; origin.y = y; origin.z = z; origin.w = 1.0;
}

void Planet::addSilicateParticle(double x_pos, double y_pos, double z_pos)
{
  silicateParticles.push_back(Particle{glm::vec4(x_pos + origin[0], y_pos + origin[1], z_pos + origin[2], 1.0f), glm::vec4(0,0,0,0), 0});
}

void Planet::addIronParticle(double x, double y, double z)
{
  ironParticles.push_back(Particle{glm::vec4(x + origin[0], y + origin[1], z + origin[2], 1.0f), glm::vec4(0,0,0,0), 1});
}

std::vector<Particle> Planet::getIronParticles() { return ironParticles; }

std::vector<Particle> Planet::getSilicateParticles() { return silicateParticles; }

float *Planet::getSilicateColor() { return colorSilicate; }

float *Planet::getIronColor() { return colorIron; }

int Planet::getSilicateCount() { return silicateParticles.size(); }

int Planet::getIronCount() { return ironParticles.size(); }
