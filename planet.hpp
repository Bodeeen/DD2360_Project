#ifndef __PLANET_HPP__
#define __PLANET_HPP__

#include <vector>
#include <chrono>
#include <random>
#include <glm/vec3.hpp>
#include "common.hpp"

class Planet
{
	private:
		glm::vec3 origin;

		std::vector<Particle> silicateParticles;
		std::vector<Particle> ironParticles;

		//std::vector<float> colorSilicate;
		//std::vector<float> colorIron;

		float colorSilicate[4] = {0.0f, 1.0f, 0.0f, 0.1f};
		float colorIron[4] = {0.0f, 0.0f, 1.0f, 0.1f};

	public:
		Planet(glm::vec3, glm::vec3, glm::vec3, int, int);

		void setOrigin(glm::vec3);
		void setOrigin(double, double, double);

		void setLinearVelocity(glm::vec3, glm::vec3);

		void addSilicateParticle(double, double, double);
		void addIronParticle(double, double, double);

		std::vector<Particle> getIronParticles();
		std::vector<Particle> getSilicateParticles();

		float *getSilicateColor();
		float *getIronColor();

		int getSilicateCount();
		int getIronCount();
};

#endif
