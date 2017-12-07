#ifndef __PLANET_HPP__
#define __PLANET_HPP__

#include <vector>
#include <glm/vec3.hpp>
#include "common.hpp"

class Planet
{
	private:
		float origin[3] = { 0, 0, 0 };
		std::vector<Particle> silicateParticles;
		std::vector<Particle> ironParticles;
		float colorSilicate[4] = {0, 1, 0, 0.4};
		float colorIron[4] = {0, 0, 1, 0.4};
	public:
		void setOrigin(double x, double y, double z) {
			origin[0] = x; origin[1] = y; origin[2] = z;
		};
		void addSilicateParticle(double x_pos, double y_pos, double z_pos) {
			silicateParticles.push_back(Particle{glm::vec3(x_pos + origin[0], y_pos + origin[1], z_pos + origin[2]), velocity, false});
			// silicateParticles.push_back(glm::vec3(x + origin[0], y + origin[1], z + origin[2]));
		};
		void addIronParticle(double x, double y, double z) {
			ironParticles.push_back(glm::vec3(x + origin[0], y + origin[1], z + origin[2]));
		};
		std::vector<glm::vec3> getIronParticles() { return ironParticles; };
		std::vector<glm::vec3> getSilicateParticles() { return silicateParticles; };
		float *getSilicateColor() { return colorSilicate; };
		float *getIronColor() { return colorIron; };
		int getSilicateCount() { return silicateParticles.size(); } ;
		int getIronCount() { return ironParticles.size(); };
};

#endif
