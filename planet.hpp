#include <vector>
#include <glm/vec3.hpp>

class planet
{
	private:
		float origin[3] = { 0, 0, 0 };
		std::vector<glm::vec3> silicateParticles;
		std::vector<glm::vec3> ironParticles;
    		float colorSilicate[4] = {0, 1, 0, 0.4};
    		float colorIron[4] = {0, 0, 1, 0.4};
	public:
		void setOrigin(double x, double y, double z) {
			origin[0] = x; origin[1] = y; origin[2] = z;
		};
		void addSilicateParticle(double x, double y, double z) {
			silicateParticles.push_back(glm::vec3(x + origin[0], y + origin[1], z + origin[2]));
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
