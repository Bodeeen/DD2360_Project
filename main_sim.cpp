#include <helper.h>
#include <simulation.cuh>
// #include <cuda_runtime_api.h>

int main(int argc, char const *argv[])
{
	int n, it, size;
	handle_opt( argc, argv, &n, &it, &size);
	Simulation *simulation = new MockupSimulation(n, it, size);
	// simulation->preset_values(n, it, size);
	simulation->init();
	for (int i = 0; i < it; ++i)
	{
		simulation->update();
	}
	simulation->display();
	simulation->release();

	Simulation *simulation_cpu = new CPUSimulation(n, it, size);

	simulation_cpu->init();
	for (int i = 0; i < it; ++i)
	{
		simulation_cpu->update();
	}
	simulation_cpu->display();
	simulation_cpu->release();

	return 0;
}
