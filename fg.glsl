#version 450 core

uniform vec4 myColor;
flat in uint instanceID;
out vec4 FragColor;

struct ssbo_data_t
{
	vec4 position;
	vec4 velocity;
	uint material;
};

layout(std430, binding = 3) buffer particles_ssbo
{
	ssbo_data_t ssbo_data[];
};


void main()
{
	//myColor = vec4(0.0, 0.0, 0.0, 1.0);
	if(ssbo_data[instanceID].material!=1)
		FragColor = vec4(0.0, 1.0, 0.0, 0.4);
	else
		FragColor = vec4(0.0, 0.0, 1.0, 0.4);
}
