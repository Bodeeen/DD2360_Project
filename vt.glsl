#version 450 core

in vec3 aPos;
flat out uint instanceID;
uniform mat4 model, view, projection;

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
	vec4 position = ssbo_data[gl_InstanceID].position;


	gl_Position = projection * view * vec4(aPos.x + position.x, aPos.y + position.y, aPos.z + position.z, 1.0);

	instanceID = gl_InstanceID;
}
