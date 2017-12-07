#version 130

in vec3 aPos;
uniform mat4 model, view, projection;

void main()
{
	gl_Position = projection * view * model * vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
