#version 130

uniform vec4 myColor;
out vec4 FragColor;

void main()
{
	//myColor = vec4(0.0, 1.0, 0.0, 0.4);
	FragColor = myColor;
}
