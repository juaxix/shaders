#version 330 core
// Input vertex data
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec2 vertexTexCoord;

// Output to fragment shader
out vec2 fragTexCoord;

// Transformation matrices
uniform mat4 model;      // Model space to world space
uniform mat4 view;       // World space to camera space
uniform mat4 projection; // Camera space to clip space

void main()
{
	// Pipeline step 1: Vertex transformation
	// Calculate final position in clip space (MVP matrix)
	gl_Position = projection * view * model * vec4(vertexPosition, 1.0);

	// Pass texture coordinate to fragment shader
	fragTexCoord = vertexTexCoord;
}
