#version 330 core
// Input from vertex shader
in vec2 fragTexCoord;

// Output - final fragment color
out vec4 fragColor;

// Texture sampler
uniform sampler2D textureSampler;

void main()
{
	// Pipeline step 5: Fragment processing

	// Sample texture at the interpolated texture coordinate
	vec4 texColor = texture(textureSampler, fragTexCoord);

	// fragColor is our final output for this fragment/pixel
	fragColor = texColor;
}