import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os

pygame.init()

# project settings
screen_width = 1000
screen_height = 800
background_color = (0, 0, 0, 1)
drawing_color = (1, 1, 1, 1)

screen = pygame.display.set_mode((screen_width, screen_height), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Textured Cube - OpenGL Pipeline Example")

# Define cube vertices (8 corners of a cube)
vertices = [
	# Front face
	(-0.5, -0.5, 0.5),  # Bottom left
	(0.5, -0.5, 0.5),  # Bottom right
	(0.5, 0.5, 0.5),  # Top right
	(-0.5, 0.5, 0.5),  # Top left

	# Back face
	(-0.5, -0.5, -0.5),  # Bottom left
	(0.5, -0.5, -0.5),  # Bottom right
	(0.5, 0.5, -0.5),  # Top right
	(-0.5, 0.5, -0.5),  # Top left
]

# Define faces using indices into the vertices array
faces = [
	(0, 1, 2, 3),  # Front face
	(5, 4, 7, 6),  # Back face
	(4, 0, 3, 7),  # Left face
	(1, 5, 6, 2),  # Right face
	(3, 2, 6, 7),  # Top face
	(4, 5, 1, 0),  # Bottom face
]

# Texture coordinates for each vertex of each face
tex_coords = [
	(0, 0), (1, 0), (1, 1), (0, 1),  # Front face
	(0, 0), (1, 0), (1, 1), (0, 1),  # Back face
	(0, 0), (1, 0), (1, 1), (0, 1),  # Left face
	(0, 0), (1, 0), (1, 1), (0, 1),  # Right face
	(0, 0), (1, 0), (1, 1), (0, 1),  # Top face
	(0, 0), (1, 0), (1, 1), (0, 1),  # Bottom face
]

def load_shader_from_file(file_path):
	"""Load shader source code from a file."""
	with open(file_path, 'r', encoding='utf-8-sig') as file:
		return file.read()

# Vertex Shader: Processes each vertex position
vertex_shader = load_shader_from_file("vertex_shader.glsl")

# Fragment Shader: Calculates the color for each pixel/fragment
fragment_shader = load_shader_from_file("fragment_shader.glsl")

def load_texture(filename):
	"""Load a texture from file and prepare it for OpenGL."""
	# Load image using pygame
	image = pygame.image.load(filename)
	image_data = pygame.image.tostring(image, "RGBA", True)
	width, height = image.get_size()

	# Create OpenGL texture
	texture_id = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, texture_id)

	# Set texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

	# Upload texture data
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)

	return texture_id

def compile_shader(shader_source, shader_type):
	"""Compile shader from source code."""
	shader = glCreateShader(shader_type)
	glShaderSource(shader, shader_source)
	glCompileShader(shader)

	# Check compile status
	result = glGetShaderiv(shader, GL_COMPILE_STATUS)
	if not result:
		error_log = glGetShaderInfoLog(shader)
		print(f"Shader compilation failed: {error_log}")
		glDeleteShader(shader)
		return None

	return shader

def create_shader_program(vertex_shader_source, fragment_shader_source):
	"""Create a shader program from vertex and fragment shader sources."""
	# Compile shaders
	vs = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)
	fs = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER)

	# Create and link program
	program = glCreateProgram()
	glAttachShader(program, vs)
	glAttachShader(program, fs)
	glLinkProgram(program)

	# Check link status
	result = glGetProgramiv(program, GL_LINK_STATUS)
	if not result:
		error_log = glGetProgramInfoLog(program)
		print(f"Shader program linking failed: {error_log}")
		glDeleteProgram(program)
		return None

	# Clean up
	glDeleteShader(vs)
	glDeleteShader(fs)

	return program

def initialise():
	"""Initialize OpenGL settings and shader program."""
	glClearColor(background_color[0], background_color[1], background_color[2], background_color[3])

	# Enable depth testing for proper 3D rendering
	glEnable(GL_DEPTH_TEST)
	glEnable(GL_TEXTURE_2D)

	# Create and use shader program
	global shader_program
	shader_program = create_shader_program(vertex_shader, fragment_shader)
	glUseProgram(shader_program)

	# Generate vertex array object (VAO)
	global vao
	vao = glGenVertexArrays(1)
	glBindVertexArray(vao)

	# Prepare vertex data for the GPU
	# Pipeline step 0: Prepare data for the vertex shader
	vertex_data = []
	for face_idx, face in enumerate(faces):
		for i, vertex_idx in enumerate(face):
			# Add position
			vertex_data.extend(vertices[vertex_idx])
			# Add texture coordinate
			vertex_data.extend(tex_coords[face_idx * 4 + i])

	vertex_data = np.array(vertex_data, dtype=np.float32)

	# Generate and bind vertex buffer object (VBO)
	global vbo
	vbo = glGenBuffers(1)
	glBindBuffer(GL_ARRAY_BUFFER, vbo)
	glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

	# Set vertex attribute pointers
	# Position attribute (3 floats)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), None)
	glEnableVertexAttribArray(0)

	# Texture coordinate attribute (2 floats)
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), ctypes.c_void_p(3 * sizeof(GLfloat)))
	glEnableVertexAttribArray(1)

	# Load texture (replace with your texture path)
	global texture
	# Create a sample texture if none exists
	sample_texture_path = "sample_texture.png"
	if not os.path.exists(sample_texture_path):
		# Create a simple checkered texture
		size = 256
		texture_surface = pygame.Surface((size, size), pygame.SRCALPHA)

		# Draw checkered pattern
		square_size = 32
		for y in range(0, size, square_size):
			for x in range(0, size, square_size):
				color = (255, 0, 0, 255) if (x + y) % (square_size * 2) == 0 else (0, 255, 0, 255)
				pygame.draw.rect(texture_surface, color, (x, y, square_size, square_size))

		# Draw some text on the texture to demonstrate orientation
		font = pygame.font.SysFont(None, 48)
		text = font.render("TEXTURE", True, (255, 255, 255))
		texture_surface.blit(text, (size // 2 - text.get_width() // 2, size // 2 - text.get_height() // 2))

		pygame.image.save(texture_surface, sample_texture_path)

	texture = load_texture(sample_texture_path)

	# Set up transformation matrices
	# Pipeline step 1: Define transformations
	projection = glm_perspective(60, screen_width / screen_height, 0.1, 100.0)
	view = glm_translate(0, 0, -3)
	model = glm_rotate(0, 0, 1, 0)  # Initial model matrix

	# Get uniform locations
	projection_loc = glGetUniformLocation(shader_program, "projection")
	view_loc = glGetUniformLocation(shader_program, "view")
	global model_loc
	model_loc = glGetUniformLocation(shader_program, "model")

	# Set uniform values
	glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)
	glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
	glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)


def glm_perspective(fov, aspect, near, far):
	"""Create a perspective projection matrix."""
	# Convert to radians
	fov_rad = np.radians(fov)

	# Calculate matrix values
	f = 1.0 / np.tan(fov_rad / 2.0)

	# Create the matrix
	perspective = np.zeros((4, 4), dtype=np.float32)
	perspective[0, 0] = f / aspect
	perspective[1, 1] = f
	perspective[2, 2] = (far + near) / (near - far)
	perspective[2, 3] = -1.0
	perspective[3, 2] = (2.0 * far * near) / (near - far)

	return perspective.flatten()


def glm_translate(x, y, z):
	"""Create a translation matrix."""
	translation = np.identity(4, dtype=np.float32)
	translation[3, 0] = x
	translation[3, 1] = y
	translation[3, 2] = z

	return translation.flatten()


def glm_rotate(angle, x, y, z):
	"""Create a rotation matrix."""
	# Convert to radians
	angle_rad = np.radians(angle)

	c = np.cos(angle_rad)
	s = np.sin(angle_rad)
	axis = np.array([x, y, z], dtype=np.float32)
	if np.linalg.norm(axis) > 0:
		axis = axis / np.linalg.norm(axis)

	x, y, z = axis

	# Create the rotation matrix
	rotation = np.zeros((4, 4), dtype=np.float32)

	rotation[0, 0] = x * x * (1 - c) + c
	rotation[0, 1] = y * x * (1 - c) + z * s
	rotation[0, 2] = x * z * (1 - c) - y * s
	rotation[0, 3] = 0.0

	rotation[1, 0] = x * y * (1 - c) - z * s
	rotation[1, 1] = y * y * (1 - c) + c
	rotation[1, 2] = y * z * (1 - c) + x * s
	rotation[1, 3] = 0.0

	rotation[2, 0] = x * z * (1 - c) + y * s
	rotation[2, 1] = y * z * (1 - c) - x * s
	rotation[2, 2] = z * z * (1 - c) + c
	rotation[2, 3] = 0.0

	rotation[3, 0] = 0.0
	rotation[3, 1] = 0.0
	rotation[3, 2] = 0.0
	rotation[3, 3] = 1.0

	return rotation.flatten()


def display():
	"""Render the scene."""
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

	# Use our shader program
	glUseProgram(shader_program)

	# Update model matrix for rotation
	# Pipeline step 2: Update transformations for this frame
	global rotation_angle
	rotation_angle += 1  # Increment rotation angle
	model = glm_rotate(rotation_angle, 0.5, 1, 0.3)  # Rotate around axis (0.5, 1, 0.3)
	glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

	# Bind the texture
	# Pipeline step 3: Bind resources (textures)
	glBindTexture(GL_TEXTURE_2D, texture)

	# Bind VAO and draw the cube
	# Pipeline step 4: Draw command initiates the GPU rendering pipeline
	glBindVertexArray(vao)
	glDrawArrays(GL_QUADS, 0, 24)  # 6 faces * 4 vertices per face

	# Note: After this draw call, the graphics pipeline takes over:
	# 1. Vertex Shader processes each vertex
	# 2. Primitive Assembly groups vertices into triangles
	# 3. Rasterization converts triangles to fragments
	# 4. Fragment Shader processes each fragment to determine its color
	# 5. Per-Fragment Operations (depth testing, blending, etc.)
	# 6. Framebuffer operations (writing to the display)


# Initialize rotation angle
rotation_angle = 0


def cleanup():
	"""Clean up OpenGL resources."""
	glDeleteVertexArrays(1, [vao])
	glDeleteBuffers(1, [vbo])
	glDeleteTextures(1, [texture])
	glDeleteProgram(shader_program)


# Main loop
initialise()
try:
	done = False
	while not done:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				done = True
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					done = True

		# Render the scene
		display()

		# Swap buffers
		pygame.display.flip()

		# Control frame rate
		pygame.time.wait(16)  # ~60 FPS

finally:
	# Clean up resources
	cleanup()
	pygame.quit()
