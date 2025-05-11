import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

pygame.init()
screen_width = 1000
screen_height = 800

screen = pygame.display.set_mode((screen_width, screen_height), DOUBLEBUF|OPENGL)
pygame.display.set_caption("Open GL with Python")

#setup cam
def init_ortho():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, 1000, 800, 0)

init_ortho()

def draw_star(x, y, size):
    glPointSize(size)
    glBegin(GL_POINTS)
    glVertex2i(x, y)
    glEnd()

is_done = False
angle = 0
while not is_done:
    for event in pygame.event.get():
        if event.type.__eq__(pygame.QUIT):
            is_done = True

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    w2 = int(screen_width/2)
    h2 = int(screen_height/2)
    angle += 0.1
    draw_star(w2, h2, 5)
    #draw_star(231, 151, 20)
    #draw_star(257, 253, 20)
    #draw_star(303, 180, 15)
    pygame.draw.rect(screen, (50, 220, 10), pygame.Rect(50, 50, 100, 100))
    pygame.display.flip() # Flip to the second buffer (we use a double one)
    pygame.time.wait(100)

pygame.quit()