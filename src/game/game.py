import pygame

from .gameObject import GameObject
from .bird import Bird
from .pipeManager import PipeManager

pygame.init()

# confiugration
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512

BIRD_WIDTH = 34
BIRD_HEIGHT = 24

DIFFICULTY = 0.1

FPS = 244

#pygame related objects
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("RLFlappy")

running = True

# assets
backgroundImage = pygame.image.load("../flappy-bird-assets/sprites/background-day.png").convert()

# game objects 
bird = Bird((SCREEN_WIDTH-BIRD_WIDTH)/2, (SCREEN_HEIGHT-BIRD_HEIGHT)/2, BIRD_WIDTH, BIRD_HEIGHT, ["../flappy-bird-assets/sprites/redbird-upflap.png", "../flappy-bird-assets/sprites/redbird-midflap.png", "../flappy-bird-assets/sprites/redbird-downflap.png"])

pipeManager = PipeManager(SCREEN_WIDTH, SCREEN_HEIGHT, DIFFICULTY)

def isColliding(a:GameObject, b:GameObject):
    return not (
        a.x + a.w < b.x or  # A is left of B
        b.x + b.w < a.x or  # B is left of A
        a.y + a.h < b.y or  # A is above B
        b.y + b.h < a.y     # B is above A
    )

while running:
    screen.blit(backgroundImage, (0,0))

    bird.applyForce(0, 1e-2)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                bird.applyForce(0, -1.5)
        elif event.type == pygame.KEYUP:
            if event.key in (pygame.K_w, pygame.K_SPACE, pygame.K_UP):
                bird.applyForce(0, -1.5)
    # pygame.draw.rect(screen, (243, 0, 0), pygame.Rect(30, 30, 90, 60))
    pipeManager.update(screen)
    bird.update(screen)
    
    if bird.y < -BIRD_HEIGHT or bird.y > SCREEN_HEIGHT:
        running = False

    for pipe in pipeManager.q:
        if isColliding(bird, pipe):
            running = False
            break

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()