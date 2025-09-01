import pygame
from .gameObject import GameObject
from .bird import Bird

pygame.init()

# confiugration
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512

BIRD_WIDTH = 34
BIRD_HEIGHT = 24

DIFFICULTY = 1

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

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.blit(backgroundImage, (0,0))

    # pygame.draw.rect(screen, (243, 0, 0), pygame.Rect(30, 30, 90, 60))
    bird.update(screen)
    pygame.display.flip()
    clock.tick(FPS)
pygame.quit()