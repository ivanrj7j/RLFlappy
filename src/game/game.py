import pygame
from .gameObject import GameObject
from .bird import Bird
from .pipe import Pipe

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

pipe = Pipe(2*SCREEN_WIDTH/3, 0, 30, 90, difficulty=DIFFICULTY)

while running:
    screen.blit(backgroundImage, (0,0))

    bird.applyForce(0, 1e-2)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                print("JUMP")
                bird.applyForce(0, -1.5)
        elif event.type == pygame.KEYUP:
            if event.key in (pygame.K_w, pygame.K_SPACE, pygame.K_UP):
                print("JUMP")
                bird.applyForce(0, -1.5)

    # pygame.draw.rect(screen, (243, 0, 0), pygame.Rect(30, 30, 90, 60))
    bird.update(screen)
    pipe.update(screen)


    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()