import pygame
from .gameObject import GameObject

pygame.init()

screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption("RLFlappy")

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.draw.rect(screen, (243, 0, 0), pygame.Rect(30, 30, 90, 60))
    pygame.display.flip()
pygame.quit()