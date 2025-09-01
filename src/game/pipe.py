import pygame
from .gameObject import GameObject
import random

class Pipe(GameObject):
    def __init__(self, x, y, w, h, asset = None, difficulty:float=0):
        super().__init__(x, y, w, h, asset)

        self.setVelocity(-difficulty, 0)

        self.pipe = self.asset if self.asset else pygame.Surface((self.w, self.h))
        if not self.asset:
            self.pipe.fill((0, 255, 0))

    def onUpdate(self):
        pass

    def render(self, surface):
        surface.blit(self.pipe, (self.x, self.y))