import pygame
from .gameObject import GameObject
import random

class Pipe(GameObject):
    def __init__(self, x, y, w, h, assets:list[str] = None, difficulty:float=0):
        asset = None
        if assets:
            asset = assets[0] if random.random() >= 0.5 else assets[1]
            # 0: top to down 
            # 1: bottom to up 

        super().__init__(x, y, w, h, asset)
        self.setVelocity(-difficulty, 0)

    def onUpdate(self):
        pass

    def render(self, surface):
        pipe = self.asset if self.asset else pygame.Surface((self.w, self.h))
        if not self.asset:
            pipe.fill((0, 255, 0))
        surface.blit(pipe, (self.x, self.y))

    def update(self, surface):
        return super().update(surface)