from abc import ABC, abstractmethod
import pygame

class GameObject(ABC):
    def __init__(self, x:float, y:float, w:float, h:float, asset:str=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.asset = None

        self.vX = 0
        self.vY = 0

        if asset:
            self.asset = pygame.image.load(asset).convert()

    def setVelocity(self, x:float, y:float):
        self.vX = x
        self.vY = y

    def applyForce(self, x:float, y:float):
        self.vX += x
        self.vY += y

    @abstractmethod
    def render(self, surface:pygame.Surface):
        pass

    @abstractmethod
    def onUpdate(self):
        pass

    def update(self, surface:pygame.Surface):
        self.x += self.vX
        self.y += self.vY
        self.onUpdate()
        self.render(surface)
    
    def __str__(self):
        return f"Game object at ({self.x}, {self.y}) of dimension ({self.w}, {self.h})"