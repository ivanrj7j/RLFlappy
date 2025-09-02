from .gameObject import GameObject
import pygame

class Bird(GameObject):
    UPDATE_EVERY = 30
    # number of frames to update the sprite 

    def __init__(self, x, y, w, h, asset:list[str] = None, color=(255, 0, 0)):
        # asset will be a list of character sprites which will be used as animation (3 assets)
        super().__init__(x, y, w, h, asset if not asset else asset[1])
        self.resetArguments = (x, y, w, h, asset, color)
        self.color = color
        self.assets = [pygame.image.load(x) for x in asset]
        self.currentAsset = 0
        

    def reset(self):
        self.__init__(*self.resetArguments)

    def render(self, surface):
        self.bird = self.assets[self.currentAsset//Bird.UPDATE_EVERY] if self.asset else pygame.Surface((self.w, self.h))
        if not self.asset:
            self.bird.fill(self.color)
            
        surface.blit(self.bird, (self.x, self.y))

    def jump(self):
        self.applyForce(0, -1.5)

    def onUpdate(self):
        self.currentAsset = (self.currentAsset+1) % (3*Bird.UPDATE_EVERY)