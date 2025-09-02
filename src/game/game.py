import pygame
from .gameObject import GameObject
from .bird import Bird
from .pipeManager import PipeManager

class FlappyGame:
    SCREEN_WIDTH = 288
    SCREEN_HEIGHT = 512
    BIRD_WIDTH = 34
    BIRD_HEIGHT = 24
    DIFFICULTY = 0.1
    FPS = 244

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("RLFlappy")
        self.backgroundImage = pygame.image.load("../flappy-bird-assets/sprites/background-day.png").convert()
        self.bird = Bird((self.SCREEN_WIDTH-self.BIRD_WIDTH)/2, (self.SCREEN_HEIGHT-self.BIRD_HEIGHT)/2, self.BIRD_WIDTH, self.BIRD_HEIGHT, ["../flappy-bird-assets/sprites/redbird-upflap.png", "../flappy-bird-assets/sprites/redbird-midflap.png", "../flappy-bird-assets/sprites/redbird-downflap.png"])
        self.pipeManager = PipeManager(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.DIFFICULTY)
        self.score = 0
        self.running = True

        self.onUpdateMethods = []
        self.onPointMethods = []

    def addUpdateListener(self, func):
        self.onUpdateMethods.append(func)

    def addPointListener(self, func):
        self.onPointMethods.append(func)

    @staticmethod
    def isColliding(a: GameObject, b: GameObject):
        return not (
            a.x + a.w < b.x or  # A is left of B
            b.x + b.w < a.x or  # B is left of A
            a.y + a.h < b.y or  # A is above B
            b.y + b.h < a.y     # B is above A
        )

    def run(self):
        while self.running:
            self.screen.blit(self.backgroundImage, (0,0))
            self.bird.applyForce(0, 1e-2)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.bird.jump()
                elif event.type == pygame.KEYUP:
                    if event.key in (pygame.K_w, pygame.K_SPACE, pygame.K_UP):
                        self.bird.jump()

            self.pipeManager.update(self.screen)
            self.bird.update(self.screen)

            if self.bird.y < -self.BIRD_HEIGHT or self.bird.y > self.SCREEN_HEIGHT:
                self.running = False

            for pipe in self.pipeManager.q:
                if self.isColliding(self.bird, pipe):
                    self.running = False
                    break
                if not pipe.passed and pipe.x+pipe.w < self.bird.x:
                    self.score += 1
                    pipe.passed = True
                    for pointMethod in self.onPointMethods:
                        pointMethod()

            for updateMethod in self.onUpdateMethods:
                updateMethod()

            pygame.display.flip()
            self.clock.tick(self.FPS)
        pygame.quit()
        return self.score