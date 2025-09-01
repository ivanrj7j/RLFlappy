from collections import deque
from random import random
from itertools import chain

from .pipe import Pipe


class PipeManager:
    PIPE_WIDTH = 52
    PIPE_DISTANCE = 45
    def __init__(self, screenWidth:float, screenHeight:float, difficulty:float, p=3/244):
        # p is the probablity that a pipe spawns
        self.w = screenWidth
        self.h = screenHeight
        self.difficulty = difficulty*3
        
        pipes = []
        for i in range(10):
            pipes.extend(self.generatePipePair(self.w + max(10, i*(PipeManager.PIPE_DISTANCE + PipeManager.PIPE_WIDTH))))
        self.q = deque(pipes)

    def generatePipePair(self, x:float):
        h1 = random() * 0.8 * self.h
        h2 = random() * 0.8 * self.h

        if h1 + h2 > self.h*0.85:
            percentage = 2 - (h1+h2)/(self.h*0.85)
            h1 *= percentage
            h2 *= percentage

        pipe1 = Pipe(x, 0, PipeManager.PIPE_WIDTH, h1, difficulty=self.difficulty)
        pipe2 = Pipe(x, self.h-h2, PipeManager.PIPE_WIDTH, h2, difficulty=self.difficulty)

        return pipe1, pipe2
    
    def update(self, screen):
        while len(self.q) > 0 and self.q[0].x < -PipeManager.PIPE_WIDTH:
            self.q.popleft()
            if(len(self.q)) <= 18:
                self.q.extend(self.generatePipePair(self.q[-1].x + (PipeManager.PIPE_DISTANCE + PipeManager.PIPE_WIDTH)))


        for pipe in self.q:
            pipe.update(screen)