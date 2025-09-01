from collections import deque
from random import random, randint
from itertools import chain

from .pipe import Pipe

"""
Pipe generation:

Pipes will have a minimum and maximum distance between themselves
Pipes will have a minimum size
The y of center point of two consecutive pipe set will have a maximum absolute difference
The gaps can be imagined something like a candlestick chart
"""


class PipeManager:
    PIPE_WIDTH = 52

    MIN_PIPE_HEIGHT = 40
    MAX_PIPE_HEIGHT = 0.8
    MIN_PIPE_DISTANCE = 80
    MAX_PIPE_DISTANCE = 130
    MAX_CENTER_DIFF = 100
    GAP_SIZE = 100

    def __init__(self, screenWidth:float, screenHeight:float, difficulty:float, p=3/244):
        # p is the probablity that a pipe spawns
        self.w = screenWidth
        self.h = screenHeight
        self.difficulty = difficulty*3
        self.last_center_y = None
        
        pipes = []
        for i in range(10):
            pipes.extend(self.generatePipePair(self.w + max(10, i*(PipeManager.MIN_PIPE_DISTANCE + PipeManager.PIPE_WIDTH + randint(0, 30)))))
        self.q = deque(pipes)

    def generatePipePair(self, x:float):
        # Determine gap center
        if self.last_center_y is None:
            center_y = randint(int(self.h*0.3), int(self.h*0.7))
        else:
            min_center = max(int(self.last_center_y - PipeManager.MAX_CENTER_DIFF), int(self.h*0.2))
            max_center = min(int(self.last_center_y + PipeManager.MAX_CENTER_DIFF), int(self.h*0.8))
            center_y = randint(min_center, max_center)

        self.last_center_y = center_y

        gap_size = PipeManager.GAP_SIZE + randint(-20, 20)  # Candlestick-like gap
        top_height = max(PipeManager.MIN_PIPE_HEIGHT, center_y - gap_size//2)
        bottom_height = max(PipeManager.MIN_PIPE_HEIGHT, self.h - (center_y + gap_size//2))

        # Clamp pipe heights
        top_height = min(top_height, int(self.h * PipeManager.MAX_PIPE_HEIGHT))
        bottom_height = min(bottom_height, int(self.h * PipeManager.MAX_PIPE_HEIGHT))

        pipe1 = Pipe(x, 0, PipeManager.PIPE_WIDTH, top_height, difficulty=self.difficulty)
        pipe2 = Pipe(x, self.h-bottom_height, PipeManager.PIPE_WIDTH, bottom_height, difficulty=self.difficulty)

        return pipe1, pipe2
    
    def update(self, screen):
        while len(self.q) > 0 and self.q[0].x < -PipeManager.PIPE_WIDTH:
            self.q.popleft()
            if(len(self.q)) <= 18:
                self.q.extend(self.generatePipePair(self.q[-1].x + (PipeManager.MIN_PIPE_DISTANCE + PipeManager.PIPE_WIDTH + randint(0, 30))))


        for pipe in self.q:
            pipe.update(screen)