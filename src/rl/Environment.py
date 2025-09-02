from ..game.game import FlappyGame
from ..game.gameObject import GameObject

import numpy as np

class FlappyEnv:
    def __init__(self, game:FlappyGame):
        self.game = game
        self.pipePassed = False

    def reset(self):
        self._resetGameObjects()
        state = self.__getState()
        return state

    def step(self, action:int):
        self._applyAction(action)
        reward = self.__computeReward()
        done = self._isTerminal()
        nextState = self._getState()
        info = {}

        return nextState, reward, done, info
    
    def _resetGameObjects(self):
        self.game.bird.reset()
        self.game.pipeManager.reset()
        self.game.score = 0

    def __getState(self):
        state = [
            self.game.bird.y / self.game.SCREEN_HEIGHT, # normalized position of bird
            (self.game.bird.vY+GameObject.MIN_VEL) / (GameObject.MIN_VEL + GameObject.MAX_VEL), #normalized velocity of bird
        ]

        pipesConsidered = 0
        
        for pipe in self.game.pipeManager.q:
            if pipe.x + pipe.w < self.game.bird.x:
                continue
            if pipesConsidered == 6:
                break
            
            state.append(pipe.x/self.game.SCREEN_WIDTH)
            state.append((pipe.x + pipe.w)/self.game.SCREEN_WIDTH)
            state.append(pipe.y/self.game.SCREEN_WIDTH)
            state.append((pipe.y + pipe.h)/self.game.SCREEN_WIDTH)

            pipesConsidered += 1
        
        return np.array(state, dtype=np.float32)

    def _applyAction(self, action:int):
        if action == 1:
            self.game.bird.jump()

    def __computeReward(self):
        if self.pipePassed:
            return 10
        
        if self.game.running:
            return 1
        
        return -100

    def _isTerminal(self):
        return not self.game.running
    
    def __passPipe(self):
        self.pipePassed = True
    
    def runEnvironemnt(self):
        self.pipePassed = False
        self.game.addPointListener(lambda : self.__passPipe)
        self.game.addPointListener(lambda : print(self.game.score))
        self.game.addUpdateListener(lambda: print(self.__getState()))
        self.game.run()