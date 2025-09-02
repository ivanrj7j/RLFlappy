from .Environment import FlappyEnv
from ..game.game import FlappyGame

game = FlappyGame()
env = FlappyEnv(game)

env.runEnvironemnt()