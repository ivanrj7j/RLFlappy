from .game import FlappyGame

game = FlappyGame()

def displayScore():
    print("Current score is", game.score)

game.addPointListener(displayScore)

score = game.run()
print("\n\n==============================")
print("       YOUR SCORE WAS")
print(f"            {score}")
print("==============================\n\n")