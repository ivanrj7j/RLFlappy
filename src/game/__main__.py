from .game import FlappyGame

game = FlappyGame()
score = game.run()
print("\n\n==============================")
print("       YOUR SCORE WAS")
print(f"            {score}")
print("==============================\n\n")