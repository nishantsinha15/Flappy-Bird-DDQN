from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
# agent = myAgentHere(allowed_actions=p.getActionSet())

p.init()
reward = 0.0


for i in range(100000000):
    if p.game_over():
        p.reset_game()
        print("Game Ended ", reward)
    observation = game.getGameState()
    print(observation)

    # get action from agent
    if random.random() < 0.1: action = 119
    else: action = None
    reward = p.act(action)
