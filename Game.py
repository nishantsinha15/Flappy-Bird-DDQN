from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random
import DDQN_Agent


def train(FRAME_TRAIN=1000000):
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=True)
    p.init()
    reward = 0.0
    ob = game.getGameState()
    state = ob
    next_state = ob
    total_reward = 0
    agent = DDQN_Agent.DeepQAgent()
    agent2 = DDQN_Agent.DeepQAgent()
    batch_size = 32

    for i in range(FRAME_TRAIN):
        coin_toss = random.random > 0.5

        if p.game_over():
            p.reset_game()
            print("Total reward = ", total_reward)
            total_reward = 0

        # get action from agent
        if coin_toss:
            action = agent.act(state)
        else:
            action = agent2.act(state)

        # take action
        reward = p.act(p.getActionSet[action])
        next_state = game.getGameState()
        total_reward += reward

        # remember and replay
        if coin_toss:
            agent.remember(state, action, reward, next_state, p.game_over())
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, agent2)
        else:
            agent2.remember(state, action, reward, next_state, p.game_over())
            if len(agent2.memory) > batch_size:
                agent2.replay(batch_size, agent)

        state = next_state

if __name__ == '__main__':
    train()
