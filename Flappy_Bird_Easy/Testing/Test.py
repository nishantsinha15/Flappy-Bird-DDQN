from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random
import DDQN_Agent
import time
import matplotlib.pyplot as plt


def plot(data):
    plt.plot(data, 'r')
    plt.savefig('Testing95KTough.png')


def train(FRAME_TRAIN=1000005):
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=True)
    p.init()
    ob = game.getGameState()
    state = ob
    state = np.reshape(np.asarray(list(state.values())) , [1, 8])
    total_reward = 0
    agent = DDQN_Agent.DeepQAgent()
    agent.load("model95000")
    batch_size = 32
    my_timer = time.time()
    prev_frame = 0
    data = []
    for i in range(FRAME_TRAIN):
        if p.game_over():
            data.append(total_reward)
            p.reset_game()
            print("Total reward = {}, Frame = {}, epsilon = {}, frame/second = {}".format
                  (total_reward, i, agent.epsilon, (i - prev_frame)/(time.time() - my_timer)))
            total_reward = 0
            prev_frame = i
            my_timer = time.time()

        # get action from agent
        action = agent.act(state)

        # take action
        reward = p.act(p.getActionSet()[action])

        # making the reward space less sparse
        if reward < 0:
            reward = -1

        total_reward += reward
        next_state = np.asarray(list(game.getGameState().values()))
        next_state = np.reshape(next_state, [1, 8])

        state = next_state
        # time.sleep(0.3)
        # Plot socre
        if i % 1000 == 0:
            plot(data)

if __name__ == '__main__':
    train()