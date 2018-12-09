from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random
import DDQN_Agent
import time
import matplotlib.pyplot as plt

def plot(data, data2):
    plt.plot(data, 'r', data2, 'g')
    plt.savefig('Chart.png')

def train(FRAME_TRAIN=1000005):
    game = FlappyBird(pipe_gap = 250)
    p = PLE(game, fps=30, display_screen=True)
    p.init()
    ob = game.getGameState()
    state = ob
    state = np.reshape(np.asarray(list(state.values())) , [1, 8])
    total_reward = 0
    survival = 0
    agent = DDQN_Agent.DeepQAgent()
    batch_size = 32
    my_timer = time.time()
    prev_frame = 0
    data = []
    data2 = []
    for i in range(FRAME_TRAIN):
        if p.game_over():
            data.append(total_reward)
            data2.append(survival)
            p.reset_game()
            print("Total reward = {}, Frame = {}, epsilon = {}, frame/second = {}, survival = {}".format(total_reward, i, agent.epsilon,
                                                                                          (i - prev_frame)/(time.time() - my_timer)
                                                                                                         , survival))
            survival = 0
            total_reward = 0
            prev_frame = i
            my_timer = time.time()

        # get action from agent
        action = agent.act(state)

        # take action
        reward = p.act(p.getActionSet()[action])
        total_reward += reward

        # making the reward space less sparse
        if not p.game_over():
            reward += 0.01

        next_state = np.asarray(list(game.getGameState().values()))
        next_state = np.reshape(next_state, [1, 8])
        survival += reward
        # remember and replay
        agent.remember(state, action, reward, next_state, p.game_over())
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        state = next_state

        # save Model
        if i % 5000 == 0:
            print("Updating weights")
            agent.save('model')
            agent.target_model.set_weights(agent.model.get_weights())

        if i % 1000 == 0:
            plot(data, data2)

if __name__ == '__main__':
    train()