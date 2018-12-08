from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random
import DDQN_Agent
import time


def train(FRAME_TRAIN=1000000):
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=False)
    p.init()
    ob = game.getGameState()
    state = ob
    state = np.reshape(np.asarray(list(state.values())) , [1, 8])
    total_reward = 0
    agent = DDQN_Agent.DeepQAgent()
    agent2 = DDQN_Agent.DeepQAgent()
    batch_size = 32
    my_timer = time.time()
    prev_frame = 0
    for i in range(FRAME_TRAIN):
        coin_toss = random.random() > 0.5

        if p.game_over():
            p.reset_game()
            print("Total reward = {}, Frame = {}, epsilon = {}, frame/second = {}".format(total_reward, i, agent.epsilon,
                                                                                          (i - prev_frame)/(time.time() - my_timer)))
            total_reward = 0
            prev_frame = i
            my_timer = time.time()

        # get action from agent
        if coin_toss:
            action = agent.act(state)
        else:
            action = agent2.act(state)

        # take action
        reward = p.act(p.getActionSet()[action])
        next_state = np.asarray(list(game.getGameState().values()))
        next_state = np.reshape(next_state, [1, 8])
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

        # save Model
        if i % 10000 == 0:
            agent.save('model')
            agent2.save('model2')

if __name__ == '__main__':
    train()