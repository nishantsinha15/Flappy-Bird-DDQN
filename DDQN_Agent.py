import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd,Adam
import matplotlib.pyplot as plt


EPISODES = 1000
# actions_space = get_actions()


def plot(data):
    x=[]
    y=[]
    for i,j in data:
        x.append(i)
        y.append(j)
    plt.plot(x,y)
    plt.savefig(file_name + '.png')


class DeepQAgent:
    def __init__(self, state_size = 8, action_size = 2, replay_len = 200000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=replay_len)
        self.gamma = 0.995
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 # todo learn how to decay epsilon properly
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()


    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size))  # changed this from linear
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(low=0, high=self.action_size)
        act_values = self.model.predict(state.get_input_layer())[0]
        return np.argmax(act_values) # returns the index of the action

    def replay(self, batch_size, agent2):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                best_action = -1
                val = -100000000
                temp_val = self.model.predict(next_state.get_input_layer())[0]
                for a in self.action_space:
                    if temp_val[a] > val:
                        val = temp_val[a]
                        best_action = a
                '''check understading '''
                target = (reward + self.gamma * agent2.model.predict(next_state.get_input_layer())[0][best_action])  # Double Q learning
            target_f = self.model.predict(state.get_input_layer())
            target_f[0][action] = target
            '''check understading '''
            self.model.fit(state.get_input_layer(), target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)