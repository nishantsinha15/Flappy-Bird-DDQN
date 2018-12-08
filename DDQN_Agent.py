import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd, Adam


class DeepQAgent:
    def __init__(self, state_size=8, action_size=2, replay_len=200000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=replay_len)
        self.gamma = 0.995
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 10 ** ((-1) * (10 ** (-5) ))  # todo learn how to decay epsilon properly
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if np.random.rand() <= self.epsilon:
            return np.random.randint(low=0, high=self.action_size)
        act_values = self.model.predict(state)[0]
        return np.argmax(act_values)  # returns the index of the greedy action

    def replay(self, batch_size):  # todo learn to fit in bigger batches to reduce redundant computations
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                best_action = -1
                val = -100000000
                temp_val = self.model.predict(next_state)[0]
                for a in range(self.action_size):
                    if temp_val[a] > val:
                        val = temp_val[a]
                        best_action = a

                target = (reward + self.gamma * self.target_model.predict(next_state)[0][best_action])  # Double Q learning
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name + "target")

    def save(self, name):
        self.model.save_weights(name)
        self.target_model.save_weights(name + "target")
