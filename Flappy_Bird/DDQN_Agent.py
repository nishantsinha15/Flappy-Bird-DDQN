import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd, Adam
from keras import backend as K
from keras.callbacks import TensorBoard


class DeepQAgent:
    def __init__(self, state_size=8, action_size=2, replay_len=20000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=replay_len)
        self.gamma = 0.995
        self.epsilon = 0.75
        self.epsilon_min = 0.1
        self.epsilon_decay = 10 ** ((-1) * (10 ** (-5) ))
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                                  write_graph=True, write_images=True)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss=self.huber_loss , optimizer=Adam(lr=self.learning_rate))
        return model

    def huber_loss(self, a, b, in_keras=True):
        error = a - b
        quadratic_term = error * error / 2
        linear_term = abs(error) - 1 / 2
        use_linear_term = (abs(error) > 1.0)
        if in_keras:
            # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
            use_linear_term = K.cast(use_linear_term, 'float32')

        return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if np.random.rand() <= self.epsilon:
            if np.random.rand() <= 0.10:
                return 0
            else:
                return 1
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
