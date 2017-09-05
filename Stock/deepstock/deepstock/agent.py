import numpy as np
import pandas as pd
import logging
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.losses import mean_squared_error

LOGGER = logging.getLogger(__name__)


class Agent:
    def __init__(self,
                 input_shape,
                 action_size,
                 epochs,
                 batch_size=16,
                 layer_decrease_multiplier=0.8,
                 min_epsilon=0.1):
        self.input_shape = input_shape
        self.action_size = action_size
        self.epochs = epochs

        self.batch_size = batch_size
        self.layer_decrease_multiplier = layer_decrease_multiplier
        self.min_epsilon = min_epsilon

        self.epsilon = 1
        self.memory = []

        self._build_model()

    def _build_model(self):
        first_layer_size = int(self.input_shape[0] * self.layer_decrease_multiplier)
        second_layer_size = int(first_layer_size * self.layer_decrease_multiplier)
        third_layer_size = int(second_layer_size * self.layer_decrease_multiplier)

        model = Sequential()
        model.add(Dense(first_layer_size, input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))

        model.add(Dense(second_layer_size))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))

        model.add(Dense(third_layer_size))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))

        model.add(Dense(self.action_size))
        model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

        adam = Adam()
        model.compile(loss=mean_squared_error, optimizer=adam)
        LOGGER.info('Model successfully built with hidden layers: {}, {}, {}'
                    .format(first_layer_size,
                            second_layer_size,
                            third_layer_size))
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def decrease_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= (1 / self.epochs)

    def act(self, state):
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        # act_values = self.model.predict(state)
        # return np.argmax(act_values[0])  # returns action

        if random.random() < self.epsilon:  # choose random action
            action = np.random.randint(0, 4)
        else:  # choose best action from Q(s,a) values
            q_val = self.model.predict(state, batch_size=1)
            action = (np.argmax(q_val))
        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
