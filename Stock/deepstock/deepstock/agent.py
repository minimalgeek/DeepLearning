from collections import deque

import numpy as np
import pandas as pd
import logging
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.losses import mean_squared_error

LOGGER = logging.getLogger(__name__)


class Agent:
    def __init__(self,
                 input_shape,
                 action_size,
                 epochs,
                 batch_size=8,
                 layer_decrease_multiplier=0.8,
                 min_epsilon=0.1,
                 gamma=0.2,
                 memory_buffer=32,
                 memory_queue_buffer=128):
        self.input_shape = input_shape
        self.action_size = action_size
        self.epochs = epochs

        self.batch_size = batch_size
        self.layer_decrease_multiplier = layer_decrease_multiplier
        self.min_epsilon = min_epsilon
        self.gamma = gamma # how much should we look into the future predicitons

        self.epsilon = 1
        self.memory_buffer = memory_buffer
        self.memory = deque(maxlen=memory_queue_buffer)
        self.replay_index = 0

        self._build_model()

    def _build_model(self):
        first_layer_size = int(self.input_shape[0] * self.layer_decrease_multiplier)
        second_layer_size = int(first_layer_size * self.layer_decrease_multiplier)
        third_layer_size = int(second_layer_size * self.layer_decrease_multiplier)

        model = Sequential()
        model.add(Dense(first_layer_size, input_shape=self.input_shape))
        model.add(Flatten())
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

    def decrease_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= (1 / self.epochs)

    def act(self, state):
        if random.random() < self.epsilon:  # choose random action
            action = np.random.randint(0, 4)
        else:  # choose best action from Q(s,a) values
            q_val = self.model.predict(state, batch_size=1)
            action = (np.argmax(q_val))
        return action

    def remember(self, state, action, reward, next_state, done):
        action_tuple = (state, action, reward, next_state, done)
        self.memory.append(action_tuple)
        if len(self.memory) >= self.memory_buffer:
            self.replay()
            # state = new_state ???

    def replay(self):
        # randomly sample our experience replay memory
        mini_batch = random.sample(self.memory, self.memory_buffer)
        LOGGER.info('Experience replay for {} memories'.format(len(mini_batch)))
        x_train = []
        y_train = []
        for mem in mini_batch:
            state, action, reward, next_state, done = mem

            state_vals = np.expand_dims(state.values, axis=0)  # (50, 15) -> (1, 50, 15)
            next_state_vals = np.expand_dims(next_state.values, axis=0)  # (50, 15) -> (1, 50, 15)

            old_q = self.model.predict(state_vals, batch_size=1)
            new_q = self.model.predict(next_state_vals, batch_size=1)
            max_q = np.max(new_q)
            y = np.zeros((1, self.action_size))
            y[:] = old_q[:]
            if not done:
                update = (reward + (self.gamma * max_q))
            else:
                update = reward
            y[0][action] = update
            x_train.append(state.values)
            y_train.append(y[0])

        x_train = np.array(x_train)  # (32, 50, 15)
        y_train = np.array(y_train)  # (32, 90)
        self.model.fit(x_train,
                       y_train,
                       batch_size=self.batch_size,
                       epochs=1,
                       verbose=1)

    def load(self, name):
        LOGGER.info("Load '%s' model", name)
        self.model.load_weights(name)

    def save(self, name):
        LOGGER.info("Save '%s' model", name)
        self.model.save_weights(name)
