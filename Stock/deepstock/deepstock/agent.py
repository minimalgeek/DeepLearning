from collections import deque

import numpy as np
import logging
import random

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.losses import mean_squared_error

LOGGER = logging.getLogger(__name__)


class Agent:
    def __init__(self,
                 input_shape,
                 action_size,
                 epochs,
                 layer_decrease_multiplier=0.8,
                 min_epsilon=0.05,
                 gamma=0.05,
                 replay_buffer=32,
                 memory_queue_length=128,
                 learning_rate=0.01):
        self.input_shape = input_shape
        self.action_size = action_size
        self.epochs = epochs

        self.layer_decrease_multiplier = layer_decrease_multiplier
        self.min_epsilon = min_epsilon
        self.gamma = gamma  # how much should we look into the future predicitons

        self.epsilon = 1
        self.replay_buffer = replay_buffer
        self.memory = deque(maxlen=memory_queue_length)
        self.learning_rate = learning_rate
        self.replay_index = 0

        self._build_model()

    def _build_model(self):
        first_layer_size = int(self.input_shape[0] *
                               self.layer_decrease_multiplier)
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
        model.add(Activation('linear'))

        adam = Adam(lr=self.learning_rate)
        model.compile(loss=mean_squared_error, optimizer=adam)
        LOGGER.info('Model successfully built with hidden layers: {}, {}, {}'
                    .format(first_layer_size,
                            second_layer_size,
                            third_layer_size))
        self.model = model

    def decrease_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= (1 / self.epochs)
        LOGGER.info('New epsilon is {}'.format(self.epsilon))

    def act(self, state, with_random=True):
        if with_random and random.random() < self.epsilon:  # choose random action
            action = np.random.randint(0, self.action_size)
        else:  # choose best action from Q(s,a) values
            q_val = self.model.predict(Agent.df_to_array(state), batch_size=1)
            action = np.argmax(q_val[0])
            LOGGER.info('\t\tChoosen action index: {}'.format(action))
        return action

    def remember(self, state, action, reward, next_state, done):
        action_tuple = (state, action, reward, next_state, done)
        self.memory.append(action_tuple)
        self.replay_index += 1
        if self.replay_index >= self.replay_buffer:
            self.replay()
            self.replay_index = 0
            # state = new_state ???

    def replay(self):
        # randomly sample our experience replay memory
        mini_batch = random.sample(self.memory, self.replay_buffer)
        LOGGER.info('Experience replay for {} memories'.format(len(mini_batch)))
        x_train = []
        y_train = []
        for mem in mini_batch:
            state, action, reward, next_state, done = mem

            state_vals = Agent.df_to_array(state)
            next_state_vals = Agent.df_to_array(next_state)

            old_q = self.model.predict(state_vals, batch_size=1)
            new_q = self.model.predict(next_state_vals, batch_size=1)
            max_q = np.max(new_q)
            update = reward
            if not done:
                update += self.gamma * max_q
            old_q[0][action] = update
            x_train.append(state.values)
            y_train.append(old_q[0])

        x_train = np.array(x_train)  # (32, 50, 15)
        y_train = np.array(y_train)  # (32, 90)
        self.model.fit(x_train,
                       y_train,
                       epochs=1,  # self.replay_buffer
                       verbose=0)

    def load(self, name):
        LOGGER.info("Load '%s' model", name)
        self.model.load_weights(name)

    def save(self, name):
        LOGGER.info("Save '%s' model", name)
        self.model.save_weights(name)

    @staticmethod
    def df_to_array(df):
        return np.expand_dims(df.values, axis=0)  # (50, 15) -> (1, 50, 15)
