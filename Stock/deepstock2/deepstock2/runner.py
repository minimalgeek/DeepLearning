import logging
from pprint import pprint
from datetime import datetime
import csv

import deepstock2
from agent import Agent
from environment import Environment

LOGGER = logging.getLogger(__name__)

epochs = 1000  # number of games
ticker = 'AAPL'  # 'NVDA', 'GOOG', 'INTC'


def main(train, action_bias=0):
    environment = Environment(ticker,
                              from_date=datetime(2004, 1, 1),
                              to_date=datetime(2010, 1, 1))
    agent = Agent(environment.state_size(),
                  environment.action_size(),
                  epochs=epochs,
                  gamma=0.2,
                  replay_buffer=64,
                  memory_queue_length=32)

    environment.set_agent(agent)

    if train:
        for i in range(epochs):
            state = environment.reset()
            done = False

            agent.decrease_epsilon()
            LOGGER.info('Balance for current game: %d', environment.deposit)

        pprint(environment.actions)
        agent.save(environment.main_ticker + '.h5')
    else:
        agent.load(environment.main_ticker + '.h5')

    # Test on!
    test_environment = Environment(ticker,
                                   from_date=datetime(2010, 1, 1),
                                   to_date=datetime(2013, 1, 1),
                                   scaler=environment.scaler)

    state = test_environment.reset()
    done = False

    while not done:
        action = agent.act(state, False, action_bias)
        next_state, _, done = test_environment.step(action)
        state = next_state


if __name__ == '__main__':
    main(train=False, action_bias=0)
