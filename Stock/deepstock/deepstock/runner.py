import logging
from pprint import pprint

from .agent import Agent
from .environment import Environment

AAPL_IBM_GOOG = 'aapl-ibm-goog.h5'

LOGGER = logging.getLogger(__name__)

epochs = 20  # number of games


def main(train):
    environment = Environment(['AAPL', 'IBM', 'GOOG'],
                              initial_deposit=1000,
                              min_days_to_hold=3,
                              max_days_to_hold=7)
    agent = Agent(environment.state_size(),
                  environment.action_size(),
                  epochs,
                  memory_queue_buffer=256,
                  gamma=0.3)

    if train:
        for i in range(epochs):
            state = environment.reset()
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done = environment.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
            agent.decrease_epsilon()
            LOGGER.info('Balance for current game: %d', environment.deposit)

        pprint(environment.actions)
        agent.save(AAPL_IBM_GOOG)
    else:
        agent.load(AAPL_IBM_GOOG)

    # Test on!
    state = environment.switch_to_test_data()
    done = False

    while not done:
        action = agent.act(state, False)
        next_state, reward, done = environment.step(action)
        state = next_state
    LOGGER.info('Balance for current game: %d', environment.deposit)
    pprint(environment.actions)

if __name__ == '__main__':
    main(train=False)
