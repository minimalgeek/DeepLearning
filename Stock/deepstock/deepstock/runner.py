import logging
from pprint import pprint

from .agent import Agent
from .environment import Environment

LOGGER = logging.getLogger(__name__)

epochs = 50  # number of games


def main():
    environment = Environment(['AAPL', 'IBM', 'GOOG'],
                              min_days_to_hold=3,
                              max_days_to_hold=7)
    agent = Agent(environment.state_size(),
                  environment.action_size(),
                  epochs,
                  memory_queue_buffer=256,
                  gamma=0.3)

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
    agent.save('aapl-ibm-goog.h5')

    state = environment.switch_to_test_data()
    # TODO continue with testing


if __name__ == '__main__':
    main()
