import logging

from .agent import Agent
from .environment import Environment

LOGGER = logging.getLogger(__name__)

epochs = 3000  # number of games


def main():
    environment = Environment(['AAPL', 'IBM', 'GOOG'],
                              min_days_to_hold=3,
                              max_days_to_hold=7)
    agent = Agent(environment.state_size(),
                  environment.action_size(),
                  epochs)

    for i in range(epochs):
        state = environment.reset()
        done = False

        while not done:
            try:
                action = agent.act(state)
                next_state, reward, done = environment.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
            except ValueError as e:
                print(e)
        agent.decrease_epsilon()
        LOGGER.info('Balance for current game: %d', environment.deposit)

    agent.save('aapl-ibm-goog.h5')

    environment.on_train = False
    environment.reset()

if __name__ == '__main__':
    main()
