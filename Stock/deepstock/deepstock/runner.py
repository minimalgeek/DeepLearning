import logging
from pprint import pprint
from datetime import datetime

from .agent import Agent
from .environment import Environment

WEIGHTS_FILE = 'model.h5'

LOGGER = logging.getLogger(__name__)

epochs = 1000  # number of games
tickers = ['AAPL', 'NVDA', 'GOOG', 'INTC']
min_days_to_hold = 2
max_days_to_hold = 5


def main(train):
    environment = Environment(tickers,
                              from_date=datetime(2007, 1, 1),
                              to_date=datetime(2013, 1, 1),
                              min_days_to_hold=min_days_to_hold,
                              max_days_to_hold=max_days_to_hold)
    agent = Agent(environment.state_size(),
                  environment.action_size(),
                  epochs=epochs,
                  replay_buffer=64,
                  memory_queue_length=64,
                  gamma=0.2)  # the future trade has ~zero influence

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
        agent.save(WEIGHTS_FILE)
    else:
        agent.load(WEIGHTS_FILE)

    # Test on!
    test_environment = Environment(tickers,
                                   from_date=datetime(2013, 1, 1),
                                   to_date=datetime(2017, 1, 1),
                                   min_days_to_hold=min_days_to_hold,
                                   max_days_to_hold=max_days_to_hold,
                                   scaler=environment.scaler)

    state = test_environment.reset()
    done = False

    while not done:
        action = agent.act(state, False)
        next_state, _, done = test_environment.step(action)
        state = next_state
    LOGGER.info('Balance for current game: %d', test_environment.deposit)
    pprint(test_environment.actions)


if __name__ == '__main__':
    main(train=True)
