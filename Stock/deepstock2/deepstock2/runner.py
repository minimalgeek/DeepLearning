import deepstock2
import logging
from datetime import datetime

from agent import Agent
from environment import Environment

LOGGER = logging.getLogger(__name__)

epochs = 2000  # number of games
ticker = 'AAPL'  # 'NVDA', 'GOOG', 'INTC'


def main(train):
    LOGGER.info('=== main started ===')
    environment = Environment(ticker,
                              from_date=datetime(2004, 1, 1),
                              to_date=datetime(2010, 1, 1))
    agent = Agent(environment.state_size(),
                  environment.action_size(),
                  epochs=epochs)

    environment.set_agent(agent)

    if train:
        for i in range(epochs):
            environment.reset()
            environment.run()
            agent.decrease_epsilon()
            LOGGER.info('#### {}/{} game finished ####\nBalance: {}'.format(
                str(i+1),
                epochs,
                environment.cerebro.broker.get_value()))

        agent.save(environment.ticker + '.h5')
    else:
        agent.load(environment.ticker + '.h5')

    # Test on!
    test_environment = Environment(ticker,
                                   from_date=datetime(2010, 1, 1),
                                   to_date=datetime(2013, 1, 1),
                                   scaler=environment.scaler)
    test_environment.set_agent(agent)

    test_environment.reset()
    test_environment.run()


if __name__ == '__main__':
    main(train=True)
