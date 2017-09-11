import zipline
from zipline.api import order, record, symbol
from datetime import datetime

from .agent import Agent
from .environment import Environment

tickers = ['AAPL', 'NVDA', 'GOOG', 'INTC']
start = datetime(2013, 1, 1)
end = datetime(2017, 1, 1)

def initialize(context):
    environment = Environment(tickers,
                              from_date=datetime(2007, 1, 1),
                              to_date=datetime(2013, 1, 1))

    test_environment = Environment(tickers,
                                   from_date=start,
                                   to_date=end,
                                   scaler=environment.scaler)

    agent = Agent(environment.state_size(),
                  environment.action_size())
    agent.load('model.h5')


def handle_data(context, data):
    order(symbol('AAPL'), 10)
    record(AAPL=data.current(symbol('AAPL'), 'price'))


if __name__ == '__main__':
    zipline.run_algorithm(start=start,
                          end=end,
                          initialize=initialize,
                          capital_base=1000,
                          handle_data=handle_data)

