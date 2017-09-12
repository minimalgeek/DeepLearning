import zipline
from zipline.api import order, record, symbol
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# from .agent import Agent
# from .environment import Environment

tickers = ['AAPL', 'NVDA', 'GOOG', 'INTC']
start = datetime(2013, 1, 1)
end = datetime(2017, 1, 1)

LOGGER = logging.getLogger(__name__)

def initialize(context):
    LOGGER.info('Initialize')
    # environment = Environment(tickers,
    #                           from_date=datetime(2007, 1, 1),
    #                           to_date=datetime(2013, 1, 1))
    #
    # test_environment = Environment(tickers,
    #                                from_date=start,
    #                                to_date=end,
    #                                scaler=environment.scaler)
    #
    # agent = Agent(environment.state_size(),
    #               environment.action_size())
    # agent.load('model.h5')


def handle_data(context, data):
    aapl = symbol('AAPL')

    order(aapl, 10)
    record(AAPL=data.current(aapl, 'price'))


if __name__ == '__main__':
    perf = zipline.run_algorithm(start=start,
                                 end=end,
                                 initialize=initialize,
                                 capital_base=1000,
                                 handle_data=handle_data)

    LOGGER.info(perf)
    LOGGER.info('Creating plot')

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value')
    perf.AAPL.plot(ax=ax2)
    ax2.set_ylabel('AAPL stock price')
    fig.savefig('STOCK.png')
    plt.close()
