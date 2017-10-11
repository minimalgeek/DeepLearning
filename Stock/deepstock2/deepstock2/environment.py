import pandas as pd
import numpy as np
import datetime
import logging
import backtrader as bt
import backtrader.indicators as btind
import backtrader.feeds as btf
import pandas_datareader as pdr
from sklearn.preprocessing import StandardScaler
from action import Action
from agent import Agent

LOGGER = logging.getLogger(__name__)


class EnvironmentStrategy(bt.Strategy):
    params = dict(environment=None)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        LOGGER.info('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.order = None
        self.previous_action: StateToRemember = None
        self.previous_hold_action: StateToRemember = None
        # read from params
        self.environment = self.params.environment
        self.agent = self.environment.agent

        self.sma = btind.SimpleMovingAverage(self.datas[0], period=self.environment.window + 2)  # meh

    def notify_cashvalue(self, cash, value):
        self.value = value

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            direction = 'BUY' if order.isbuy() else 'SELL'
            self.log('%s EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                     (direction,
                      order.executed.price,
                      order.executed.value,
                      order.executed.comm))

            if self.previous_action.action.act in [Action.BUY, Action.SELL]:
                self.previous_action.refresh_with_state()
                self.agent.remember(*self.previous_action.to_tuple())

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

        if self.previous_action.action.act == Action.CLOSE:
            self.previous_action.refresh_with_state(trade.pnlcomm)
            self.agent.remember(*self.previous_action.to_tuple())

    def next(self):

        if self.previous_hold_action:
            self.previous_hold_action.refresh_with_state()
            self.agent.remember(*self.previous_hold_action.to_tuple())
            self.previous_hold_action = None

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # it's not really next, but the current one in case of acting
        action_idx = self.agent.act(self.next_state())
        current_action = self.environment.action_space[action_idx]
        action_text = current_action.act

        self.previous_action = StateToRemember(current_action, action_idx, self)
        if action_text == Action.HOLD:
            self.previous_hold_action = StateToRemember(current_action, action_idx, self)
        elif self.position and action_text == Action.CLOSE:
            self.close()
        elif not self.position:
            if action_text == Action.BUY:
                self.buy()
            elif action_text == Action.SELL:
                self.sell()
        else:
            self.previous_action = None

    def _shift(self, data, value=0):
        return data.get(size=self.environment.window+1, ago=value)

    def _create_pct_change_dataframe(self, value=0):
        frame = pd.DataFrame(
            np.array([self._shift(self.data_open, value),
                      self._shift(self.data_high, value),
                      self._shift(self.data_low, value),
                      self._shift(self.data_close, value)]).T,
            columns=['Open', 'High', 'Low', 'Close'])
        return frame.pct_change().fillna(0).iloc[1:]

    def current_state(self):
        current_frame = self._create_pct_change_dataframe(-1)
        return self.environment.scaler.transform(current_frame)

    def next_state(self):
        next_frame = self._create_pct_change_dataframe()
        return self.environment.scaler.transform(next_frame)

    def is_terminal_state(self):
        return False  # self.environment.min_value > self.value


class StateToRemember:
    def __init__(self, action, action_idx, strategy: EnvironmentStrategy):
        self.state = None
        self.next_state = None
        self.action = action
        self.action_idx = action_idx
        self.reward = 0
        self.done = False
        self.strategy = strategy

    def refresh_with_state(self, reward=0):
        self.state = self.strategy.current_state()
        self.next_state = self.strategy.next_state()
        self.done = self.strategy.is_terminal_state()
        self.reward = reward

    def to_tuple(self):
        return (self.state, self.action_idx, self.reward, self.next_state, self.done)


class Environment:
    def __init__(self,
                 ticker,
                 from_date=datetime.datetime(2007, 1, 1),
                 to_date=datetime.datetime(2017, 1, 1),
                 window=50,
                 scaler=None):
        self.ticker = ticker
        self.from_date = from_date
        self.to_date = to_date
        self.window = window
        self.scaler = scaler

        self.action_space = [Action(self.ticker, act) for act in Action.acts]

    def _setup_scaler(self):
        flat_data = self.pandas_data.pct_change().fillna(0)
        if self.scaler is None:
            LOGGER.info('Create new scaler')
            self.scaler = StandardScaler()
            self.scaler.fit(flat_data)
        else:
            LOGGER.info('Use existing scaler')

    def reset(self):
        self.cerebro = bt.Cerebro()
        # self.data = btf.YahooFinanceData(dataname=self.ticker,
        #                                  from_date=self.from_date,
        #                                  to_date=self.to_date,
        #                                  adjusted=True,
        #                                  reverse=False)
        self.pandas_data = pdr.get_data_google(self.ticker, start=self.from_date, end=self.to_date)
        self.pandas_data.drop('Volume', inplace=True, axis=1)
        self.data = bt.feeds.PandasData(dataname=self.pandas_data)
        self.cerebro.adddata(self.data)
        self.cerebro.broker.setcash(100000.0)
        self.cerebro.broker.setcommission(commission=0.001)
        self.cerebro.addstrategy(EnvironmentStrategy, environment=self)
        self.cerebro.addsizer(bt.sizers.PercentSizer, percents=5)

        self.min_value = self.cerebro.broker.getvalue() * 0.7
        self._setup_scaler()

    def run(self):
        self.cerebro.run()

    def set_agent(self, agent: Agent):
        self.agent = agent

    def action_size(self):
        return len(self.action_space)

    def state_size(self):
        return (self.window, 4)
