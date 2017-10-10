import pandas as pd
import datetime
import logging
import backtrader as bt
import backtrader.indicators as btind
import backtrader.feeds as btf
from sklearn.preprocessing import StandardScaler
from action import Action
from agent import Agent

LOGGER = logging.getLogger(__name__)


class EnvironmentStrategy(bt.Strategy):
    params = dict(period1=5,
                  period2=10,
                  period3=20,
                  environment=None)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        LOGGER.info('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.sma1 = btind.SimpleMovingAverage(self.datas[0], period=self.params.period1)
        self.sma2 = btind.SimpleMovingAverage(self.datas[0], period=self.params.period2)
        self.sma3 = btind.SimpleMovingAverage(self.datas[0], period=self.params.period3)

        self.order = None
        self.action = {}
        # read from params
        self.environment = self.params.environment
        self.agent = self.environment.agent

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
            self.log('% EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                     (direction,
                      order.executed.price,
                      order.executed.value,
                      order.executed.comm))

            if self.action['action'].act is not Action.CLOSE:
                self.agent.remember(self.current_state(),
                                    self.action['action'],  # supposed to be a BUY or SELL
                                    0,
                                    self.next_state(),
                                    False)  # self.environment.min_value > self.value)
                self.action = {}

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

        self.agent.remember(self.current_state(),
                            self.action['action'],  # supposed to be a CLOSE
                            trade.pnlcomm,
                            self.next_state(),
                            False)  # self.environment.min_value > self.value)
        self.action = {}

    def next(self):
        self.log('Close, %.2f' % self.data.close[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # it's not really next, but the current one in case of acting
        action_idx = self.params.agent.act(self.next_state())
        current_action = self.environment.action_space[action_idx]
        action_text = current_action.act

        if action_text == Action.HOLD:
            self.action['action'] = current_action
            return

        if self.position and action_text == Action.CLOSE:
            self.action['action'] = current_action
            self.close()
        else:
            if action_text == Action.BUY:
                self.action['action'] = current_action
                self.buy()
            elif action_text == Action.SELL:
                self.action['action'] = current_action
                self.sell()

    def current_state(self):
        def get(data):
            return data.get(size=self.environment.window, ago=-1)

        temp = get(self.sma1)
        return get(self.datas[0])

    def next_state(self):
        def get(data):
            return data.get(size=self.environment.window)

        return get(self.datas[0])


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

        self.preprocess_data()
        self.reset()

    def preprocess_data(self):
        data_unstacked = self.data.unstack(level=0)
        data_unstacked = data_unstacked.pct_change().fillna(0)

        rows = data_unstacked.shape[0]
        LOGGER.info('Data size: %d' % rows)

        if self.scaler is None:
            LOGGER.info('Create new scaler')
            self.scaler = StandardScaler()
            data_unstacked_scaled = self.scaler.fit_transform(data_unstacked)
        else:
            LOGGER.info('Use existing scaler')
            data_unstacked_scaled = self.scaler.transform(data_unstacked)
        self.scaled_data = pd.DataFrame(data=data_unstacked_scaled, columns=data_unstacked.columns,
                                        index=data_unstacked.index)

    def reset(self):
        self.cerebro = bt.Cerebro()
        self.data = btf.YahooFinanceData(dataname=self.ticker,
                                         from_date=self.from_date,
                                         to_date=self.to_date,
                                         adjusted=True,
                                         reverse=False)
        self.cerebro.adddata(self.data)
        self.cerebro.broker.setcash(100000.0)
        self.cerebro.broker.setcommission(commission=0.001)
        self.cerebro.addstrategy(EnvironmentStrategy, agent=self.agent, environment=self)
        self.cerebro.addsizer(bt.sizers.PercentSizer, stake=3)

        self.min_value = self.cerebro.broker.getvalue() * 0.7

        self.actions = {}
        return self.state()

    def set_agent(self, agent: Agent):
        self.agent = agent

    def action_size(self):
        return len(self.action_space)

    def state_size(self):
        return self.cerebro.strats[0]
