import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
import logging
from sklearn.preprocessing import StandardScaler

from .action import Action

LOGGER = logging.getLogger(__name__)


class Environment:
    TRAIN_DATA_PCT = 0.75

    def __init__(self,
                 ticker_list,
                 initial_deposit=100000,
                 from_date=datetime.datetime(2007, 1, 1),
                 to_date=datetime.datetime(2017, 1, 1),
                 window=50,
                 min_days_to_hold=1,
                 max_days_to_hold=10):
        self.initial_deposit = initial_deposit
        self.window = window

        def get(tickers, startdate, enddate):
            def data(ticker):
                return pdr.get_data_google(ticker, start=startdate, end=enddate)

            datas = map(data, tickers)
            return pd.concat(datas, keys=tickers, names=['Ticker', 'Date'])

        self.data = get(ticker_list, from_date, to_date)

        days_to_holds = np.arange(min_days_to_hold,
                                  max_days_to_hold + 1)

        self.action_space = [Action(ticker, act, days, 3)
                             for act in Action.acts
                             for days in days_to_holds
                             for ticker in ticker_list]

        self.preprocess_data()
        self.reset()

    def preprocess_data(self):
        all_data_unstacked = self.data.unstack(level=0)
        all_data_unstacked = all_data_unstacked.pct_change().fillna(0)

        rows = all_data_unstacked.shape[0]
        train_size = int(rows * Environment.TRAIN_DATA_PCT)
        test_size = rows - train_size
        LOGGER.info('Data size: %d, train size: %d, test_size: %d' % (rows, train_size, test_size))

        scaler = StandardScaler()

        train_X = all_data_unstacked.iloc[0:train_size]
        test_X = all_data_unstacked.iloc[train_size:]

        train_X_scaled = scaler.fit_transform(train_X)
        test_X_scaled = scaler.transform(test_X)

        self.train_X_df = pd.DataFrame(data=train_X_scaled, columns=train_X.columns)
        self.test_X_df = pd.DataFrame(data=test_X_scaled, columns=test_X.columns)

    def reset(self):
        self.deposit = self.initial_deposit
        self.current_index = self.window
        self.actions = {}

        return self.state()

    def score(self):
        return self.deposit

    def enough_data_provided(self):
        return self.current_index + Environment.max_days_to_hold <= self.data_length

    def state(self):
        return self.data.iloc[self.current_index - self.window:self.current_index]['Close']

    # def price_state(self):
    #    return self.data.iloc[self.current_index - self.window:self.current_index]['Close']

    def state_size(self):
        return self.window

    def action_size(self):
        return len(self.action_space)

    def step(self, action_idx: int):
        action = self.action_space[action_idx]
        # print('\t=> current action is: {} at {}'.format(action, self.data.index[self.current_index]))

        df = self.data.iloc[self.current_index: self.current_index + action.days]
        on_date = df.index[0]
        first_day_price = df.iloc[0]['Close']
        last_day_price = df.iloc[-1]['Close']

        if action.act == BUY:
            reward = last_day_price - first_day_price
        elif action.act == SELL:
            reward = first_day_price - last_day_price
        elif action.act == SKIP:
            reward = 0

        self.actions[on_date] = (action, reward)

        self.current_index += action.days
        self.deposit += reward * (self.deposit * action.percentage / 100)

        if reward < 0:
            self.drawdowns += 1
        else:
            self.drawdowns = 0

        next_state = self.state()
        done = self.drawdowns > Environment.max_drawdowns
        _ = None
        return next_state, reward, done, _
