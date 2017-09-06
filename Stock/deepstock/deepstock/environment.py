import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
import logging
import math
from sklearn.preprocessing import StandardScaler

from .action import Action

LOGGER = logging.getLogger(__name__)


class Environment:
    TRAIN_DATA_PCT = 0.75
    MIN_DEPOSIT_PCT = 0.7
    SKIP_REWARD_MULTIPLIER = 0.01

    def __init__(self,
                 ticker_list,
                 initial_deposit=100000,
                 from_date=datetime.datetime(2007, 1, 1),
                 to_date=datetime.datetime(2017, 1, 1),
                 window=50,
                 min_days_to_hold=1,
                 max_days_to_hold=10,
                 on_train=True):
        self.initial_deposit = initial_deposit
        self.window = window
        self.on_train = on_train
        self.max_days_to_hold = max_days_to_hold

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

        self.train_X_df = pd.DataFrame(data=train_X_scaled, columns=train_X.columns, index=train_X.index)
        self.test_X_df = pd.DataFrame(data=test_X_scaled, columns=test_X.columns, index=test_X.index)

    def reset(self):
        self.deposit = self.initial_deposit
        self.current_index = self.window
        self.actions = {}
        if self.on_train:
            self.max_current_index = len(self.train_X_df) - self.max_days_to_hold
        else:
            self.max_current_index = len(self.test_X_df) - self.max_days_to_hold
        return self.state()

    def switch_to_test_data(self):
        self.on_train = False
        return self.reset()

    def step(self, action_idx: int):
        action = self.action_space[action_idx]
        # print('\t=> current action is: {} at {}'.format(action, self.data.index[self.current_index]))

        covered_df = self.original_data_for_action(action)
        on_date = covered_df.index[0]
        first_day_price = covered_df.iloc[0]['Close']
        last_day_price = covered_df.iloc[-1]['Close']

        if action.act == Action.BUY:
            reward = last_day_price - first_day_price
        elif action.act == Action.SELL:
            reward = first_day_price - last_day_price
        elif action.act == Action.SKIP:
            # let's say it's better not to spend money, instead of losing it
            reward = math.fabs(last_day_price - first_day_price) * Environment.SKIP_REWARD_MULTIPLIER

        self.current_index += action.days

        # store information for further inspectation
        self.deposit += reward * (self.deposit * action.percentage / 100)
        self.actions[on_date] = (action, reward)

        next_state = self.state()
        done = self.deposit < self.initial_deposit * Environment.MIN_DEPOSIT_PCT or \
               self.max_current_index < self.current_index
        return next_state, reward, done

    def original_data_for_action(self, action: Action):
        indexes = self.state().index
        return self.data.loc[action.ticker][indexes[0]:indexes[0 + action.days]]

    def state(self):
        if self.on_train:
            return self.train_X_df.iloc[self.current_index - self.window: self.current_index]
        else:
            return self.test_X_df.iloc[self.current_index - self.window: self.current_index]

    def state_size(self):
        return self.state().shape

    def action_size(self):
        return len(self.action_space)

    @staticmethod
    def shrink_df_for_ticker(df, ticker):
        idx = pd.IndexSlice
        df = df.loc[:, idx[:, ticker]]
        df.columns = df.columns.droplevel(1)
        return df
