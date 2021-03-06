import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
import logging
import math

from sklearn.preprocessing import StandardScaler

from action import Action

####################################
# TODO: remove this after API update
from pandas_datareader.google.daily import GoogleDailyReader

@property
def url(self):
    return 'http://finance.google.com/finance/historical'

GoogleDailyReader.url = url
# remove ends
####################################

LOGGER = logging.getLogger(__name__)


class Environment:
    MIN_DEPOSIT_PCT = 0.7

    def __init__(self,
                 ticker_list,
                 initial_deposit=1000,
                 from_date=datetime.datetime(2007, 1, 1),
                 to_date=datetime.datetime(2017, 1, 1),
                 window=70,
                 min_days_to_hold=5,
                 max_days_to_hold=5,
                 days_step=10,
                 scaler=None):
        self.initial_deposit = initial_deposit
        self.window = window
        self.max_days_to_hold = max_days_to_hold

        def get(tickers, startdate, enddate):
            def data(ticker):
                return pdr.get_data_google(ticker, start=startdate, end=enddate)

            datas = map(data, tickers)
            return pd.concat(datas, keys=tickers, names=['Ticker', 'Date'])

        self.data = get(ticker_list, from_date, to_date)
        self.data.drop('Volume', inplace=True, axis=1)

        days_to_holds = np.arange(min_days_to_hold,
                                  max_days_to_hold + 1, days_step)

        self.main_ticker = ticker_list[0]
        self.action_space = [Action(self.main_ticker, act, days, 10)
                             for act in Action.acts
                             for days in days_to_holds]  # for ticker in ticker_list
        self.minimal_deposit = self.initial_deposit * Environment.MIN_DEPOSIT_PCT
        self.scaler = scaler
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
        self.deposit = self.initial_deposit
        self.max_current_index = len(self.scaled_data) - self.max_days_to_hold
        self.current_index = self.window
        self.actions = {}
        return self.state()

    def step(self, action_idx: int):
        if action_idx == -1:
            LOGGER.info('Skip action for {}'.format(self.data.loc[self.main_ticker].iloc[self.current_index - 1].name))
            self.current_index += 1
            next_state = self.state()
            return next_state, None, (self.max_current_index < self.current_index)

        action = self.action_space[action_idx]

        covered_df = self.future_data_for_action(action)
        on_date = covered_df.index[0]
        first_day_price = covered_df.iloc[0]['Open']
        last_day_price = covered_df.iloc[-1]['Close']

        if action.act == Action.BUY:
            reward = (last_day_price - first_day_price) / first_day_price
        elif action.act == Action.SELL:
            reward = (first_day_price - last_day_price) / first_day_price
        else:
            reward = 0

        if math.isnan(reward): # sometimes the first_day_price is NaN
            reward = 0

        self.current_index += 1  # action.days

        # store information for further inspectation
        invested_amount = self.deposit * action.percentage / 100
        deposit_reward = reward * invested_amount
        self.deposit += deposit_reward
        self.actions[on_date] = (action, reward, deposit_reward, first_day_price, last_day_price, invested_amount)

        next_state = self.state()
        done = self.deposit < self.minimal_deposit or \
               self.max_current_index < self.current_index
        return next_state, reward * 10000, done

    def future_data_for_action(self, action: Action):
        trade_day_index = self.current_index
        return self.data.loc[action.ticker].iloc[trade_day_index: trade_day_index + action.days]

    def state(self):
        return self.scaled_data.iloc[self.current_index - self.window: self.current_index]

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
