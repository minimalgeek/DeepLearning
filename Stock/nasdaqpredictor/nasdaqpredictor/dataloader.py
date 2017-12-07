import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
from collections import defaultdict
import nasdaqpredictor
import logging
import os

LOGGER = logging.getLogger(__name__)


class DataLoader:
    MAX_DOWNLOAD_ATTEMPT = 3

    def __init__(self,
                 ticker_file_name: str,
                 from_date: datetime,
                 to_date: datetime,
                 max_shift: int = 30,
                 return_shift_days: int = 5):
        self.from_date = from_date
        self.to_date = to_date
        self.max_shift = max_shift
        self.return_shift_days = return_shift_days

        self.all_tickers = pd.read_csv(os.path.dirname(__file__) + ticker_file_name)
        self.original_data_dict = None

    def reload_all(self):
        self.original_data_dict = defaultdict(lambda: None)
        self._attempt_count = 0
        self.load_for_tickers(self.all_tickers.ticker)

    def load_for_tickers(self, tickers):
        LOGGER.info('Load tickers')

        skipped_tickers = []
        for ticker in tickers:
            path = self.construct_full_path(ticker)
            if os.path.exists(path):
                data = pd.read_csv(path)
            else:
                data = self._download(skipped_tickers, ticker)
                if data is not None:
                    data.to_csv(path)

            if data is not None:
                self.original_data_dict[ticker] = data

        if len(skipped_tickers) > 0 and self._attempt_count < DataLoader.MAX_DOWNLOAD_ATTEMPT:
            self._attempt_count += 1
            LOGGER.info('Retry ({}) for skipped tickers: {}'.format(self._attempt_count, str(skipped_tickers)))
            self.load_for_tickers(skipped_tickers)

    def construct_file_name(self, ticker):
        return '{}__{}__{}.csv'.format(ticker,
                                       self.from_date.strftime('%Y_%m_%d'),
                                       self.to_date.strftime('%Y_%m_%d'))

    def construct_full_path(self, ticker):
        return os.path.abspath(os.path.join(nasdaqpredictor.DATA_PATH,
                                            self.construct_file_name(ticker)))

    def _download(self, skipped_tickers, ticker):
        try:
            original_data = pdr.get_data_yahoo(ticker, self.from_date, self.to_date)
            original_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)
            return original_data
        except Exception as e:
            LOGGER.error(e)
            skipped_tickers.append(ticker)
            return None
        else:
            LOGGER.info(ticker + ' downloaded successfully')


class DataTransformer:
    def __init__(self,
                 data_loader: DataLoader,
                 return_shift_days: int = 3):
        self.data_loader: DataLoader = data_loader
        self.return_shift_days = return_shift_days
        self.transformed_data_dict = None

    def transform(self):
        self.transformed_data_dict = {}
        self.data_loader.reload_all()

        for ticker, data in self.data_loader.original_data_dict.items():
            for step in self.steps():
                try:
                    data = step(data)
                except Exception as e:
                    LOGGER.error(e)

            self.transformed_data_dict[ticker] = data
        # We no longer need this
        self.data_loader.original_data_dict = None

    def steps(self):
        yield self._set_index_column_if_necessary
        yield self._append_new_features
        yield self._create_full_dataset
        yield self._clean_structure

    def _set_index_column_if_necessary(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'Date' in data.columns:
            data.set_index('Date', inplace=True)
        return data

    def _append_new_features(self, data: pd.DataFrame) -> pd.DataFrame:
        def feature(data, first_col, second_col, base_col):
            return (data[first_col] - data[second_col]) / data[base_col]
        data['OC diff'] = feature(data, 'Open', 'Close', 'Close')
        data['HL diff'] = feature(data, 'High', 'Low', 'Close')
        data['OL diff'] = feature(data, 'Open', 'Low', 'Close')
        data['CH diff'] = feature(data, 'Close', 'High', 'Close')
        data['Return'] = 100 * data['Close'].pct_change(self.return_shift_days).shift(-self.return_shift_days)
        return data

    def _create_full_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        #full = pd.concat((data.iloc[:, 0:4].pct_change(), data.iloc[:, 4:8], data['Return']), axis=1)
        full = pd.concat((data.iloc[:, 4:8], data['Return']), axis=1)
        # return full.iloc[1:-self.return_shift_days]
        return full.iloc[:-self.return_shift_days]

    def _clean_structure(self, data) -> pd.DataFrame:
        return data.replace([np.inf, -np.inf, np.NaN, np.NAN], 0.0)
