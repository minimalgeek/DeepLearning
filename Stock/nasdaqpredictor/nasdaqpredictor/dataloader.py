import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
from collections import defaultdict
from functools import reduce
import nasdaqpredictor
import logging
import os

LOGGER = logging.getLogger(__name__)


class DataLoader:
    MAX_DOWNLOAD_ATTEMPT = 3

    def __init__(self,
                 ticker_file_name: str,
                 return_shift_days: int = 5):
        self.return_shift_days = return_shift_days

        self.original_data_dict = None
        self._attempt_count = 0
        self.skipped_tickers = pd.DataFrame()

        self.rows = pd.read_csv(ticker_file_name, parse_dates=['from', 'to'])
        self.rows['from'].fillna(datetime(1990, 1, 1), inplace=True)
        self.rows['to'].fillna(datetime(2017, 12, 1), inplace=True)

    def reload_all(self):
        self.original_data_dict = defaultdict(lambda: None)
        self.load_for_tickers(self.rows)

    def load_for_tickers(self, rows):
        LOGGER.info('Load tickers')

        for index, row in rows.iterrows():
            path = DataLoader.construct_full_path(row)
            if os.path.exists(path):
                data = pd.read_csv(path)
            else:
                data = self._download(row)
                if data is not None:
                    data.to_csv(path)

            if data is not None:
                self.original_data_dict[tuple(row)] = data

        if len(self.skipped_tickers) > 0 and self._attempt_count < DataLoader.MAX_DOWNLOAD_ATTEMPT:
            self._attempt_count += 1
            LOGGER.info('Retry ({}) for skipped tickers: {}'.format(self._attempt_count, str(self.skipped_tickers)))
            self.skipped_tickers = pd.DataFrame()
            self.load_for_tickers(self.skipped_tickers)

    def construct_file_name(row):
        return '{}__{}__{}.csv'.format(row['ticker'],
                                       row['from'].strftime('%Y_%m_%d'),
                                       row['to'].strftime('%Y_%m_%d'))

    def construct_full_path(ticker):
        return os.path.abspath(os.path.join(nasdaqpredictor.DATA_PATH,
                                            DataLoader.construct_file_name(ticker)))

    def _download(self, row):
        try:
            original_data: pd.DataFrame = pdr.get_data_google(row['ticker'],
                                                              row['from'].to_datetime(),
                                                              row['to'].to_datetime())
            original_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True, errors='ignore')
            return original_data
        except Exception as e:
            LOGGER.error(e)
            self.skipped_tickers = self.skipped_tickers.append(row)
            return None


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
        # full = pd.concat((data.iloc[:, 0:4].pct_change(), data.iloc[:, 4:8], data['Return']), axis=1)
        full = pd.concat((data.iloc[:, 4:8], data['Return']), axis=1)
        # return full.iloc[1:-self.return_shift_days]
        return full.iloc[:-self.return_shift_days]

    def _clean_structure(self, data) -> pd.DataFrame:
        return data.replace([np.inf, -np.inf, np.NaN, np.NAN], 0.0)
