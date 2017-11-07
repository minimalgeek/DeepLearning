import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
from collections import defaultdict
import logging
import os

LOGGER = logging.getLogger(__name__)


class DataLoader:
    MAX_DOWNLOAD_ATTEMPT = 3

    def __init__(self,
                 all_tickers,
                 from_date: datetime,
                 to_date: datetime,
                 max_shift: int = 30,
                 return_shift_days: int = 5):
        self.from_date = from_date
        self.to_date = to_date
        self.max_shift = max_shift
        self.return_shift_days = return_shift_days

        self.all_tickers = all_tickers
        self.original_data_dict = None

    def reload_all(self):
        self.original_data_dict = defaultdict(lambda: None)
        self._attempt_count = 0
        self._load_for_tickers(self.all_tickers)

    def _load_for_tickers(self, tickers):
        LOGGER.info('Load tickers')

        skipped_tickers = []
        for ticker in tickers:
            path = self.construct_file_name(ticker)
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
            self._load_for_tickers(skipped_tickers)

    def construct_file_name(self, ticker):
        return '{}__{}__{}.csv'.format(ticker,
                                       self.from_date.strftime('%Y_%m_%d'),
                                       self.to_date.strftime('%Y_%m_%d'))

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
                 max_shift: int = 30,
                 return_shift_days: int = -5):
        self.data_loader: DataLoader = data_loader
        self.max_shift = max_shift
        self.return_shift_days = return_shift_days

        if self.data_loader.original_data_dict is None:
            self.data_loader.reload_all()

        self.transformed_data_dict = {}

    def transform(self):
        for ticker, data in self.data_loader.original_data_dict.items():
            for step in self.steps():
                try:
                    data = step(data)
                except Exception as e:
                    LOGGER.error(e)

            self.transformed_data_dict[ticker] = data

    def steps(self):
        yield self._set_index_column_if_necessary
        yield self._shift
        yield self._add_bulls
        yield self._add_gts
        yield self._add_return
        yield self._clean_structure

    def _clean_structure(self, data) -> pd.DataFrame:
        data = data.drop(['Open', 'High', 'Low', 'Close'], axis=1, level=1)
        data.columns = data.columns.droplevel()
        data.dropna(inplace=True)
        return data

    def _add_return(self, data) -> pd.DataFrame:
        shift_column = 'Shift ' + str(self.return_shift_days)

        shifted = data.iloc[:, [0, 1, 2, 3]].shift(self.return_shift_days)
        shifted.columns = pd.MultiIndex.from_product([[shift_column], ['Open', 'High', 'Low', 'Close']])
        data = pd.concat([data, shifted], axis=1)
        cls_5 = data[shift_column, 'Close']
        cls = data['Shift 0', 'Close']
        data['Shift 0', 'Return'] = 100 * (cls_5 - cls) / cls_5
        return data

    def _add_gts(self, data) -> pd.DataFrame:
        for i in range(0, self.max_shift):
            opn = data['Shift ' + str(i), 'Open']
            prv_cls = data['Shift ' + str(i + 1), 'Close']
            data['Shift ' + str(i), 'GT ' + str(i)] = 100 * (opn - prv_cls) / opn
        return data

    def _add_bulls(self, data) -> pd.DataFrame:
        for i in range(0, self.max_shift):
            cls = data['Shift ' + str(i), 'Close']
            opn = data['Shift ' + str(i), 'Open']
            data['Shift ' + str(i), 'Bull ' + str(i)] = 100 * (cls - opn) / cls
        return data

    def _shift(self, data) -> pd.DataFrame:
        data.columns = pd.MultiIndex.from_product([['Shift 0'], ['Open', 'High', 'Low', 'Close']])
        for i in range(1, self.max_shift + 1):
            shifted = data.iloc[:, [0, 1, 2, 3]].shift(i)
            shifted.columns = pd.MultiIndex.from_product([['Shift ' + str(i)], ['Open', 'High', 'Low', 'Close']])
            data = pd.concat([data, shifted], axis=1)
        return data

    def _set_index_column_if_necessary(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'Date' in data.columns:
            data.set_index('Date', inplace=True)
        return data
