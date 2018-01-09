import pytest
from nasdaqpredictor.dataloader import DataLoader, DataTransformer
from datetime import datetime
import nasdaqpredictor
import logging
import os

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def loader():
    return DataLoader('/nasdaq_tickers.csv',
                      datetime(2000, 1, 1),
                      datetime(2017, 1, 1))


@pytest.fixture
def transformer(loader):
    return DataTransformer(loader)


@pytest.fixture
def small_transformer():
    return DataTransformer(DataLoader('/nasdaq_tickers_small.csv',
                                      datetime(2000, 1, 1),
                                      datetime(2017, 1, 1)))


def test_data_path():
    assert nasdaqpredictor.DATA_PATH is not None
    assert os.path.isdir(nasdaqpredictor.DATA_PATH)


# Loader

def test_loader_init(loader: DataLoader):
    assert len(loader.all_tickers) == 138
    assert loader.all_tickers.ticker.iloc[0] == 'AAL'
    assert loader.all_tickers.ticker.iloc[-1] == 'XRAY'


def test_construct_file_name():
    file_name = DataLoader.construct_file_name('AAPL')
    assert file_name == 'AAPL__2000_01_01__2017_01_01.csv'
    file_name = DataLoader.construct_file_name('IBM')
    assert file_name == 'IBM__2000_01_01__2017_01_01.csv'


def test_load_all(loader: DataLoader):
    loader.reload_all()
    assert len(loader.original_data_dict.values()) == 138


# Transformer

def test_transformer_init(transformer: DataTransformer):
    assert transformer is not None
    assert transformer.data_loader is not None


def test_shift(small_transformer: DataTransformer):
    small_transformer.transform()
    assert len(small_transformer.transformed_data_dict) == 5
    assert small_transformer.transformed_data_dict['AAL'].shape == (2833, 9)
