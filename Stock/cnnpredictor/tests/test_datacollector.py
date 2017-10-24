import pytest
import logging
from datetime import datetime
from functools import wraps

from datacollector import Collector

LOGGER = logging.getLogger(__name__)


def log_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        LOGGER.info('-> ' + func.__name__)
        func(*args, **kwargs)
        LOGGER.info('<- ' + func.__name__)

    return wrapper


@pytest.fixture
def collector():
    return Collector(datetime(2004, 1, 1), datetime(2017, 1, 1), 'AAPL')


@log_it
def test_init(collector):
    # LOGGER.info('=== test_init ===')
    assert collector is not None
    assert collector.start_date < datetime.now() and collector.end_date < datetime.now()
    assert collector.ticker == 'AAPL'


def test_collect(collector: Collector):
    collector.collect()
