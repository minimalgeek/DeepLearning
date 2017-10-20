import pytest
import logging
from datacollector import Collector

LOGGER = logging.getLogger(__name__)

@pytest.fixture
def collector():
    LOGGER.info('=== fixture_init ===')
    return Collector()


def test_init(collector):
    LOGGER.info('=== test_init ===')
    assert collector is not None
