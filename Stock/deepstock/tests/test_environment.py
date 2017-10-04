import pytest
import logging
import random
from datetime import datetime

from deepstock.environment import Environment

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def environment():
    return Environment(['AAPL', 'IBM', 'GOOG'], min_days_to_hold=3, max_days_to_hold=7)


def test_init(environment: Environment):
    assert len(environment.action_space) == 2


def test_reset(environment: Environment):
    environment.reset()
    assert environment.deposit == 1000
    assert environment.state().shape == (70, 12)
    assert len(environment.actions) == 0


def test_make_step(environment: Environment):
    assert environment.state().index[0] == datetime(2007, 1, 3)
    next_state, reward, done = environment.step(0)
    assert next_state.index[0].to_pydatetime() == datetime(2007, 1, 4)
    assert round(reward, 2) == -23.18
    assert not done
