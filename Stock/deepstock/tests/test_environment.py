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
    # buy3...buy7,sell3...sell7,skip3...skip7 for AAPL, IBM, GOOG
    # 15 * 3
    assert len(environment.action_space) == 45
    assert environment.train_X_df.shape == (1888, 15)
    assert environment.test_X_df.shape == (630, 15)


def test_reset(environment: Environment):
    environment.reset()
    assert environment.deposit == 100000
    assert environment.state().shape == (50, 15)
    assert len(environment.actions) == 0


def test_make_step(environment: Environment):
    assert environment.state().index[0] == datetime(2007, 1, 3)
    next_state, reward, done = environment.step(0)  # AAPL BUY 3 days
    assert next_state.index[0] == datetime(2007, 1, 8)
    assert round(reward, 2) == 0.24  # 12.21 - 11.97
    assert not done
