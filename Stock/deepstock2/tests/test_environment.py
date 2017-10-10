import pytest
import logging
import random
from datetime import datetime

from deepstock2.environment import Environment

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def environment():
    return Environment()


def test_init(environment: Environment):
    assert environment is not None
