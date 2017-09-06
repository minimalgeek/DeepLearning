import pytest
import logging
from datetime import datetime
from deepstock.agent import Agent
from deepstock.environment import Environment

LOGGER = logging.getLogger(__name__)

environment = Environment(['AAPL', 'IBM', 'GOOG'],
                          from_date=datetime(2010, 1, 1),
                          to_date=datetime(2011, 1, 1))


@pytest.fixture
def agent():
    return Agent(environment.state_size(), environment.action_size(), 1000)


def test_agent_created(agent: Agent):
    assert agent is not None


def test_act(agent: Agent):
    action = agent.act(environment.state())
    assert type(action) == int
    assert action < environment.action_size()


def test_remember_smoke(agent: Agent):
    for i in range(40):
        agent.remember(environment.state(), 0, 0.3, environment.state(), False)
