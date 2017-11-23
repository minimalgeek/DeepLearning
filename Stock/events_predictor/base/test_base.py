import pytest
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@pytest.fixture
def client(request):
    test_client = app.test_client()
    return test_client


def test_dummy(client):
    response = client.get('/')
    assert b'Hello, World!' in response.data
