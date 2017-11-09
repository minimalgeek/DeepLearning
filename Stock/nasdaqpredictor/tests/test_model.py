import pytest
from datetime import datetime

from dataloader import DataLoader, DataTransformer
from nasdaqpredictor.model import Model


@pytest.mark.nn
def test_init(model: Model):
    assert model is not None


@pytest.mark.nn
def test_build_neural_net(model: Model):
    assert len(model.data) == 5
    assert model.model is not None
    assert len(model.model.layers) == 30
    assert model.scaler is not None


@pytest.mark.nn
def test_predict(model: Model):
    predicted = model.predict_one('AAPL', datetime(2015, 8, 21))
    assert predicted.shape == (1, 2)
