import pytest
from datetime import datetime

from dataloader import DataLoader, DataTransformer
from nasdaqpredictor.model import Model


@pytest.fixture
def model():
    loader = DataLoader('/nasdaq_tickers_small.csv',
                        datetime(2000, 1, 1),
                        datetime(2017, 1, 1))
    transformer = DataTransformer(loader)
    return Model(transformer, epochs=1)


def test_init(model: Model):
    assert model is not None


@pytest.mark.nn
@pytest.mark.long
def test_build_neural_net(model: Model):
    model.build_model_data()
    assert len(model.data) == 5
    model.build_neural_net()
    assert model.model is not None
    assert len(model.model.layers) == 22
    assert model.scaler is not None


@pytest.mark.nn
@pytest.mark.long
def test_predict(model: Model):
    model.run_fit = False
    model.build_model_data()
    model.build_neural_net()
    predicted = model.predict_one('AAPL', '2015-08-21')
    assert predicted.shape == (1, 2)
