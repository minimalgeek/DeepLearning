import pytest
from datetime import datetime

from dataloader import DataLoader, DataTransformer
from nasdaqpredictor.model import Model


@pytest.fixture
def model():
    loader = DataLoader('/nasdaq_tickers_small.csv',
                        datetime(2000, 1, 1),
                        datetime(2017, 1, 1))
    loader.reload_all()

    transformer = DataTransformer(loader)
    transformer.transform()

    return Model(transformer, epochs=1)


def test_init(model: Model):
    assert model is not None


@pytest.mark.long
def test_fit_neural_net(model: Model):
    model.build_model_data()
    assert len(model.data) == 5
    model.build_neural_net()
    assert model.model is not None
    assert len(model.model.layers) == 22
    model.fit_neural_net()
