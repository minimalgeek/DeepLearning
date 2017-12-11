from datetime import datetime

import pytest

from dataloader import DataLoader, DataTransformer
from model import Model


@pytest.mark.nn
@pytest.fixture(scope='session')
def model() -> Model:
    loader = DataLoader('/nasdaq_tickers_small.csv',
                        datetime(2000, 1, 1),
                        datetime(2017, 1, 1))
    transformer = DataTransformer(loader)
    mod = Model(transformer,
                epochs=1,
                dev_date=datetime(2014, 1, 1),
                run_fit=False)
    mod.build_model_data()
    mod.build_neural_net()
    return mod
