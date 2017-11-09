import pytest
import os
from datetime import datetime
import nasdaqpredictor as nas

from dataloader import DataLoader, DataTransformer
from nasdaqpredictor.model import Model
from prediction_exporter import PredictionExporter


@pytest.fixture
def exporter(model):
    return PredictionExporter(model)


@pytest.mark.nn
def test_ticker(exporter: PredictionExporter):
    assert len(exporter.tickers) == 5
    assert exporter.tickers.ticker[0] == 'AAL'


@pytest.mark.nn
def test_export_to_csv(exporter: PredictionExporter):
    exporter.export_to_csv()
    assert os.path.exists(os.path.join(nas.PRED_PATH, 'AAL.csv'))
