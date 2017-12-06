from datetime import datetime
import logging
import os
from dataloader import DataLoader, DataTransformer
from model import Model, ModelEvaluator

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    loader = DataLoader('/nasdaq_tickers.csv',
                        datetime(2000, 1, 1),
                        datetime(2017, 1, 1))
    transformer = DataTransformer(loader, return_shift_days=3)

    model = Model(transformer,
                  #file_path='models/full_model_2017_12_06_11_19.hdf5',
                  test_date=datetime(2015, 1, 1),
                  learning_rate=1e-2,
                  extra_layers=3,
                  neurons_per_layer=40,
                  dropout=0.4,
                  batch_size=2**12,
                  epochs=100,
                  extremes=4)

    model.build_model_data()
    model.build_neural_net()

    model_evaluator = ModelEvaluator(model)
    for c in [0.34 + x/100 for x in range(20)]:
        LOGGER.info('====================\nCeratinty: {}'.format(c))
        model_evaluator.evaluate(certainty=c)
