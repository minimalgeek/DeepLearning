from datetime import datetime
import logging
import os
from dataloader import DataLoader, DataTransformer
from model import Model, ModelEvaluator

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    loader = DataLoader('tickers/NASDAQ100.csv', load_from_google=False)
    transformer = DataTransformer(loader, return_shift_days=3)

    model = Model(transformer,
                  file_path='models/full_model_2017_12_11_18_44.hdf5',
                  dev_date=datetime(2013, 1, 1),
                  test_date=datetime(2015, 1, 1),
                  learning_rate=1e-3,
                  extra_layers=5,
                  neurons_per_layer=150,
                  dropout=0.3,
                  batch_size=2**7,
                  epochs=2,
                  extremes=3,
                  window=30)

    model.build_model_data()
    model.build_neural_net()

    model_evaluator = ModelEvaluator(model)
    for c in [0.35 + x/100 for x in range(50)]:
        LOGGER.info('======================== Ceratinty: {} ========================'.format(c))
        LOGGER.info('================= DEV =================')
        should_continue = model_evaluator.evaluate(certainty=c, on_set='dev')
        LOGGER.info('================= TEST =================')
        model_evaluator.evaluate(certainty=c, on_set='test')
        if not should_continue:
            break
