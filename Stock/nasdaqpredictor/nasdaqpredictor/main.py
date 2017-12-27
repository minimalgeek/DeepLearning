from datetime import datetime
import logging
from dataloader import DataLoader, DataTransformer
from model import Model, ModelEvaluator

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    loader = DataLoader('tickers/NASDAQ100.csv', load_from_google=False)
    transformer = DataTransformer(loader, return_shift_days=3)

    model = Model(transformer,
                  #file_path='models/full_model_2017_12_19_13_13.hdf5',
                  dev_date=datetime(2013, 1, 1),
                  test_date=datetime(2015, 1, 1),
                  learning_rate=1e-3,
                  dropout=0.1,
                  batch_size=2**12,
                  epochs=100,
                  extremes=3)

    model.build_model_data()
    model.build_neural_net()

    model_evaluator = ModelEvaluator(model)
    for c in [0.35 + x/50 for x in range(20)]:
        LOGGER.info('======================== Certainty: {} ========================'.format(c))
        LOGGER.info('================= DEV =================')
        should_continue = model_evaluator.evaluate(certainty=c, on_set='dev')
        LOGGER.info('================= TEST =================')
        model_evaluator.evaluate(certainty=c, on_set='test')
        if not should_continue:
            break
