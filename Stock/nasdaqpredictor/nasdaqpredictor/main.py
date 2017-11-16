from datetime import datetime
import logging
import os
from dataloader import DataLoader, DataTransformer
from model import Model, ModelEvaluator

LOGGER = logging.getLogger(__name__)


def evaluate_all_models():
    global model
    for file in os.listdir('./models/'):
        model = Model(transformer,
                      file_path='models/' + file,
                      test_date=datetime(2014, 1, 1),
                      learning_rate=0.001,
                      extra_layers=16,
                      neurons_per_layer=100,
                      dropout=0.1,
                      batch_size=1024,
                      epochs=100)

        model.build_model_data()
        model.build_neural_net()

        model_evaluator = ModelEvaluator(model, certainty=0.8)
        model_evaluator.evaluate()


def grid_search():
    global model
    for extras in [5, 10, 20]:
        for neurons in [25, 50, 100]:
            for dropout in [0.1, 0.3]:
                LOGGER.info(50 * '=')
                LOGGER.info('layers: {}, neurons: {}, dropout: {}'.format(extras, neurons, dropout))
                model = Model(transformer,
                              # file_path='models/full_model_2017_11_13_14_56.hdf5',
                              test_date=datetime(2014, 1, 1),
                              learning_rate=0.001,
                              extra_layers=extras,
                              neurons_per_layer=neurons,
                              dropout=dropout,
                              batch_size=1024,
                              epochs=100)

                model.build_model_data()
                model.build_neural_net()

                model_evaluator = ModelEvaluator(model, certainty=0.6)
                model_evaluator.evaluate()


if __name__ == '__main__':
    loader = DataLoader('/nasdaq_tickers.csv',
                        datetime(2000, 1, 1),
                        datetime(2017, 1, 1))
    transformer = DataTransformer(loader, return_shift_days=-3)

    model = Model(transformer,
                  #file_path='models/full_model_2017_11_16_15_02.hdf5',
                  test_date=datetime(2015, 1, 1),
                  learning_rate=1e-3,
                  extra_layers=20,
                  neurons_per_layer=20,
                  dropout=0.1,
                  batch_size=1024,
                  epochs=100)

    model.build_model_data()
    model.build_neural_net()

    model_evaluator = ModelEvaluator(model, certainty=0.7)
    model_evaluator.evaluate()

    # grid_search()
    # evaluate_all_models()
