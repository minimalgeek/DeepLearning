from datetime import datetime

from dataloader import DataLoader, DataTransformer
from model import Model, ModelEvaluator

if __name__ == '__main__':
    loader = DataLoader('/nasdaq_tickers.csv',
                        datetime(2000, 1, 1),
                        datetime(2017, 1, 1))
    loader.reload_all()

    transformer = DataTransformer(loader)
    transformer.transform()

    model = Model(transformer,
                  test_size=0.1,
                  learning_rate=0.001,
                  extra_layers=5,
                  neurons_per_layer=60,
                  batch_size=1024,
                  epochs=200,
                  run_fit=True)

    model.build_model_data()
    model.build_neural_net()
    model.fit_neural_net()

    model_evaluator = ModelEvaluator(model,
                                     certainty_multiplier=0.7)
    model_evaluator.evaluate()
