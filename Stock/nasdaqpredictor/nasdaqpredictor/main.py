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
                  neurons_per_layer=50,
                  learning_rate=0.001,
                  batch_size=512,
                  epochs=1000,
                  run_fit=True)

    model.build_model_data()
    model.build_neural_net()
    model.fit_neural_net()

    model_evaluator = ModelEvaluator(model,
                                     certainty_multiplier=0.95)
    model_evaluator.evaluate()
