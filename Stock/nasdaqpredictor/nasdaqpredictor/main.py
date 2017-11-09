from datetime import datetime

from dataloader import DataLoader, DataTransformer
from model import Model, ModelEvaluator

if __name__ == '__main__':
    loader = DataLoader('/nasdaq_tickers.csv',
                        datetime(2000, 1, 1),
                        datetime(2017, 1, 1))
    transformer = DataTransformer(loader)
    model = Model(transformer,
                  test_date=datetime(2014, 1, 1),
                  learning_rate=0.0001,
                  extra_layers=10,
                  neurons_per_layer=70,
                  batch_size=2048,
                  epochs=1000,
                  run_fit=True)

    model.build_model_data()
    model.build_neural_net()

    model_evaluator = ModelEvaluator(model, certainty=0.7)
    model_evaluator.evaluate()
