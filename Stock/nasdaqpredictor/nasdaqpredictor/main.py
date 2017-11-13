from datetime import datetime

from dataloader import DataLoader, DataTransformer
from model import Model, ModelEvaluator

if __name__ == '__main__':
    loader = DataLoader('/nasdaq_tickers.csv',
                        datetime(2000, 1, 1),
                        datetime(2017, 1, 1))
    transformer = DataTransformer(loader)
    model = Model(transformer,
                  #file_path='models/full_model_2017_11_13_13_25.hdf5',
                  test_date=datetime(2014, 1, 1),
                  learning_rate=0.001,
                  extra_layers=16,
                  neurons_per_layer=60,
                  dropout=0.1,
                  batch_size=512,
                  epochs=200)

    model.build_model_data()
    model.build_neural_net()

    model_evaluator = ModelEvaluator(model, certainty=0.7)
    model_evaluator.evaluate()
    #model_evaluator.evaluate_report()
