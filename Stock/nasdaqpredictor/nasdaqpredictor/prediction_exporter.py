from datetime import datetime
import pandas as pd
from dataloader import DataLoader, DataTransformer
from model import Model, ModelEvaluator
import nasdaqpredictor as nas
import logging

LOGGER = logging.getLogger(__name__)


class PredictionExporter:
    def __init__(self,
                 model: Model):
        self.model = model
        if len(self.model.data) == 0:
            self.model.build_model_data()
            self.model.build_neural_net()
        self.tickers = self.model.transformer.data_loader.all_tickers

    def export_to_csv(self):
        from_date = self.model.test_date
        to_date = self.model.transformer.data_loader.to_date

        for ticker in self.tickers.ticker:
            predicted_dict = {}
            datelist = pd.date_range(start=from_date, end=to_date, freq='B')
            for date in datelist:
                try:
                    predicted = self.model.predict_one(ticker, date)
                    predicted_dict[date.strftime('%Y-%m-%d')] = predicted[0]
                except Exception as e:
                    LOGGER.error(e)

            self._to_dataframe_and_write_to_file(predicted_dict, ticker)

    def _to_dataframe_and_write_to_file(self, predicted_dict, ticker):
        try:
            df = pd.DataFrame.from_dict(predicted_dict,
                                        orient='index')
            df.columns = ['long_probability', 'short_probability']
            df.to_csv('{}/{}.csv'.format(nas.PRED_PATH, ticker), index_label='date')
        except Exception as e:
            LOGGER.error(e)


if __name__ == '__main__':
    loader = DataLoader('/nasdaq_tickers.csv',
                        datetime(2000, 1, 1),
                        datetime(2017, 1, 1))
    transformer = DataTransformer(loader)
    model = Model(transformer,
                  file_path='models/full_model_2017_11_15_15_52.hdf5',
                  test_date=datetime(2015, 1, 1),
                  learning_rate=1e-3,
                  extra_layers=15,
                  neurons_per_layer=70,
                  dropout=0.05,
                  batch_size=1024,
                  epochs=100)
    exporter = PredictionExporter(model=model)
    exporter.export_to_csv()
