import numpy as np
import pandas as pd
import logging

from keras.models import Sequential
from keras.layers import Activation, Dense, LeakyReLU, Dropout, BatchNormalization
from keras.losses import mean_squared_error, binary_crossentropy
from keras.optimizers import Adam, RMSprop
from keras import metrics
from keras import regularizers
from keras.callbacks import LambdaCallback
from keras.models import load_model

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from dataloader import DataTransformer
from collections import defaultdict

FILEPATH = 'full_model.hdf5'

LOGGER = logging.getLogger(__name__)


class Model:
    def __init__(self, transformer: DataTransformer):
        self.transformer = transformer
        
        if self.transformer.transformed_data_dict is None:
            self.transformer.transform()
        self.data = defaultdict(lambda: None)

    def build_model_data(self):
        LOGGER.info('Build model data for training')
        for ticker, data in self.transformer.transformed_data_dict.items():
            self._build_model_data_for_ticker(data, ticker)

    def _build_model_data_for_ticker(self, data, ticker):
        scaler = StandardScaler()
        
        X = data.drop('Return', axis=1)
        X_val = scaler.fit_transform(X)
        self.data[ticker] = pd.DataFrame(X_val, index=X.index, columns=X.columns)

    def build_neural_net(self):
        self.model = load_model(FILEPATH)
        LOGGER.info('Architecture: ')
        self.model.summary(print_fn=LOGGER.info)

    def predict_one(self, ticker, date_to_predict):
        X_test = self.data[ticker].loc[date_to_predict]
        X_test_to_network = np.expand_dims(X_test.values, axis=0)
        predicted = self.model.predict(X_test_to_network)
        return predicted
    