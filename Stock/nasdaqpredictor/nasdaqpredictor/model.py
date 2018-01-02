import logging
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from keras.callbacks import LambdaCallback, TensorBoard
from keras.layers import Activation, Dense, Dropout, LSTM, Conv1D, MaxPool1D, Flatten, BatchNormalization
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

from nasdaqpredictor import TENSORBOARD_PATH
from dataloader import DataTransformer

LOGGER = logging.getLogger(__name__)


class Model:
    def __init__(self,
                 transformer: DataTransformer,
                 file_path=None,
                 dev_date=datetime(2013, 1, 1),
                 test_date=datetime(2015, 1, 1),
                 dropout=0.2,
                 epochs=100,
                 batch_size=512,
                 learning_rate=0.001,
                 extremes=3):
        self.transformer = transformer
        if file_path is None:
            now = datetime.now().strftime('%Y_%m_%d_%H_%M')
            self.file_path = os.path.join('models', 'full_model_' + now + '.hdf5')
            self.run_fit = True
        else:
            self.file_path = file_path
            self.run_fit = False
        self.dev_date = dev_date.strftime('%Y-%m-%d')
        self.test_date = test_date.strftime('%Y-%m-%d')
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.extremes = extremes

        self.data = defaultdict(lambda: {})

    def build_model_data(self):
        LOGGER.info('Build model data')
        if self.transformer.transformed_data_dict is None:
            self.transformer.transform()
            self._build_scaler()

        for ticker, data in self.transformer.transformed_data_dict.items():
            self._build_model_data_for_ticker(data, ticker)

    def _build_scaler(self):
        self.scaler = StandardScaler()
        frames = self.transformer.transformed_data_dict.values()
        train_subset = [frame.drop('Return', axis=1)[:self.dev_date] for frame in frames]
        full_X_train = np.concatenate(train_subset)
        self.data_shape = (full_X_train.shape[1], 1)
        self.scaler.fit(full_X_train)
        LOGGER.info('Data shape: ' + str(self.data_shape))

    def _build_model_data_for_ticker(self, data, ticker):

        def apply_to_all(func, iterable):
            return [func(subset) if subset is not None and len(subset) > 0 else None for subset in iterable]

        def scale(df: pd.DataFrame):
            orig_shape = df.shape
            returns = df['Return']
            scaled_values = self.scaler.transform(df.drop('Return', axis=1).values)
            ret = pd.DataFrame(np.concatenate([scaled_values, np.expand_dims(returns.values, axis=1)], axis=1),
                               index=df.index, columns=df.columns)
            assert ret.shape == orig_shape
            return ret

        def split_into_x_and_y(df: pd.DataFrame):
            X = df.drop('Return', axis=1)
            y = df['Return']
            return X, y

        train_dev_test = (data[:self.dev_date], data[self.dev_date:self.test_date], data[self.test_date:])
        train_dev_test = apply_to_all(scale, train_dev_test)
        train_dev_test = apply_to_all(split_into_x_and_y, train_dev_test)

        def get_by_position(x, y):
            if train_dev_test is None or train_dev_test[x] is None:
                return None
            return train_dev_test[x][y]

        self.data[ticker]['X_train'] = get_by_position(0, 0)
        self.data[ticker]['X_dev'] = get_by_position(1, 0)
        self.data[ticker]['X_test'] = get_by_position(2, 0)
        self.data[ticker]['y_train'] = self.series_to_binarized_columns(get_by_position(0, 1))
        self.data[ticker]['y_dev'] = self.series_to_binarized_columns(get_by_position(1, 1))
        self.data[ticker]['y_test'] = self.series_to_binarized_columns(get_by_position(2, 1))
        self.data[ticker]['dev_returns'] = get_by_position(1, 1)
        self.data[ticker]['test_returns'] = get_by_position(2, 1)

    def series_to_binarized_columns(self, y):
        if y is None or len(y) == 0:
            return None
        pos = y > self.extremes
        neg = y < -self.extremes
        meds = (y > -self.extremes) & (y < self.extremes)
        y = np.array([neg, meds, pos]).T
        return y

    def build_neural_net(self):
        if self.run_fit:
            LOGGER.info('Build neural network architecture')

            model = Sequential()

            model.add(LSTM(16, return_sequences=True, input_shape=self.data_shape))
            model.add(Dropout(self.dropout))
            model.add(LSTM(16))
            model.add(Dropout(self.dropout))

            model.add(Dense(3, kernel_initializer='glorot_uniform'))
            model.add(Activation('softmax'))

            model.compile(optimizer=Adam(lr=self.learning_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            self.model = model
            self._fit_neural_net()
            self.model.save(self.file_path)
        else:
            LOGGER.info('Load neural net from filepath: {}'.format(self.file_path))
            self.model = load_model(self.file_path)

        LOGGER.info('Architecture: ')
        self.model.summary(print_fn=LOGGER.info)

    def _fit_neural_net(self):
        LOGGER.info('Train neural network')

        batch_print_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: [LOGGER.info('===> epoch {} ended'.format(epoch + 1)),
                                              LOGGER.info(logs)])

        # tensorboard = TensorBoard(log_dir=TENSORBOARD_PATH, histogram_freq=0,
        #                           write_graph=True, write_images=True)

        y_train = np.concatenate([data['y_train'] for ticker, data in self.data.items() if data['y_train'] is not None])
        temp_y = np.argmax(y_train, axis=1)
        cw = class_weight.compute_class_weight('balanced', np.unique(temp_y), temp_y)

        LOGGER.info('Class weights: ' + str(cw))

        def get_numpy_array(df: pd.DataFrame):
            return np.expand_dims(df.values, axis=-1)

        for ticker, data in self.data.items():
            LOGGER.info(f'Fitting {ticker}')
            if data['X_train'] is not None:
                self.model.fit(get_numpy_array(data['X_train']), data['y_train'],
                               # validation_data=(get_numpy_array(data['X_dev']), data['y_dev']),
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               verbose=0,
                               class_weight=cw,
                               callbacks=[batch_print_callback]) # tensorboard

    def predict(self, X_test):
        predicted = self.model.predict(X_test)
        return predicted

    def predict_classes(self, X_test):
        predicted = self.model.predict_classes(X_test)
        return predicted


class ModelEvaluator:
    def __init__(self,
                 model: Model):
        self.model = model

    def evaluate(self, certainty=0.34, on_set='dev'):
        all_returns = []
        for ticker, data in self.model.data.items():
            if data['X_' + on_set] is None:
                continue
            returns = self.calculate_returns(data['X_' + on_set],
                                             data['y_' + on_set],
                                             data[on_set + '_returns'],
                                             certainty)
            all_returns.append(returns)

        LOGGER.info('===\nStrategy returns\n===')
        return self.print_returns_distribution(np.concatenate(all_returns))

    def calculate_returns(self, X, y, ret, certainty):
        predicted = self.model.predict(np.expand_dims(X, axis=-1))
        real_ups = y[:, 2]
        real_downs = y[:, 0]
        predicted_ups = (predicted[:, 2] > certainty) & (np.argmax(predicted, axis=1) == 2)
        predicted_downs = (predicted[:, 0] > certainty) & (np.argmax(predicted, axis=1) == 0)
        returns = np.append(ret[predicted_ups],
                            (-1 * ret[predicted_downs]))

        LOGGER.debug('Real ups count: {}'.format(pd.value_counts(real_ups[predicted_ups])))
        LOGGER.debug('Real downs count: {}'.format(pd.value_counts(real_downs[predicted_downs])))
        return returns

    def print_returns_distribution(self, returns):
        lose = np.sum(returns[returns < 0])
        win = np.sum(returns[returns > 0])
        if lose == 0 and win == 0:
            return False
        LOGGER.info('Negative returns: ' + str(lose))
        LOGGER.info('Positive returns: ' + str(win))
        LOGGER.info('Pos/Neg ratio: ' + str(win / (lose * -1)))
        LOGGER.info('Sum of returns: ' + str(np.sum(returns)))
        return True
