import numpy as np
import pandas as pd
import logging
import os

from datetime import datetime

from keras import Input
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from keras.models import load_model
import keras_resnet.models

from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler

from dataloader import DataTransformer
from collections import defaultdict

LOGGER = logging.getLogger(__name__)


class Model:
    def __init__(self,
                 transformer: DataTransformer,
                 file_path=None,
                 dev_date: datetime = datetime(2013, 1, 1),
                 test_date: datetime = datetime(2015, 1, 1),
                 neurons_per_layer=150,
                 dropout=0.2,
                 extra_layers=4,
                 epochs=500,
                 batch_size=512,
                 learning_rate=0.001,
                 window=30,
                 extremes=5):
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
        self.neurons_per_layer = neurons_per_layer
        self.dropout = dropout
        self.extra_layers = extra_layers  # beyond the first hidden layer
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.window = window
        self.extremes = extremes

        self.data = defaultdict(lambda: {})

    def build_model_data(self):
        LOGGER.info('Build model data')
        if self.transformer.transformed_data_dict is None:
            self.transformer.transform()
            self._build_scaler()

        for ticker, data in self.transformer.transformed_data_dict.items():
            self._build_model_data_for_ticker(data, ticker)

        self._summarize_and_scale_data()

    def _build_scaler(self):
        self.scaler = StandardScaler()
        frames = self.transformer.transformed_data_dict.values()
        train_subset = [frame.drop('Return', axis=1)[:self.dev_date] for frame in frames]
        full_X_train = np.concatenate(train_subset)
        self.scaler.fit(full_X_train)

    def _build_model_data_for_ticker(self, data, ticker):

        X_data = data.drop('Return', axis=1)
        y_data = data['Return']

        def apply_to_all(func, X_subsets, min_size=0):
            return tuple(
                [func(subset) if subset is not None and len(subset) >= min_size else None for subset in X_subsets])

        def build_all_the_windows(input):
            return [input[i:i + self.window] for i in range(0, input.shape[0] - self.window + 1)]

        def stack_together_all_the_windows(X):
            return np.stack(X) if len(X) > 0 else None

        # train, dev, test set
        X_subset = (X_data[:self.dev_date], X_data[self.dev_date:self.test_date], X_data[self.test_date:])
        X_subset = apply_to_all(self.scaler.transform, X_subset, self.window)
        X_subset = apply_to_all(build_all_the_windows, X_subset, self.window)
        X_subset = apply_to_all(stack_together_all_the_windows, X_subset, 1)

        y_train = y_data[:self.dev_date].iloc[self.window - 1:]
        y_dev = y_data[self.dev_date:self.test_date].iloc[self.window - 1:]
        y_test = y_data[self.test_date:].iloc[self.window - 1:]

        self.data[ticker]['X_train'] = X_subset[0]
        self.data[ticker]['X_dev'] = X_subset[1]
        self.data[ticker]['X_test'] = X_subset[2]
        self.data[ticker]['y_train'] = self.series_to_binarized_columns(y_train)
        self.data[ticker]['y_dev'] = self.series_to_binarized_columns(y_dev)
        self.data[ticker]['y_test'] = self.series_to_binarized_columns(y_test)
        self.data[ticker]['dev_returns'] = y_dev
        self.data[ticker]['test_returns'] = y_test

        if not hasattr(self, 'data_shape') and X_subset[0] is not None:
            self.data_shape = X_subset[0].shape[1:]

    def series_to_binarized_columns(self, y):
        if len(y) == 0:
            return None
        pos = y > self.extremes
        neg = y < -self.extremes
        meds = (y > -self.extremes) & (y < self.extremes)
        y = np.array([neg, meds, pos]).T
        return y

    def _summarize_and_scale_data(self):
        X_train = []
        X_dev = []
        X_test = []
        y_train = []
        y_dev = []
        y_test = []
        dev_returns = []
        test_returns = []

        def append(left, right):
            if right is not None and len(right) > 0:
                left.append(right)

        for ticker, datas in self.data.items():
            append(X_train, datas['X_train'])
            append(X_dev, datas['X_dev'])
            append(X_test, datas['X_test'])
            append(y_train, datas['y_train'])
            append(y_dev, datas['y_dev'])
            append(y_test, datas['y_test'])
            append(dev_returns, datas['dev_returns'])
            append(test_returns, datas['test_returns'])
        # self.X_train = np.concatenate(X_train)
        # self.X_dev = np.concatenate(X_dev)
        # self.X_test = np.concatenate(X_test)

        self.X_train = np.expand_dims(np.concatenate(X_train), axis=-1)
        self.X_dev = np.expand_dims(np.concatenate(X_dev), axis=-1)
        self.X_test = np.expand_dims(np.concatenate(X_test), axis=-1)

        self.y_train = np.concatenate(y_train)
        self.y_dev = np.concatenate(y_dev)
        self.y_test = np.concatenate(y_test)

        assert len(self.X_train) == len(self.y_train)
        assert len(self.X_dev) == len(self.y_dev)
        assert len(self.X_test) == len(self.y_test)

        self.dev_returns = np.concatenate(dev_returns)
        self.test_returns = np.concatenate(test_returns)

    def build_neural_net(self):
        if self.run_fit:
            LOGGER.info('Build neural network architecture')

            # model = Sequential()
            #
            # # model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu',
            # #                  input_shape=self.data_shape))
            # # model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
            # # model.add(Dropout(self.dropout))
            # # model.add(MaxPool1D(pool_size=2, padding='same'))
            # # model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
            # # model.add(Conv1D(filters=128, kernel_size=6, padding='same', activation='relu'))
            # # model.add(Dropout(self.dropout))
            # # model.add(MaxPool1D(pool_size=2, padding='same'))
            #
            # model.add(Flatten(input_shape=self.data_shape))
            # # model.add(Flatten())
            #
            # for _ in range(self.extra_layers):
            #     model.add(Dense(self.neurons_per_layer, kernel_initializer='glorot_uniform'))
            #     model.add(BatchNormalization())
            #     model.add(Activation('relu'))
            #     model.add(Dropout(self.dropout))
            #
            # model.add(Dense(3, kernel_initializer='glorot_uniform'))
            # model.add(Activation('softmax'))

            x = Input(self.data_shape+(1,))
            model = keras_resnet.models.ResNet18(x, classes=3)
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

        temp_y = np.argmax(self.y_train, axis=1)
        cw = class_weight.compute_class_weight('balanced', np.unique(temp_y), temp_y)

        LOGGER.info('Class weights: ' + str(cw))

        self.model.fit(self.X_train, self.y_train,
                       validation_data=(self.X_dev, self.y_dev),
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=0,
                       class_weight=cw,
                       callbacks=[batch_print_callback])

    def predict(self, X_test):
        """
        Predict from an array
        :param X_test: Input, should be already scaled
        :return: prediction
        """
        predicted = self.model.predict(X_test)
        return predicted

    def predict_classes(self, X_test):
        predicted = self.model.predict_classes(X_test)
        return predicted

    def predict_one(self, ticker, date_to_predict: datetime):
        X_test = self.data[ticker]['X'].loc[date_to_predict.strftime('%Y-%m-%d')]
        X_test_to_network = np.expand_dims(X_test, axis=0)
        X_test_transformed = self.scaler.transform(X_test_to_network)
        predicted = self.model.predict(X_test_transformed)
        return predicted


class ModelEvaluator:
    def __init__(self,
                 model: Model):
        self.model = model
        LOGGER.info('===\nAll returns\n===')
        self.print_returns_distribution(self.model.dev_returns)

    def evaluate(self, certainty=0.34, on_set='dev'):
        if on_set == 'dev':
            predicted_downs, predicted_ups, real_downs, real_returns, real_ups = self.calculate_returns(
                self.model.X_dev, certainty,
                self.model.dev_returns, self.model.y_dev)
        elif on_set == 'test':
            predicted_downs, predicted_ups, real_downs, real_returns, real_ups = self.calculate_returns(
                self.model.X_test, certainty,
                self.model.test_returns, self.model.y_test)

        LOGGER.info('Real ups count')
        LOGGER.info(pd.value_counts(real_ups[predicted_ups]))
        LOGGER.info('Real downs count')
        LOGGER.info(pd.value_counts(real_downs[predicted_downs]))

        LOGGER.info('===\nStrategy returns\n===')
        return self.print_returns_distribution(real_returns)

    def calculate_returns(self, X, certainty, ret, y):
        predicted = self.model.predict(X)
        real_ups = y[:, 0]
        real_downs = y[:, 2]
        predicted_ups = (predicted[:, 0] > certainty) & (np.argmax(predicted, axis=1) == 0)
        predicted_downs = (predicted[:, 2] > certainty) & (np.argmax(predicted, axis=1) == 2)
        real_returns = np.append(ret[predicted_ups],
                                 (-1 * ret[predicted_downs]))
        return predicted_downs, predicted_ups, real_downs, real_returns, real_ups

    def print_returns_distribution(self, returns):
        neg = np.sum(returns[returns < 0])
        pos = np.sum(returns[returns > 0])
        if neg == 0 and pos == 0:
            return False
        LOGGER.info('Negative returns: ' + str(neg))
        LOGGER.info('Positive returns: ' + str(pos))
        LOGGER.info('Pos/Neg ratio: ' + str(pos / (neg * -1)))
        LOGGER.info('Sum of returns: ' + str(np.sum(returns)))
        return True
