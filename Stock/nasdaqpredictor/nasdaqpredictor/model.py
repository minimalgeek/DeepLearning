import numpy as np
import pandas as pd
import logging
import os

from datetime import datetime
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from keras.models import load_model
from keras.layers import Conv1D, MaxPool1D

from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from dataloader import DataTransformer
from collections import defaultdict
from itertools import product
from functools import partial

LOGGER = logging.getLogger(__name__)


class Model:
    def __init__(self,
                 transformer: DataTransformer,
                 file_path=None,
                 test_date: datetime = datetime(2014, 1, 1),
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

        for ticker, data in self.transformer.transformed_data_dict.items():
            self._build_model_data_for_ticker(data, ticker)

        self._summarize_and_scale_data()

    def _build_model_data_for_ticker(self, data, ticker):

        X_data = data.drop('Return', axis=1)
        y_data = data['Return']

        X_train = X_data[:self.test_date]
        X_test = X_data[self.test_date:]
        y_train = y_data[:self.test_date].iloc[self.window - 1:]
        y_test = y_data[self.test_date:].iloc[self.window - 1:]

        if len(X_test) == 0 or len(X_train) == 0:
            return

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        self.data[ticker]['scaler'] = scaler

        def build_2D_input_data(input):
            return [input[i:i + self.window] for i in range(0, input.shape[0] - self.window + 1)]

        X_train = build_2D_input_data(X_train)
        X_test = build_2D_input_data(X_test)

        X_train = np.stack(X_train)
        X_test = np.stack(X_test)

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        self.data[ticker]['X_train'] = X_train
        self.data[ticker]['X_test'] = X_test
        self.data[ticker]['y_train'] = self.series_to_binarized_columns(y_train)
        self.data[ticker]['y_test'] = self.series_to_binarized_columns(y_test)
        self.data[ticker]['test_returns'] = y_test

        if not hasattr(self, 'data_shape'):
            self.data_shape = X_train.shape[1:]

    def series_to_binarized_columns(self, y):
        pos = y > self.extremes
        neg = y < -self.extremes
        meds = (y > -self.extremes) & (y < self.extremes)
        y = np.array([neg, meds, pos]).T
        return y

    def _summarize_and_scale_data(self):
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        test_returns = []
        for ticker, datas in self.data.items():
            X_train.append(datas['X_train'])
            X_test.append(datas['X_test'])
            y_train.append(datas['y_train'])
            y_test.append(datas['y_test'])
            test_returns.append(datas['test_returns'])
        self.X_train = np.concatenate(X_train)
        self.X_test = np.concatenate(X_test)
        self.y_train = np.concatenate(y_train)
        self.y_test = np.concatenate(y_test)
        self.test_returns = np.concatenate(test_returns)

    def build_neural_net(self):
        if self.run_fit:
            LOGGER.info('Build neural network architecture')

            model = Sequential()

            model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu',
                             input_shape=self.data_shape))
            model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
            model.add(Dropout(self.dropout))
            model.add(MaxPool1D(pool_size=2, padding='same'))
            model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
            model.add(Conv1D(filters=128, kernel_size=6, padding='same', activation='relu'))
            model.add(Dropout(self.dropout))
            model.add(MaxPool1D(pool_size=2, padding='same'))

            model.add(Flatten())

            for _ in range(self.extra_layers):
                model.add(Dense(self.neurons_per_layer, kernel_initializer='glorot_uniform'))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
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

        temp_y = np.argmax(self.y_train, axis=1)
        cw = class_weight.compute_class_weight('balanced', np.unique(temp_y), temp_y)

        LOGGER.info('Class weights: ' + str(cw))

        self.model.fit(self.X_train, self.y_train,
                       validation_data=(self.X_test, self.y_test),
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
        self.print_returns_distribution(self.model.test_returns)

    def evaluate(self, export_image=False, certainty=0.34):
        predicted = self.model.predict(self.model.X_test)

        real_ups = self.model.y_test[:, 0]
        real_downs = self.model.y_test[:, 2]
        predicted_ups = (predicted[:, 0] > certainty) & (np.argmax(predicted, axis=1) == 0)
        predicted_downs = (predicted[:, 2] > certainty) & (np.argmax(predicted, axis=1) == 2)

        LOGGER.info('Real ups count')
        LOGGER.info(pd.value_counts(real_ups[predicted_ups]))
        LOGGER.info('Real downs count')
        LOGGER.info(pd.value_counts(real_downs[predicted_downs]))

        real_returns = np.append(self.model.test_returns[predicted_ups],
                                 (-1 * self.model.test_returns[predicted_downs]))

        LOGGER.info('===\nStrategy returns\n===')
        self.print_returns_distribution(real_returns)
        if export_image:
            self.display_returns(real_returns)

    def evaluate_report(self):
        predicted = self.model.predict_classes(self.model.X_test)
        LOGGER.info(classification_report(self.model.y_test[:, 0], predicted))

    def print_returns_distribution(self, returns):
        neg = np.sum(returns[returns < 0])
        pos = np.sum(returns[returns > 0])
        LOGGER.info('Negative returns: ' + str(neg))
        LOGGER.info('Positive returns: ' + str(pos))
        LOGGER.info('Pos/Neg ratio: ' + str(pos / (neg * -1)))
        LOGGER.info('Sum of returns: ' + str(np.sum(returns)))

    def display_returns(self, returns):
        import seaborn as sns
        plot = sns.tsplot(returns)
        plot.get_figure().savefig(self.model.file_path + '.png')
