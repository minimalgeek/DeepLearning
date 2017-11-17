import numpy as np
import pandas as pd
import logging
import os

from datetime import datetime
from keras.models import Sequential
from keras.layers import Activation, Dense, LeakyReLU, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from keras.models import load_model

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from dataloader import DataTransformer
from collections import defaultdict

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
                 learning_rate=0.001):
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

        self.data = defaultdict(lambda: {})

    def build_model_data(self):
        LOGGER.info('Build model data')
        if self.transformer.transformed_data_dict is None:
            self.transformer.transform()

        for ticker, data in self.transformer.transformed_data_dict.items():
            self._build_model_data_for_ticker(data, ticker)

        self._summarize_and_scale_data()

    def _build_model_data_for_ticker(self, data, ticker):
        X = data.drop('Return', axis=1)
        y = data['Return']
        self.data[ticker]['X'] = X  # store this, we may need it in future predictions

        X_train = X[:self.test_date]
        X_test = X[self.test_date:]
        y_train = y[:self.test_date]
        y_test = y[self.test_date:]

        self.data[ticker]['X_train'] = X_train
        self.data[ticker]['X_test'] = X_test
        self.data[ticker]['y_train'] = Model.series_to_binarized_columns(y_train)
        self.data[ticker]['y_test'] = Model.series_to_binarized_columns(y_test)
        self.data[ticker]['train_returns'] = y_train
        self.data[ticker]['test_returns'] = y_test

        if not hasattr(self, 'data_width'):
            self.data_width = X_train.shape[1]

    def series_to_binarized_columns(y):
        y = y > 0
        y = np.expand_dims(y, axis=1)
        y = np.hstack((y, 1 - y))
        return y

    def build_neural_net(self):
        if self.run_fit:
            LOGGER.info('Build neural network architecture')
            model = Sequential()

            model.add(Dense(self.neurons_per_layer, input_dim=self.data_width, kernel_initializer='glorot_uniform'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(self.dropout))

            for _ in range(self.extra_layers):
                model.add(Dense(self.neurons_per_layer, kernel_initializer='glorot_uniform'))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Dropout(self.dropout))

            model.add(Dense(2, kernel_initializer='uniform'))
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

        self.model.fit(self.X_train, self.y_train,
                       validation_data=(self.X_test, self.y_test),
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=3,
                       callbacks=[batch_print_callback])
        # score = self.model.evaluate(self.X_test, self.y_test)
        # LOGGER.info('Test loss: {}, Test accuracy: {}'.format(score[0], score[1]))

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
        X_train = np.concatenate(X_train)
        X_test = np.concatenate(X_test)
        self.y_train = np.concatenate(y_train)
        self.y_test = np.concatenate(y_test)
        self.test_returns = np.concatenate(test_returns)

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)


class ModelEvaluator:
    def __init__(self,
                 model: Model,
                 certainty=0.6):
        self.model = model
        self.certainty = certainty

    def evaluate(self):
        predicted = self.model.predict(self.model.X_test)

        predicted_ups = predicted[:, 0] > self.certainty
        predicted_downs = predicted[:, 1] > self.certainty

        real_ups = self.model.y_test[:, 0] == 1
        real_downs = self.model.y_test[:, 1] == 1

        LOGGER.info('Real ups count')
        LOGGER.info(pd.value_counts(real_ups[predicted_ups]))
        LOGGER.info('Real downs count')
        LOGGER.info(pd.value_counts(real_downs[predicted_downs]))

        real_returns = np.append(self.model.test_returns[predicted_ups],
                                 (-1 * self.model.test_returns[predicted_downs]))

        LOGGER.info('===\nStrategy returns\n===')
        self.print_returns_distribution(real_returns)
        self.display_returns(real_returns)

        LOGGER.info('===\nAll returns\n===')
        self.print_returns_distribution(self.model.test_returns)

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
