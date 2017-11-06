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
    def __init__(self,
                 transformer: DataTransformer,
                 test_size=0.05,
                 neurons_per_layer=150,
                 extra_layers=4,
                 epochs=500,
                 batch_size=512,
                 learning_rate=0.001,
                 run_fit=True):
        self.transformer = transformer
        self.test_size = test_size
        self.neurons_per_layer = neurons_per_layer
        self.extra_layers = extra_layers  # beyond the first hidden layer
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.run_fit = run_fit

        if self.transformer.transformed_data_dict is None:
            self.transformer.transform()
        self.data = defaultdict(lambda: {})

    def build_model_data(self):
        LOGGER.info('Build model data for training')
        for ticker, data in self.transformer.transformed_data_dict.items():
            self._build_model_data_for_ticker(data, ticker)

    def _build_model_data_for_ticker(self, data, ticker):
        self.data[ticker]['X'] = data.drop('Return', axis=1)
        y = data['Return'] > 0
        y = np.expand_dims(y, axis=1)
        self.data[ticker]['y'] = np.hstack((y, 1 - y))
        X_train, X_test, y_train, y_test = train_test_split(self.data[ticker]['X'],
                                                            self.data[ticker]['y'],
                                                            test_size=self.test_size,
                                                            shuffle=False)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        self.data[ticker]['X_train'] = X_train
        self.data[ticker]['X_test'] = X_test
        self.data[ticker]['y_train'] = y_train
        self.data[ticker]['y_test'] = y_test
        self.data[ticker]['train_returns'] = data['Return'][:len(y_train)]
        self.data[ticker]['test_returns'] = data['Return'][len(y_train):]

        if not hasattr(self, 'data_width'):
            self.data_width = X_train.shape[1]

    def build_neural_net(self):
        LOGGER.info('Build neural network architecture')
        model = Sequential()

        model.add(Dense(self.neurons_per_layer, input_dim=self.data_width, kernel_initializer='uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        for _ in range(self.extra_layers):
            model.add(Dense(self.neurons_per_layer, kernel_initializer='uniform'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.2))

        model.add(Dense(2, kernel_initializer='uniform'))
        model.add(Activation('softmax'))

        model.compile(optimizer=Adam(lr=self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model
        LOGGER.info('Architecture: ')
        model.summary(print_fn=LOGGER.info)

    def fit_neural_net(self):
        LOGGER.info('Train neural network')

        self.X_train, self.X_test, self.y_train, self.y_test, self.test_returns = self._prepare_data_for_fit()

        batch_print_callback = LambdaCallback(
            # on_batch_end=lambda batch, logs: [LOGGER.info(logs)],
            on_epoch_end=lambda epoch, logs: [LOGGER.info('===> epoch {} ended'.format(epoch+1)), LOGGER.info(logs)])

        if self.run_fit:
            self.model.fit(self.X_train, self.y_train,
                           validation_data=(self.X_test, self.y_test),
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           verbose=2,
                           callbacks=[batch_print_callback])
            self.model.save(FILEPATH)
        else:
            self.model = load_model(FILEPATH)

        score = self.model.evaluate(self.X_test, self.y_test)
        LOGGER.info('Test loss: {}, Test accuracy: {}'.format(score[0], score[1]))

    def predict(self, X_test):
        predicted = self.model.predict(X_test)
        return predicted

    def _prepare_data_for_fit(self):
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
        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)
        test_returns = np.concatenate(test_returns)
        return X_train, X_test, y_train, y_test, test_returns


class ModelEvaluator:
    def __init__(self,
                 model: Model,
                 certainty_multiplier=0.999):
        self.model = model
        self.certainty_multiplier = certainty_multiplier

    def evaluate(self):
        predicted = self.model.predict(self.model.X_test)

        certainty_percentage = predicted.max() * self.certainty_multiplier
        LOGGER.info('Certainty is {}%'.format(certainty_percentage))

        predicted_ups = predicted[:, 0] > certainty_percentage
        predicted_downs = predicted[:, 1] > certainty_percentage

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

        LOGGER.info('===\nAll returns\n===')
        self.print_returns_distribution(self.model.test_returns)

    def print_returns_distribution(self, returns):
        neg = np.sum(returns[returns < 0])
        pos = np.sum(returns[returns > 0])
        LOGGER.info('Negative returns: ' + str(neg))
        LOGGER.info('Positive returns: ' + str(pos))
        LOGGER.info('Pos/Neg ratio: ' + str(pos / (neg * -1)))
        LOGGER.info('Sum of returns: ' + str(np.sum(returns)))
