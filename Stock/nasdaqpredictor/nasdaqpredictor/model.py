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
import keras.backend as K
from keras.layers import Conv1D, MaxPooling1D

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
        X = data.drop('Return', axis=1)
        y = data['Return']
        self.data[ticker]['X'] = X  # store this, we may need it in future predictions

        X_train = X[:self.test_date]
        X_test = X[self.test_date:]
        y_train = y[:self.test_date]
        y_test = y[self.test_date:]

        self.data[ticker]['X_train'] = X_train
        self.data[ticker]['X_test'] = X_test
        self.data[ticker]['y_train'] = self.series_to_binarized_columns(y_train)
        self.data[ticker]['y_test'] = self.series_to_binarized_columns(y_test)
        self.data[ticker]['train_returns'] = y_train
        self.data[ticker]['test_returns'] = y_test

        if not hasattr(self, 'data_width'):
            self.data_width = X_train.shape[1]

    def series_to_binarized_columns(self, y):
        pos = y > self.extremes
        neg = y < -self.extremes
        meds = (y > -self.extremes) & (y < self.extremes)
        y = np.array([neg, meds, pos]).T
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

            model.add(Dense(3, kernel_initializer='uniform'))
            model.add(Activation('softmax'))

            model.compile(optimizer=Adam(lr=self.learning_rate),
                          #loss=self.create_entropy(),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            self.model = model
            self._fit_neural_net()
            self.model.save(self.file_path)
        else:
            LOGGER.info('Load neural net from filepath: {}'.format(self.file_path))
            self.model = load_model(self.file_path,
                                    custom_objects={'w_categorical_crossentropy': self.create_entropy()})

        LOGGER.info('Architecture: ')
        self.model.summary(print_fn=LOGGER.info)

    def create_entropy(self):
        def w_categorical_crossentropy(y_true, y_pred, weights):
            nb_cl = len(weights)
            final_mask = K.zeros_like(y_pred[:, 0])
            y_pred_max = K.max(y_pred, axis=1)
            y_pred_max = K.expand_dims(y_pred_max, 1)
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += (
                    K.cast(weights[c_t, c_p], K.floatx()) *
                    K.cast(y_pred_max_mat[:, c_p], K.floatx()) *
                    K.cast(y_true[:, c_t], K.floatx())
                )
            return K.categorical_crossentropy(y_pred, y_true) * final_mask

        weight_matrix = np.array([[0.1, 4, 7],
                                  [2, 0.1, 2],
                                  [7, 4, 0.1]]).astype(np.float64)
        wcce = partial(w_categorical_crossentropy, weights=weight_matrix)
        wcce.__name__ = 'w_categorical_crossentropy'
        return wcce

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

    def evaluate(self, export_image=False):
        predicted = self.model.predict(self.model.X_test)

        real_ups = self.model.y_test[:, 0]
        real_downs = self.model.y_test[:, 2]
        predicted_ups = np.logical_and(predicted[:, 0] > self.certainty, predicted[:,0]>predicted[:,2])
        predicted_downs = np.logical_and(predicted[:, 2] > self.certainty, predicted[:,2]>predicted[:,0])

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

            # LOGGER.info('===\nAll returns\n===')
            # self.print_returns_distribution(self.model.test_returns)

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
