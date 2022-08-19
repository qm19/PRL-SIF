# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:02:13 2020
@author: MS
"""

from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
np.random.seed(27)

import keras.backend as K
from keras.models import Model
# from keras.optimizers import SGD
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras import regularizers


class PUNN(object):
    def __init__(self,
                 input_dims,
                 act_fun = 'sigmoid',
                 l2_parameter=0.001,
	piror = 0.469):

        super(PUNN, self).__init__()

        self.input_dims = input_dims
        self.l2_para = l2_parameter
        self.prior = piror

        self.inputlayers = Input(shape=(self.input_dims,), name='input')
        self.full_connected = Dense(50, name='full_connected_1', activation='elu',
                                    kernel_regularizer=regularizers.l2(self.l2_para),
                                    bias_regularizer=regularizers.l2(self.l2_para))(self.inputlayers)
        self.full_connected = Dense(20, name='full_connected_2', activation='elu',
                                    kernel_regularizer=regularizers.l2(self.l2_para),
                                    bias_regularizer=regularizers.l2(self.l2_para))(self.full_connected)
        self.full_connected = Dense(10, name='full_connected_3', activation='elu',
                                    kernel_regularizer=regularizers.l2(self.l2_para),
                                    bias_regularizer=regularizers.l2(self.l2_para))(self.full_connected)
        self.probability = Dense(1, name='label_prediction', activation=act_fun,
                                    kernel_regularizer=regularizers.l2(self.l2_para),
                                    bias_regularizer=regularizers.l2(self.l2_para))(self.full_connected)
        self.model = Model(inputs=self.inputlayers,
                           outputs=self.probability)
        self.y_pred = []

    def load_weights(self, weights_path):  # load weights of PUNN model
        self.model.load_weights(weights_path)
        print('weights is loaded successfully.')

    def extract_feature(self, x):  # extract features from before output layer
        return self.full_connected.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        y_pred = self.model.predict(x, verbose=0)
        return y_pred

    def predict_label(self, x, threshold=0.5):  # predict cluster labels using the output of clustering layer
        y_pred = self.model.predict(x, verbose=0)
        y_label = np.where(y_pred > threshold, 1, -1)
        return y_label

    def LSIFLoss(self, y_true, y_pred):
        # y_pred = tf.clip_by_value(y_pred, 0, 1)
        positive_index, unlabeled_index = (y_true + 1) / 2, (-y_true + 1) / 2
        n_u, n_p = K.max([1, K.sum(unlabeled_index)]), K.max([1, K.sum(positive_index)])
        loss_unlabeled = y_pred ** 2
        loss_positive = y_pred
        objective = K.sum( loss_unlabeled * unlabeled_index / (2*n_u) ) - K.sum( loss_positive * positive_index / n_p)
        return objective

    def PUSBLoss(self, y_true, y_pred):
        sigmoid_loss = (lambda x: K.sigmoid(-x))
        positive_index, unlabeled_index = (y_true + 1) / 2, (-y_true + 1) / 2
        n_positive, n_unlabeled = K.max([1, K.sum(positive_index)]), K.max([1, K.sum(unlabeled_index)])
        y_log = K.log(y_pred)
        y_log1 = K.log(1-y_pred)
        positive_risk = K.sum(-self.prior * positive_index * y_log / n_positive)
        negative_risk = K.sum(self.prior * positive_index * y_log1 / n_positive) - K.sum(unlabeled_index * y_log1 / n_unlabeled)
        objective = tf.where(negative_risk < K.constant(0), -0.5 * negative_risk,
                             positive_risk + negative_risk)
        return objective

    def compile(self, method='PULB', optimizer='adam'):
        if method == 'PULB':
            self.model.compile(loss=self.LSIFLoss, optimizer=optimizer)
        elif method == 'PUSB':
            self.model.compile(loss=self.PUSBLoss, optimizer=optimizer)

    def fit(self, x, y, batch_size=4, train_epochs=40):
        early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=100, verbose=2, restore_best_weights=True)
        self.model.fit(x, y, batch_size=batch_size, epochs=train_epochs, shuffle=True,
                           callbacks=[early_stopping], verbose=0)


