# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:02:13 2020
benchmark2：即不使用卷积自编码器进行重构，直接使用简单的全连接神经网络进行NNPU learning
@author: MS
"""

from time import time
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping

import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import pickle
np.random.seed(27)

def dataset_process(dataPath):
    with open(dataPath, 'rb') as f:  # python路径要用反斜杠
        X_train = pickle.load(f)
        Y_train = pickle.load(f)
        L_train = pickle.load(f)
        X_test = pickle.load(f)
        Y_test = pickle.load(f)
    return X_train, Y_train, L_train, X_test, Y_test


def fullyConnected(input_dims, act='relu'):
    x = Input(shape=(input_dims,), name='input')
    h = x
    h = Dense(100, activation=act, name='encoder_0')(h)
    h = Dense(40, activation=act, name='encoder_1')(h)
    h = Dense(40, activation=act, name='encoder_2')(h)

    # hidden layer
    h = Dense(10, name='hidden_layer')(h)

    return Model(inputs=x, outputs=h)


class NNPU(object):
    def __init__(self,
                 input_dims,
                 prior=0.85,
                 beta=0,
                 gamma=1):

        super(NNPU, self).__init__()

        self.input_dims = input_dims
        self.prior = prior
        self.beta = beta
        self.gamma = gamma
        self.fullyConnected = fullyConnected(self.input_dims, 'elu')
        self.hidden = self.fullyConnected.get_layer(name='hidden_layer').output

        # prepare NNPU model
        self.hidden = BatchNormalization(name='batch_normalization')(self.hidden)
        self.full_connected = Dense(10, name='full_connected_1', activation='elu')(self.hidden)
        self.full_connected = Dense(10, name='full_connected_2', activation='elu')(self.full_connected)
        self.probability = Dense(1, name='label_prediction', activation='elu')(self.full_connected)
        self.model = Model(inputs=self.fullyConnected.input,
                           outputs=self.probability)

    def load_weights(self, weights_path):  # load weights of NNPU model
        self.model.load_weights(weights_path)
        print('weights is loaded successfully.')

    def predict_label(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        y_pred = tf.clip_by_value(q, -1, 1)
        y_pred = K.eval(y_pred)
        y_label = np.where(y_pred > 0, 1, -1)
        y_pred = (y_pred+1)/2
        return y_pred, y_label

    def PULoss(self, y_true, y_pred):
        sigmoid_loss = (lambda x: K.sigmoid(-x))
        positive_index, unlabeled_index = (y_true + 1) / 2, (-y_true + 1) / 2
        n_positive, n_unlabeled = K.max([1, K.sum(positive_index)]), K.max([1, K.sum(unlabeled_index)])
        # y_pred = tf.clip_by_value(y_pred, 0, 1)
        y_positive = sigmoid_loss(y_pred)
        y_unlabeled = sigmoid_loss(-y_pred)
        positive_risk = K.sum(self.prior / n_positive * positive_index * y_positive)
        negative_risk = K.sum(unlabeled_index * y_unlabeled / n_unlabeled) - K.sum(
            self.prior * positive_index * y_unlabeled / n_positive)

        objective = tf.where(negative_risk < K.constant(self.beta), -self.gamma * negative_risk,
                             positive_risk + negative_risk)
        '''
        objective = positive_risk + negative_risk
        '''
        return objective

    def compile(self, loss=PULoss, optimizer='adam'):
        self.model.compile(loss=loss, optimizer=optimizer)

    def fit(self, x, y, epochs=200, batch_size=16):
        early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=50, verbose=2)
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, shuffle=True,
                       callbacks=[early_stopping], verbose=2)


def classifier_result_save(score, saveFile_name):
    score_df = pd.DataFrame(score)
    score_df.columns = ['num', 'train precision', 'train F1-score', 'train AUC', 'test precision', 'test F1-score', 'test AUC']
    writer = pd.ExcelWriter(saveFile_name)
    score_df.to_excel(writer,'Page 1',float_format='%.3f') # float_format 控制精度
    writer.save()

if __name__ == "__main__":
    fileName = 'LB_006_200'
    piror_estimate_method = 'KM1'
    prior = 0.281
    Train = False

    saveFile_name = '../Results/nnPU_' + fileName + '_'+ piror_estimate_method +'_result.xls'
    list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    score = np.empty((len(list), 7))
    m = 0
    # setting the hyper parameters
    batch_size = 16
    gamma = 1
    train_epochs = 1000
    optimizer = 'adam'  # SGD(lr=0.01, momentum=0.99)
    dataPath = '../Data/' + fileName
    # load dataset
    x, y_train_true, y, x_test, y_test = dataset_process(dataPath=dataPath)
    for num in list:
        save_nnPU_dic = './saveWeights/nnPU_' + fileName + '_' + piror_estimate_method + '_' + str(num) + '.h5'
        if Train:
            nnpu_weights = None
        else:
            nnpu_weights = save_nnPU_dic
        # Define NNPU model
        nnpu = NNPU(input_dims=x.shape[1], prior=prior, beta=0, gamma=gamma)
        # plot_model(nnpu.model, to_file='benchmark2.png', show_shapes=True)
        nnpu.model.summary()

        t0 = time()

        # Pretrain fullyConnecteds before clustering
        nnpu.compile(loss=nnpu.PULoss, optimizer=optimizer)
        if nnpu_weights is None:
            nnpu.fit(x, y, batch_size=batch_size, epochs=train_epochs)
            nnpu.model.save_weights(save_nnPU_dic)
        else:
            nnpu.load_weights(nnpu_weights)

        # Show the final results
        y_pred_train, y_pred_label_train = nnpu.predict_label(x)
        y_pred_test, y_pred_label_test = nnpu.predict_label(x_test)
        print('train time: %d seconds.' % int(time() - t0))

        from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
        score[m, :] = [num,
                       accuracy_score(-y_train_true, -y_pred_label_train),
                       f1_score(-y_train_true, -y_pred_label_train),
                       roc_auc_score(y_train_true, y_pred_train),
                       accuracy_score(-y_test, -y_pred_label_test),
                       f1_score(-y_test, -y_pred_label_test),
                       roc_auc_score(y_test, y_pred_test)]
        m += 1
    classifier_result_save(score, saveFile_name)
    print(np.mean(score[:,1:], axis=0))
    print(np.std(score[:, 1:], axis=0))

