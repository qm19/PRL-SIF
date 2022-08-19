# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:02:13 2020
@author: MS
"""

from time import time
import numpy as np
import keras.backend as K
from keras.models import Model
# from keras.optimizers import SGD
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, Callback

import tensorflow as tf
import pandas as pd
import math
from sklearn import preprocessing
from sklearn.cluster import KMeans
np.random.seed(27)
import pickle

def dataset_process(dataPath):
    with open(dataPath, 'rb') as f:  # python路径要用反斜杠
        X_train = pickle.load(f)
        Y_train = pickle.load(f)
        L_train = pickle.load(f)
        X_test = pickle.load(f)
        Y_test = pickle.load(f)
    return X_train, Y_train, L_train, X_test, Y_test


def autoencoder(input_dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    # input
    x = Input(shape=(input_dims,), name='input')
    h = x

    # internal layers in encoder
    # internal layers in encoder
    h = Dense(100, activation=act, name='encoder_0')(h)
    h = Dense(40, activation=act, name='encoder_1')(h)
    h = Dense(40, activation=act, name='encoder_2')(h)

    # hidden layer
    h = Dense(10, name='hidden_layer')(h)

    # internal layers in decoder
    h = Dense(40, activation=act, name='decoder_2')(h)
    h = Dense(40, activation=act, name='decoder_1')(h)

    # output
    h = Dense(input_dims, name='decoder_0')(h)

    return Model(inputs=x, outputs=h)


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters=2, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PUHN(object):
    def __init__(self,
                 input_dims,
                 trainSet,
                 prior=0.5,
                 beta=0,
                 gamma=1,
                 n_clusters=2,
                 alpha=1.0):

        super(PUHN, self).__init__()

        self.input_dims = input_dims
        self.prior = prior
        self.beta = beta
        self.gamma = gamma
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.trainSet = trainSet

        self.autoencoder = autoencoder(self.input_dims, 'elu')
        self.hidden = self.autoencoder.get_layer(name='hidden_layer').output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=self.hidden)
        self.hidden = BatchNormalization(name='batch_normalization')(self.hidden)

        self.clustering_layer = ClusteringLayer(self.n_clusters, alpha=self.alpha, name='clustering')(self.hidden)

        # prepare PUHN model
        self.full_connected = Dense(10, name='full_connected_1', activation='elu')(self.hidden)
        self.full_connected = Dense(10, name='full_connected_2', activation='elu')(self.full_connected)
        self.probability = Dense(1, name='label_prediction', activation='elu')(self.full_connected)
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[self.probability, self.autoencoder.output, self.clustering_layer])

        self.pretrained = False
        self.centers = []
        self.y_pred = []

    def pretrain(self, x, batch_size=256, epochs=200, optimizer='adam'):
        print('...Pretraining...')
        early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=50, verbose=1)
        self.autoencoder.compile(loss='mse', optimizer=optimizer)  # SGD(lr=0.01, momentum=0.9),
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping])
        self.autoencoder.save_weights(save_ae_dic)
        print('Pretrained weights are saved.')
        self.pretrained = True

    def load_weights(self, weights_path):  # load weights of PUHN model
        self.model.load_weights(weights_path)
        print('weights is loaded successfully.')

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        _, _, q = self.model.predict(x, verbose=0)
        return q #q.argmax(1)

    def predict_label(self, x):  # predict cluster labels using the output of clustering layer
        q, _, _ = self.model.predict(x, verbose=0)
        y_pred = tf.clip_by_value(q, -1, 1)
        y_pred = K.eval(y_pred)
        y_label = np.where(y_pred > 0, 1, -1)
        y_pred = (y_pred + 1) / 2
        return y_pred, y_label

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / K.sum(q,axis=0)
        return tf.transpose(tf.transpose(weight) / K.sum(weight,axis=1))

    def CLoss(self, y_true, y_pred):
        #y_true = self.target_distribution(y_pred)
        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

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

    def compile(self, loss=[PULoss, 'mse', CLoss], loss_weights=[1, 1, 1], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, x, y, epochs=200, batch_size=16, update_interval=40):
        # Step 1: pretrain
        if not self.pretrained and ae_weights is None:
            print('...pretraining autoencoders using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x, batch_size)
            self.pretrained = True
        elif ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        early_stopping2 = EarlyStopping(monitor='loss', min_delta=0.001, patience=50, verbose=2, restore_best_weights=True)
        for ite in range(math.ceil(epochs/update_interval)):
            _, _, q = self.model.predict(x, verbose=0)
            p = self.target_distribution(q)
            self.model.fit(x, [y, x, p], steps_per_epoch=batch_size, epochs=update_interval, shuffle=True,
                           callbacks=[early_stopping2], verbose=2)


def classifier_result_save(score, saveFile_name):
    score_df = pd.DataFrame(score)
    score_df.columns = ['num', 'train precision', 'train F1-score', 'train AUC', 'test precision', 'test F1-score', 'test AUC']
    writer = pd.ExcelWriter(saveFile_name)
    score_df.to_excel(writer,'Page 1',float_format='%.3f') # float_format 控制精度
    writer.save()


if __name__ == "__main__":
    fileName = 'LB_006_200'
    Train = False

    saveFile_name = '../Results/PUHN_' + fileName + '_result.xls'
    list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    score = np.empty((len(list), 7))
    m = 0
    # setting the hyper parameters
    pretrain_epochs = 200
    loss_weight_ae = 0.5
    loss_weight_c = 0.5
    batch_size = 16
    gamma = 1
    train_epochs = 200
    update_interval = 40
    optimizer = 'adam'  # SGD(lr=0.01, momentum=0.99)
    dataPath = '../Data/' + fileName

    # load dataset
    x, y_train_true, y, x_test, y_test = dataset_process(dataPath=dataPath)

    for num in list:
        save_ae_dic = './saveWeights/PUHN_AE_' + str(num) + '.h5'
        save_classifier_dic = './saveWeights/PUHN_' + fileName + '_' + str(num) + '.h5'
        if Train:
            ae_weights = save_ae_dic
            classifier_weights = None
        else:
            ae_weights = save_ae_dic
            classifier_weights = save_classifier_dic
        # Define PUHN model
        puhn = PUHN(input_dims=x.shape[1], prior=0.28, beta=0, gamma=gamma, trainSet=x, n_clusters=2)
        # plot_model(puhn.model, to_file='PUL_model_cluster.png', show_shapes=True)
        puhn.model.summary()
        t0 = time()

        # Pretrain autoencoders before clustering
        if ae_weights is None:
            puhn.pretrain(x, batch_size=batch_size, epochs=pretrain_epochs, optimizer=optimizer)

        # begin clustering, time not include pretraining part.
        puhn.compile(loss=[puhn.PULoss, 'mse', puhn.CLoss], loss_weights=[1, loss_weight_ae, loss_weight_c], optimizer=optimizer)
        if classifier_weights is None:
            puhn.fit(x, y, batch_size=batch_size, epochs=train_epochs, update_interval=update_interval)
            puhn.model.save_weights(save_classifier_dic)
        else:
            puhn.load_weights(classifier_weights)

        # Show the final results
        y_pred_train, y_pred_label_train = puhn.predict_label(x)
        y_pred_test, y_pred_label_test = puhn.predict_label(x_test)
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
    print(np.mean(score[:, 1:], axis=0))
    print(np.std(score[:, 1:], axis=0))
