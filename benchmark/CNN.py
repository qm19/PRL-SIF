from utils import *
import warnings
warnings.filterwarnings("ignore")
from time import clock

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from keras.utils import to_categorical

from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
np.random.seed(27)

import keras.backend as K
from keras.models import Model
# from keras.optimizers import SGD
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras import regularizers

import argparse
global seed
seed = 27


class CNN(object):
    def __init__(self,
                 input_dims,
                 act_fun = 'sigmoid',
                 l2_parameter=0.001):

        super(CNN, self).__init__()

        self.input_dims = input_dims
        self.l2_para = l2_parameter

        self.inputlayers = Input(shape=self.input_dims, name='input')

        self.CNN = Conv1D(8, 10, padding='same', activation='relu', name='conv_1')(self.inputlayers)
        self.CNN = MaxPooling1D(4, padding='same', name='pooling_1')(self.CNN)
        self.CNN = Conv1D(6, 5, padding='same', activation='relu', name='conv_2')(self.CNN)
        self.CNN = MaxPooling1D(3, padding='same', name='pooling_2')(self.CNN)
        self.full_connected = Flatten(name='flatten')(self.CNN)
        self.full_connected = Dense(50, name='full_connected_1', activation='elu',
                                    kernel_regularizer=regularizers.l2(self.l2_para),
                                    bias_regularizer=regularizers.l2(self.l2_para))(self.full_connected)
        self.full_connected = Dense(10, name='full_connected_2', activation='elu',
                                    kernel_regularizer=regularizers.l2(self.l2_para),
                                    bias_regularizer=regularizers.l2(self.l2_para))(self.full_connected)
        self.probability = Dense(2, name='label_prediction', activation=act_fun,
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

    def compile(self, optimizer='adam'):
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def fit(self, x, y, batch_size=4, train_epochs=40):
        early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=100, verbose=2, restore_best_weights=True)
        self.model.fit(x, y, batch_size=batch_size, epochs=train_epochs, shuffle=True,
                       callbacks=[early_stopping], verbose=0)


def parse():
    '''Parse system arguments.'''
    parser = argparse.ArgumentParser(
        description='Positive Unlabeled Learning with Labeling Bias',
        usage='CNN.py --method <string> --label_mechanism <string> --dataset_name <string>' +
              '--train <bool> --label_number <integer> --label_method <string> --act_func <string> --l2_weight <float>' +
              '--batchsize <integer> --epoch <integer> --runs <integer> --save <bool>'
    )
    parser.add_argument('--method', type=str, default='CNN', choices=['CNN'],
                        help='The method for PU learning.')
    parser.add_argument('--label_error_ratio', type=int, default=40, help='The percentage of labeling error in the training set.')
    parser.add_argument('--dataset_name', type=str, default='002', help='The filename of experiment dataset.')
    parser.add_argument('--train', type=bool, default=True,
                        help='Train a model or load a trained model.')
    parser.add_argument('--label_method', type=str, default='Linear', choices=['Linear', 'SVC'],
                        help='Method of estimating P(y|X).')
    parser.add_argument('--act_func', type=str, default='softmax', choices=['sigmoid', 'relu', 'softmax'],
                        help='The activation function of the final layer.')
    parser.add_argument('--l2_weight', type=float, default=0.001, help='The weight of the L2 regularization.')
    parser.add_argument('--batchsize', type=int, default=16, help='The batch size of the NN classifier.')
    parser.add_argument('--epoch', type=int, default=1000, help='The max epoch of the NN classifier training.')
    parser.add_argument('--runs', type=int, default=10, help='Number of independent runs.')
    parser.add_argument('--save', type=bool, default=True, help='Weather save the result to a excel file.')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse()
    # Read dataset
    data_raw, label = get_data_WT(turbine_ID=args.dataset_name)
    # Data Normalization
    data_pro = standardization(data_raw, type='standard')
    # functional analysis and functional PCA for preprocessing
    data = function_trans(data_pro)


    # Label
    X_train, Y_train, L_train, X_test, Y_test = label_error(data, label, data_pro, error_ratio=args.label_error_ratio/100,
                                                            train_ratio=0.8, esitimator=args.label_method)
    X_train = X_train.transpose(0,2,1)
    X_test = X_test.transpose(0,2,1)
    L_train = to_categorical((L_train+1)/2)

    # Show the log information
    print(
        '\nMethod: {} \nDataset_name: {} \nPositive number in training set: {} \nLabeled number: {} in {} independent run(s) ...'
            .format(args.method, args.dataset_name, str(int((len(Y_train) + sum(Y_train)) / 2)),
                    str(int((len(Y_train) + sum(Y_train)) / 2)), args.runs))
    scores = []
    times = []
    for i in range(args.runs):
        save_PU_dic = './saveWeights/' + args.method + '_' + args.dataset_name + '_LE_' + str(args.label_error_ratio) + '_' + str(i) + '.h5'
        if args.train:
            pu_weights = None
        else:
            pu_weights = save_PU_dic

        start_time = clock()
        cnn = CNN(input_dims=[X_train.shape[1], X_train.shape[2]], act_fun=args.act_func, l2_parameter=args.l2_weight)
        if i == 0:
            cnn.model.summary()
        cnn.compile(optimizer='adam')

        # Train the classifier or load the classifier
        if pu_weights is None:
            cnn.fit(X_train, L_train, batch_size=args.batchsize, train_epochs=args.epoch)
            cnn.model.save_weights(save_PU_dic)
        else:
            cnn.load_weights(pu_weights)

        # Get the prediction results
        f_train = cnn.model.predict(X_train)[:,1]
        f_test = cnn.model.predict(X_test)[:,1]
        y_pred_label_train = np.where(f_train>0.5, 1, -1)
        y_pred_label_test = np.where(f_test>0.5, 1, -1)
        times.append(clock() - start_time)
        scores.append([accuracy_score(-Y_train, -y_pred_label_train),
                       f1_score(-Y_train, -y_pred_label_train),
                       roc_auc_score(Y_train, f_train),
                       accuracy_score(-Y_test, -y_pred_label_test),
                       f1_score(-Y_test, -y_pred_label_test),
                       roc_auc_score(Y_test, f_test)])

    print('ave_run_time:\t\t{:.3f}s'.format(np.mean(times)))
    print('------------------------------')
    print('Metrics:')
    df_scores = pd.DataFrame(scores, columns=['train precision', 'train F1-score', 'train AUC', 'test precision',
                                              'test F1-score', 'test AUC'])
    for metric in df_scores.columns.tolist():
        print('{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(), df_scores[metric].std()))
    if args.save:
        classifier_result_save(df_scores,
                               '../Results/' + args.method + '_' + args.dataset_name + '_LE_' + str(args.label_error_ratio) + '.xls')


