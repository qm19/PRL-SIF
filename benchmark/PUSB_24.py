from utils import *
import warnings
warnings.filterwarnings("ignore")
from time import clock
import pickle

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from PUSB_NN import PUNN
from Kernel_MPE import wrapper

import argparse
global seed
seed = 27


def parse():
    '''Parse system arguments.'''
    parser = argparse.ArgumentParser(
        description='Positive Unlabeled Learning with Labeling Bias',
        usage='PUSB.py --method <string> --label_mechanism <string> --dataset_name <string> --readRowData <bool>' +
              '--train <bool> --label_number <integer> --label_method <string> --act_func <string> --l2_weight <float>' +
              '--batchsize <integer> --epoch <integer> --runs <integer> --save <bool>'
    )
    parser.add_argument('--method', type=str, default='PUSB', choices=['PUSB'],
                        help='The method for PU learning.')
    parser.add_argument('--label_mechanism', type=str, default='SCAR', choices=['SCAR', 'LB'],
                        help='Label mechanism of PU training set.')
    parser.add_argument('--train', type=bool, default=True,
                        help='Train a model or load a trained model.')
    parser.add_argument('--label_ratio', type=float, default=0.2,
                        help='The percentage of positive labeled sample in the training PU set.')
    parser.add_argument('--label_method', type=str, default='Linear', choices=['Linear', 'SVC'],
                        help='Method of estimating P(y|X).')
    parser.add_argument('--act_func', type=str, default='sigmoid', choices=['sigmoid', 'relu'],
                        help='The activation function of the final layer.')
    parser.add_argument('--l2_weight', type=float, default=0.001, help='The weight of the L2 regularization.')
    parser.add_argument('--batchsize', type=int, default=16, help='The batch size of the NN classifier.')
    parser.add_argument('--epoch', type=int, default=1000, help='The max epoch of the NN classifier training.')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs.')
    parser.add_argument('--save', type=bool, default=True, help='Weather save the result to a excel file.')
    return parser.parse_args()


if __name__ == '__main__':
    piror_estimate_method = 'KM'

    # Parse arguments
    args = parse()
    dataset_name_list = ['002', '004', '005', '006', '011', '012', '015', '017', '018', '019', '020', '029',
                         '037', '043', '046', '049', '052', '055', '057', '060', '063', '064', '080']
    scores = []
    times = []
    for dataset_name in dataset_name_list:
        dataPath = '../Data/' + args.label_mechanism + '_' + dataset_name + '_24'
        with open(dataPath, 'rb') as f:  # python路径要用反斜杠
            X_train = pickle.load(f)
            Y_train = pickle.load(f)
            L_train = pickle.load(f)
            X_test = pickle.load(f)
            Y_test = pickle.load(f)

        # Esitimate the class prior
        X_mixture = X_train[L_train == -1]
        X_component = X_train[L_train == 1]
        piror, _ = wrapper(X_mixture, X_component)

        # Show the log information
        print(
            '\nMethod: {} \nLabel mechanism: {} \nDataset_name: {} \nPositive number in training set: {} \nLabeling ratio: {} in {} independent run(s) ...'
                .format(args.method, args.label_mechanism, dataset_name, str(int((len(Y_train) + sum(Y_train)) / 2)),
                        args.label_ratio, args.runs))
        for i in range(args.runs):
            save_PU_dic = './saveWeights/' + args.method + '_' + args.label_mechanism + '_' + dataset_name + '_' \
                          + piror_estimate_method + '_24' + '.h5'
            if args.train:
                pu_weights = None
            else:
                pu_weights = save_PU_dic

            start_time = clock()
            punn = PUNN(input_dims=X_train.shape[1], act_fun=args.act_func, l2_parameter=args.l2_weight, piror=piror)
            if i == 0:
                punn.model.summary()
            punn.compile(optimizer='adam', method=args.method)

            # Train the classifier or load the classifier
            if pu_weights is None:
                punn.fit(X_train, L_train, batch_size=args.batchsize, train_epochs=args.epoch)
                punn.model.save_weights(save_PU_dic)
            else:
                punn.load_weights(pu_weights)

            # Get the prediction results
            f_train = punn.predict(X_train)
            f_test = punn.predict(X_test)
            threshold = np.sort(f_train[:, 0])[-int((len(Y_train) + sum(Y_train)) * piror),]
            y_pred_label_train = punn.predict_label(X_train, threshold=threshold)
            y_pred_label_test = punn.predict_label(X_test, threshold=threshold)
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
                               '../Results/' + args.method + '_' + args.label_mechanism + '_' + piror_estimate_method +
                               '_24_result.xls')

