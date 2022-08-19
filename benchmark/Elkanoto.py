# -*- coding: utf-8 -*-
"""
benchmark1：经典PU learning方法
"Learning classifiers from only positive and unlabeled data"
(published in Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, ACM, 2008).
Based on the paper A bagging SVM to learn from positive and unlabeled examples (2013) by Mordelet and Vert.
1、Classic Elkanoto method
2、Weighted Elkanoto method
3、Bagging-based PU-learning
@author: MS
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from pulearn import ElkanotoPuClassifier, WeightedElkanotoPuClassifier, BaggingPuClassifier
from sklearn.svm import SVC
import joblib
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

def classifier_result_save(score, saveFile_name):
    score_df = pd.DataFrame(score)
    score_df.columns = ['num', 'train precision', 'train F1-score', 'train AUC', 'test precision', 'test F1-score', 'test AUC']
    writer = pd.ExcelWriter(saveFile_name)
    score_df.to_excel(writer,'Page 1',float_format='%.3f') # float_format 控制精度
    writer.save()

#------------------------------------------------main--------------------------------------------------#
if __name__ == "__main__":
    fileName = 'LB_006_200'
    Train = False

    saveFile_name = '../Results/ElkanotoPuLearning_' + fileName + '_result.xls'
    dataPath = '../Data/' + fileName
    list = [1,2,3,4,5,6,7,8,9,10]
    score = np.empty((len(list), 7))
    m = 0
    for num in list:
        Model = 1 # 1--ElkanotoPuLearning, 2--WeightedElkanotoPuLearning, 3--BaggingPuLearning
        pretrain = Train # True--fit model, False--load model
        # load dataset
        x, y_train_true, y, x_test, y_test = dataset_process(dataPath=dataPath)
        svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)

        if Model == 1:
            save_weights = './saveWeights/ElkanotoPuLearning_' + fileName + '_' + str(num) + '.pkl'
            if pretrain:
                pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
                pu_estimator.fit(x, y)
                joblib.dump(pu_estimator, save_weights)
            else:
                pu_estimator = joblib.load(save_weights)
        elif Model == 2:
            save_weights = './saveWeights/WeightedElkanotoPuLearning_' + fileName + '_' + str(num) + '.pkl'
            if pretrain:
                pu_estimator = WeightedElkanotoPuClassifier(estimator=svc, labeled=20, unlabeled=20, hold_out_ratio=0.2)
                pu_estimator.fit(x, y)
                joblib.dump(pu_estimator, save_weights)
            else:
                pu_estimator = joblib.load(save_weights)
        else:
            save_weights = './saveWeights/BaggingPuLearning_' + fileName + '_' + str(num) + '.pkl'
            if pretrain:
                pu_estimator = BaggingPuClassifier(base_estimator=svc, n_estimators=15)
                pu_estimator.fit(x, y)
                joblib.dump(pu_estimator, save_weights)
            else:
                pu_estimator = joblib.load(save_weights)

        y_pred_train = pu_estimator.predict_proba(x)
        y_pred_test = pu_estimator.predict_proba(x_test)
        y_pred_label_train = pu_estimator.predict(x)
        y_pred_label_test = pu_estimator.predict(x_test)

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
