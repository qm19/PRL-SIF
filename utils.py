
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import skfda
from skfda.misc.regularization import TikhonovRegularization
from skfda.misc.operators import LinearDifferentialOperator
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

global seed
seed = 27

def list_to_array(data, axis=0):
    for i in range(len(data)):
        if i == 0:
            data_all = data[i]
        else:
            data_all = np.concatenate((data_all, data[i]), axis=axis)
    return data_all


def standardization(data, type='standard'):
    if isinstance(data,list):
        data_all = list_to_array(data)
    else:
        data_all = data
    data_new = np.reshape(data_all, (-1, data_all.shape[-1]))

    if type == 'standard':
        standard = StandardScaler()
    elif type == '0-1':
        standard = MinMaxScaler()
    data_new_sta = standard.fit_transform(data_new)
    data_sta = np.reshape(data_new_sta, (data_all.shape))

    if isinstance(data,list):
        data_processed = []
        point = 0
        for i in range(len(data)):
            data_processed.append(data_sta[point:point+data[i].shape[0]])
            point += data[i].shape[0]
    else:
        data_processed = data_sta

    return data_processed


def function_trans(data, basis_type='BSpline', n_basis=20, n_component=4):
    t = np.linspace(0, 1, data.shape[1])
    data_transform = []
    for i in range(data.shape[2]):
        x = data[:, :, i]
        fd = skfda.FDataGrid(data_matrix=x, grid_points=t)
        if basis_type == 'Fourier':
            basis = skfda.representation.basis.Fourier(n_basis=n_basis)
            period = data.shape[1]
            w = 2 * np.pi / period
            smoother = skfda.preprocessing.smoothing.BasisSmoother(
                basis,
                method='svd',
                regularization=TikhonovRegularization(
                    LinearDifferentialOperator([0, w ** 2, 0, 1]),
                    regularization_parameter=0.001
                ),
                return_basis=False)
        elif basis_type == 'BSpline':
            basis = skfda.representation.basis.BSpline(n_basis=n_basis)
            smoother = skfda.preprocessing.smoothing.BasisSmoother(
                basis,
                method='svd',
                regularization=TikhonovRegularization(
                    LinearDifferentialOperator(2),
                    regularization_parameter=0.001
                ),
                return_basis=False)
        fd_smooth = smoother.fit_transform(fd)
        X_basis = fd_smooth.to_basis(basis)

        fpca = FPCA(n_components=n_component)
        X_fpca = fpca.fit_transform(X_basis)
        data_transform.append(X_fpca.T)

    return np.array(data_transform).reshape([-1, data.shape[0]]).T


def label_mechanism(data, label, positive_num=200, train_ratio=0.8, type='LB', esitimator='Linear'):
    # 对functional data进行标记，获得PU数据集，并划分训练集和测试集
    if isinstance(data, list):
        data_pro = list_to_array(data)
        label = list_to_array(label)
    else:
        data_pro = data

    true_class = -np.array(label) * 2 + 1  # 将0（正常状态）转换成1，将1（故障状态）转换为-1
    X_train, X_test, Y_train, Y_test = train_test_split(data_pro, true_class, test_size=1 - train_ratio,
                                                        random_state=seed)
    if type == 'LB':
        if esitimator == 'Linear':
            model = LogisticRegression()
        elif esitimator == 'SVC':
            model = SVC(probability=True)
        model.fit(data_pro, label)
        label_pro = 1 - model.predict_proba(X_train)[:, 1]
        label_pro[Y_train == -1] = 0
    elif type == 'SCAR':
        label_pro = np.ones((len(X_train)))
        label_pro[Y_train == -1] = 0

    X_select_df = pd.DataFrame(X_train).sample(positive_num, weights=label_pro, random_state=seed)
    L_train = np.zeros(Y_train.shape) - 1
    L_train[X_select_df.index] = 1

    return X_train, Y_train, L_train, X_test, Y_test


def label_mechanism_error(data, label, positive_num=200, train_ratio=0.8, error_ratio=0.2, type='LB', esitimator='Linear'):
    # 对functional data进行标记，获得PU数据集，并划分训练集和测试集，并对训练集的label添加噪声
    if isinstance(data, list):
        data_pro = list_to_array(data)
        label = list_to_array(label)
    else:
        data_pro = data

    true_class = -np.array(label) * 2 + 1  # 将0（正常状态）转换成1，将1（故障状态）转换为-1
    X_train, X_test, Y_train, Y_test = train_test_split(data_pro, true_class, test_size=1 - train_ratio,
                                                        random_state=seed)

    if esitimator == 'Linear':
        model = LogisticRegression()
    elif esitimator == 'SVC':
        model = SVC(probability=True)
    model.fit(data_pro, label) # 得到样本故障的概率
    label_pro = 1 - model.predict_proba(X_train)[:, 1]
    label_pro[Y_train == -1] = 0

    # 得到训练集中标记正确的样本
    label_right_num = positive_num - int(positive_num*error_ratio)
    X_select_df = pd.DataFrame(X_train).sample(label_right_num, weights=label_pro, random_state=seed)
    L_train = np.zeros(Y_train.shape) - 1
    L_train[X_select_df.index] = 1

    # 得到训练集中标记错误的样本
    label_error_pro = 1 - model.predict_proba(X_train)[:, 1]
    label_error_pro[Y_train == 1] = 0
    label_error_num = int(positive_num*error_ratio)
    X_label_error_df = pd.DataFrame(X_train).sample(label_error_num, weights=1-label_error_pro, random_state=seed)
    L_train[X_label_error_df.index] = 1

    return X_train, Y_train, L_train, X_test, Y_test


def label_mechanism_raw(data, label, data_raw, positive_num=200, train_ratio=0.8, type='LB', esitimator='Linear'):
    # 对raw sensor data进行标记，获得PU数据集，并划分训练集和测试集
    true_class = -np.array(label) * 2 + 1  # 将0（正常状态）转换成1，将1（故障状态）转换为-1
    X_train, X_test, Y_train, Y_test = train_test_split(data, true_class, test_size=1 - train_ratio,
                                                        random_state=seed)
    X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(data_raw, true_class, test_size=1 - train_ratio,
                                                                        random_state=seed)
    if type == 'LB':
        if esitimator == 'Linear':
            model = LogisticRegression()
        elif esitimator == 'SVC':
            model = SVC(probability=True)
        model.fit(data, label)
        label_pro = 1 - model.predict_proba(X_train)[:, 1]
        label_pro[Y_train == -1] = 0
    elif type == 'SCAR':
        label_pro = np.ones((len(X_train)))
        label_pro[Y_train == -1] = 0

    X_select_df = pd.DataFrame(X_train).sample(positive_num, weights=label_pro, random_state=seed)
    L_train = np.zeros(Y_train.shape) - 1
    L_train[X_select_df.index] = 1

    return X_train_raw, Y_train, L_train, X_test_raw, Y_test

def label_mechanism_ratio(data, label, label_ratio=0.2, train_ratio=0.8, type='LB', esitimator='Linear'):
    # 对functional data进行标记，获得PU数据集，并划分训练集和测试集
    if isinstance(data, list):
        data_pro = list_to_array(data)
        label = list_to_array(label)
    else:
        data_pro = data

    true_class = -np.array(label) * 2 + 1  # 将0（正常状态）转换成1，将1（故障状态）转换为-1
    X_train, X_test, Y_train, Y_test = train_test_split(data_pro, true_class, test_size=1 - train_ratio,
                                                        random_state=seed)
    positive_num = int(len(Y_train) * label_ratio)
    if type == 'LB':
        if esitimator == 'Linear':
            model = LogisticRegression()
        elif esitimator == 'SVC':
            model = SVC(probability=True)
        model.fit(data_pro, label)
        label_pro = 1 - model.predict_proba(X_train)[:, 1]
        label_pro[Y_train == -1] = 0
    elif type == 'SCAR':
        label_pro = np.ones((len(X_train)))
        label_pro[Y_train == -1] = 0

    X_select_df = pd.DataFrame(X_train).sample(positive_num, weights=label_pro, random_state=seed)
    L_train = np.zeros(Y_train.shape) - 1
    L_train[X_select_df.index] = 1

    return X_train, Y_train, L_train, X_test, Y_test


def get_feature_ori(args):
    _COL = ['风向绝对值', '5秒偏航对风平均值', '机舱气象站风速', '测风塔环境温度',
            '变频器入口压力', '变频器出口压力', '变频器电网侧电压', '变频器电网侧无功功率', '变频器出口温度',
            '轮毂温度', '无功功率控制状态', 'x方向振动值', 'y方向振动值',
            '叶片1角度', '叶片1电池箱温度', '叶片1变桨电机温度', '叶片2角度', '叶片2电池箱温度', '叶片2变桨电机温度', '叶片3角度', '叶片3电池箱温度', '叶片3变桨电机温度',
            '机舱控制柜温度', '机舱温度', '主轴承温度1', '主轴承温度2', '液压制动压力']

    _file_path = args
    _df = pd.read_csv(_file_path, usecols=_COL)
    # 删除全0行
    _df = _df.loc[~(_df == 0).all(axis=1), :]
    # 获取文件名
    _file_name = _file_path.split('/')[-1]

    time_step = 50
    fre = _df.shape[0] // time_step
    _dict_return = np.array([np.mean(_df.iloc[i * fre:(i + 1) * fre, :], axis=0) for i in range(time_step)])

    return _dict_return, _file_name

def get_feature(args):
    _COL = ['风向绝对值', '5秒偏航对风平均值', '机舱气象站风速', '测风塔环境温度',
            '变频器入口压力', '变频器出口压力', '变频器电网侧电压', '变频器电网侧无功功率', '变频器出口温度',
            '轮毂温度', '无功功率控制状态', 'x方向振动值', 'y方向振动值',
            '叶片1角度', '叶片1电池箱温度', '叶片1变桨电机温度', '叶片2角度', '叶片2电池箱温度', '叶片2变桨电机温度', '叶片3角度', '叶片3电池箱温度', '叶片3变桨电机温度',
            '机舱控制柜温度', '机舱温度', '主轴承温度1', '主轴承温度2', '液压制动压力']

    _file_path = args
    _df = pd.read_csv(_file_path, usecols=_COL)
    # 删除全0行
    _df = _df.loc[~(_df == 0).all(axis=1), :]
    # 获取文件名
    _file_name = _file_path.split('/')[-1]

    _dict_return = []
    if (_df.shape[0] != 0):
        for col in _COL:
            _dict_return.append(np.mean(_df[col]))
            _dict_return.append(np.min(_df[col]))
            _dict_return.append(np.max(_df[col]))
            _dict_return.append(np.var(_df[col]))
            _dict_return.append(np.ptp(_df[col]))
            _dict_return.append(np.median(_df[col]))
    return np.array(_dict_return), _file_name


def get_data_WT(turbine_ID='002', train_path='../wind turbines/风机叶片开裂预警数据/data/train/',
             label_path='../wind turbines/风机叶片开裂预警数据/data/train_labels.csv'):

    label_df = pd.read_csv(label_path)
    label_df.set_index('file_name', inplace=True)

    data, file_name, label = [], [], []
    f1 = turbine_ID
    nargs = os.listdir(train_path + f1)
    nargs = [(train_path + f1 + '/' + _i) for _i in nargs]
    for item in nargs:
        data_temp, name_temp = get_feature_ori(item)
        # 部分文件出现全零值时跳出
        if np.isnan(data_temp).any():
            continue
        data.append(data_temp)
        file_name.append(name_temp)
        label.append(label_df.loc[name_temp, 'ret'])
    data = np.array(data)

    return data, label

def get_feature_WT(turbine_ID='002', train_path='../wind turbines/风机叶片开裂预警数据/data/train/',
             label_path='../wind turbines/风机叶片开裂预警数据/data/train_labels.csv'):

    label_df = pd.read_csv(label_path)
    label_df.set_index('file_name', inplace=True)

    data, file_name, label = [], [], []
    f1 = turbine_ID
    nargs = os.listdir(train_path + f1)
    nargs = [(train_path + f1 + '/' + _i) for _i in nargs]
    for item in nargs:
        data_temp, name_temp = get_feature(item)
        # 部分文件出现全零值时跳出
        if data_temp.shape[0] == 0:
            continue
        data.append(data_temp)
        file_name.append(name_temp)
        label.append(label_df.loc[name_temp, 'ret'])
    data = np.array(data)
    data = standardization(data, type='0-1')

    return data, label

def classifier_result_save(score_df, saveFile_name):
    writer = pd.ExcelWriter(saveFile_name)
    score_df.to_excel(writer,'Page 1',float_format='%.3f',index=True) # float_format 控制精度
    writer.save()

def plot_funcData(data):
    t = np.linspace(0, 1, data.shape[1])
    X = skfda.FDataGrid(data_matrix=data, grid_points=t)
    X.plot()

def plot_FPCA(data):
    t = np.linspace(0, 1, data.shape[1])
    fig, axes = plt.subplots(nrows=3, ncols=4)
    basis = skfda.representation.basis.BSpline(n_basis=20)
    for n in range(12):
        x = data[:, :, n]
        fd = skfda.FDataGrid(data_matrix=x, grid_points=t)
        smoother = skfda.preprocessing.smoothing.BasisSmoother(
            basis,
            method='svd',
            regularization=TikhonovRegularization(
                LinearDifferentialOperator(2),
                regularization_parameter=0
            ),
            return_basis=False)
        fd_smooth = smoother.fit_transform(fd)
        X_basis = fd_smooth.to_basis(basis)

        fpca = FPCA(n_components=4)
        X_fpca = fpca.fit_transform(X_basis)
        fd_fpca = fpca.inverse_transform(X_fpca)
        # fd_fpca.plot()

        ax = axes.ravel()[n]
        fig = fd_fpca.plot(axes=ax)
    fig.tight_layout()
    plt.show()

