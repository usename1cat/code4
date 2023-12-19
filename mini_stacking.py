#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score


# 读取数据
train = pd.read_excel('train.xls')
test = pd.read_excel('test.xls')
test_y = test['class']
test = test.drop(['class'], axis=1)

ntrain = train.shape[0]   #读取矩阵第一维度的长度
ntest = test.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)


# 定义一个扩展类 SklearnHelper
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_

# 定义模型训练函数
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Random Forest 参数
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

# Extra Trees 参数
et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost 参数
ada_params = {
    'n_estimators': 500,
    'learning_rate': 0.75
}

# Gradient Boosting 参数
gb_params = {
    'n_estimators': 500,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier 参数
svc_params = {
    'kernel': 'linear',
    'C': 0.025
}

# 创建5个对象 代表5个模型
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# 建立特征数据和标签数据
y_train = train['class'].ravel()  # ravel函数，将多维数组降为一维
train = train.drop(['class'], axis=1)
x_train = train.values     # x_train为去除Survived后的所有特征量
x_test = test.values

# 模型训练
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)  # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)  # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)  # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)  # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)  # Support Vector Classifier

# 特征重要性分析
rf_feature = rf.feature_importances(x_train, y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train, y_train)

rf_features = list(rf_feature)

et_features = list(et_feature)
ada_features = list(ada_feature)
gb_features = list(gb_feature)
cols = train.columns.values

# 创建一个特征的DataFrame框架
feature_dataframe = pd.DataFrame({'features': cols,
                                  'Random Forest feature importances': rf_features,
                                  'Extra Trees  feature importances': et_features,
                                  'AdaBoost feature importances': ada_features,
                                  'Gradient Boost feature importances': gb_features
                                  })

# 支持中文
plt.bar(feature_dataframe['features'].values, feature_dataframe['Random Forest feature importances'].values,
        color=['b', 'r', 'g', 'y', 'c', 'm', 'y', 'k', 'c', 'g', 'b'])
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.title("随机森林模型特征重要性图", fontsize=15)  # 设置标题
plt.xlabel("特征名字", fontsize=10)  # 设置x轴
plt.ylabel("重要性", fontsize=10)  # 设置y轴
plt.show()  # 展示图片

plt.bar(feature_dataframe['features'].values, feature_dataframe['Extra Trees  feature importances'].values,
        color=['b', 'r', 'g', 'y', 'c', 'm', 'y', 'k', 'c', 'g', 'b'])
plt.title("Extra Trees模型特征重要性图", fontsize=15)  # 设置标题
plt.xlabel("特征名字", fontsize=10)  # 设置x轴
plt.ylabel("重要性", fontsize=10)  # 设置y轴
plt.show()  # 展示图片

plt.bar(feature_dataframe['features'].values, feature_dataframe['AdaBoost feature importances'].values,
        color=['b', 'r', 'g', 'y', 'c', 'm', 'y', 'k', 'c', 'g', 'b'])
plt.title("AdaBoost模型特征重要性图", fontsize=15)  # 设置标题
plt.xlabel("特征名字", fontsize=10)  # 设置x轴
plt.ylabel("重要性", fontsize=10)  # 设置y轴
plt.show()  # 展示图片

plt.bar(feature_dataframe['features'].values, feature_dataframe['Gradient Boost feature importances'].values,
        color=['b', 'r', 'g', 'y', 'c', 'm', 'y', 'k', 'c', 'g', 'b'])
plt.title("Gradient Boost模型特征重要性图", fontsize=15)  # 设置标题
plt.xlabel("特征名字", fontsize=10)  # 设置x轴
plt.ylabel("重要性", fontsize=10)  # 设置y轴
plt.show()  # 展示图片

# 所有模型平均特征重要性

feature_dataframe['mean'] = feature_dataframe.mean(axis=1)  # axis = 1 computes the mean row-wise
print(feature_dataframe.head(3))

plt.bar(feature_dataframe['features'].values, feature_dataframe['mean'].values,
        color=['b', 'r', 'g', 'y', 'c', 'm', 'y', 'k', 'c', 'g', 'b'])
plt.title("所有模型平均特征重要性图", fontsize=15)  # 设置标题
plt.xlabel("特征名字", fontsize=10)  # 设置x轴
plt.ylabel("重要性", fontsize=10)  # 设置y轴
plt.show()  # 展示图片

base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
                                       'ExtraTrees': et_oof_train.ravel(),
                                       'AdaBoost': ada_oof_train.ravel(),
                                       'GradientBoost': gb_oof_train.ravel(),
                                       'svc': svc_oof_train.ravel()
                                       })
print(base_predictions_train.head())

# 相关性分析
df_tmp1 = base_predictions_train[
    ['RandomForest', 'ExtraTrees', 'AdaBoost', 'GradientBoost', 'svc']]  # 获取数据项
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.heatmap(df_tmp1.corr(), cmap="YlGnBu", annot=True)  # 画相关性热力图
plt.title("五种模型预测结果相关性分析图", fontsize=15)  # 设置标题
plt.show()  # 展示图片

x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

# 元分类器建模
gbm = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=4,
    min_child_weight=2,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)
print(predictions)