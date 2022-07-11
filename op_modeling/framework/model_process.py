#!/usr/bin/python

from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from termcolor import cprint

from framework.model_packing import OperatorHandle
from framework.model_packing import deserialization


def calc_percentage_error(y1, y2):
    """
    计算相对误差
    :param y1:
    :param y2:
    :return:
    """
    return np.abs(y1 - y2) / (np.finfo(float).eps + y1)


class ModelProcess:
    """
    该类抽象了整个训练的流程，针对剥离出来的内容，设置基类保证行为有明确定义
    """

    def __init__(self, op_handle):
        """
        :param op_handle: 模型本身需要开发人员自己定义，然后设置到这个流程中
        """

        self.op_handle: OperatorHandle = op_handle

    def print_feature_importance(self, feature_names):
        feature_importance = self.op_handle.model.get_feature_importance()
        if feature_importance is not None and feature_names is not None:
            if len(feature_importance) != len(feature_names):
                raise Exception("feature importance mismatch feature names")

            features = []
            for i, feature_name in enumerate(feature_names):
                features.append(dict(index=f"F{i}", name=feature_name, importance=feature_importance[i]))
            cprint("The importance of features is as follows:", color='red')
            df = pd.DataFrame(features).sort_values(by=['importance'], ascending=False, ignore_index=True)
            cprint(df, color='green')

    def train(self, data, labels, n_splits=5, feature_names=None):
        """
        训练过程，可以控制是否进行交叉验证和打印特征重要性
        """
        labels = labels.squeeze()
        # 是否进行交叉验证
        if n_splits > 0:
            rkf = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=0)
            pd.set_option('display.max_columns', None, 'display.width', 5000)

            cv_scores = []
            for k, (train, test) in enumerate(rkf.split(data)):
                if len(labels.shape) == 1:
                    self.op_handle.model.fit(data[train, :], labels[train])
                    ret = dict(name=f"{k}-th cross validation", **self.evaluate_stage(data[test, :], labels[test]))
                else:
                    self.op_handle.model.fit(data[train, :], labels[train, :])
                    ret = dict(name=f"{k}-th cross validation", **self.evaluate_stage(data[test, :], labels[test, :]))
                cv_scores.append(ret)

            cprint(str(pd.DataFrame(cv_scores)), color='green')

        self.train_stage(data, labels)
        # 如果有特征重要性信息，则打印重要性信息
        if feature_names is None:
            return
        self.print_feature_importance(feature_names)

    def load_model(self, save_path):
        """
        载入已经保存的模型，用于预测
        """
        self.op_handle = deserialization(save_path)

    def train_stage(self, train_data, train_label):
        self.op_handle.model.fit(train_data, train_label)

    def evaluate_stage(self, test_data, test_label):
        """
        此处验收的标准应该是一致的
        :test_data 测试样本的特征
        :test_label 测试样本的标签，即训练时间
        """
        test_label = test_label.squeeze()
        test_predict = self.op_handle.model.predict(test_data)

        scores = {
            'r2_score': metrics.r2_score,
            'medium_percentage_error': lambda y1, y2: np.percentile(calc_percentage_error(y1, y2), 50),
            '80_percentage_error': lambda y1, y2: np.percentile(calc_percentage_error(y1, y2), 80),
            'max_percentage_error': lambda y1, y2: np.max(calc_percentage_error(y1, y2))
        }
        ret = dict()
        for i_score_name, i_fcn in scores.items():
            i_score_val = i_fcn(test_label, test_predict)
            ret[i_score_name] = i_score_val

        return ret
