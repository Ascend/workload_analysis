# ！/usr/bin/python
import logging
from abc import abstractmethod, ABCMeta

from scipy import optimize as opt
import pwlf

import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression


class ModelBase(metaclass=ABCMeta):
    """
    定义了模型需要支持的基本行为，参数更新方法，以及预测方法
    注意，在这里面尽量提取公有内容，减少update和infer的作用范围，尽量减少全量定制的可能。
    """

    def __init__(self):
        self.feature_importances_ = None

    @abstractmethod
    def fit(self, train_data, train_label):
        """
        需要定义其自身被训练的方法
        :return:
        """
        raise Exception("train is not impl.")

    @abstractmethod
    def predict(self, test_data):
        raise Exception("predict is not impl.")

    def get_feature_importance(self):
        """
        获取特征重要性，辅助建模
        """
        return self.feature_importances_


class CurveFitBasedModel(ModelBase):
    """
    曲线拟合类回归模型
    """

    def __init__(self, fcn_to_fit, normalize_transformer=None):
        """
        :fcn_to_fit: 函数句柄，待拟合的函数
        :normalize_transformer: 归一化函数
        """
        super().__init__()
        self.fcn_to_fit = fcn_to_fit
        self.parameters = None
        self.transformer = normalize_transformer

    def fit(self, train_data, train_label):
        if self.transformer:
            train_data = self.transformer.fit_transform(train_data)

        self.parameters = opt.curve_fit(self.fcn_to_fit, train_data.squeeze(), train_label)[0]
        logging.info(f"The fitted parameters are {self.parameters}")

    def predict(self, test_data):
        if self.transformer:
            test_data = self.transformer.transform(test_data)
        return self.fcn_to_fit(test_data.squeeze(), *self.parameters)


class PiecewiseLinFitModel(ModelBase):
    """
    分段线性回归模型
    """

    def __init__(self, n_break_point, normalize_transformer=None):
        """
        :n_break_point: 断点个数
        :normalize_transformer: 归一化函数
        """
        super().__init__()
        self.n_break_point = n_break_point
        self.pwlf = None
        self.transformer = normalize_transformer

    def fit(self, train_data, train_label):
        if self.transformer:
            train_data = self.transformer.fit_transform(train_data)

        self.pwlf = pwlf.PiecewiseLinFit(train_data.squeeze(), train_label)
        self.pwlf.fit(self.n_break_point)

    def predict(self, test_data):
        if self.transformer:
            test_data = self.transformer.transform(test_data)

        return self.pwlf.predict(test_data.squeeze())


class LogRegressor(ModelBase):
    """
    预测目标取对数
    """

    def __init__(self, estimator=XGBRegressor()):
        super().__init__()
        self.estimator = estimator

    def fit(self, train_data, train_label):
        self.estimator.fit(train_data, np.log2(train_label))
        if hasattr(self.estimator, "feature_importances_"):
            self.feature_importances_ = self.estimator.feature_importances_

    def predict(self, test_data):
        predict_data = self.estimator.predict(test_data)
        return np.exp2(predict_data)


class PerformanceRegressor(ModelBase):
    """
    预测目标的Performance
    """

    def __init__(self, estimator=XGBRegressor(), flops_dim=0, log_label=False):
        super().__init__()
        self.estimator = estimator
        self.flops_dim = flops_dim
        self.log_label = log_label

    def fit(self, train_data, train_label):
        label = train_data[:, self.flops_dim].squeeze() / (train_label + np.finfo(float).eps)
        label = np.array(label).astype(np.float)
        if self.log_label:
            label = np.log2(label)
        self.estimator.fit(train_data, label)

        if hasattr(self.estimator, "feature_importances_"):
            self.feature_importances_ = self.estimator.feature_importances_

    def predict(self, test_data):
        predict_performance = self.estimator.predict(test_data)
        if self.log_label:
            predict_performance = np.exp2(predict_performance)

        predict_time = test_data[:, self.flops_dim].squeeze() / (predict_performance + np.finfo(float).eps)

        return predict_time


class PerformanceDiff(ModelBase):
    """
    以性能差距作为预测目标
    """

    def __init__(self, estimator=XGBRegressor(), all_performance_dim=0, log_label=False):
        super().__init__()
        self.estimator = estimator
        self.all_performance_dim = all_performance_dim
        self.log_label = log_label

    def fit(self, train_data, train_label):
        label = train_data[:, self.all_performance_dim].squeeze() - train_label
        label = np.array(label).astype(np.float)
        if self.log_label:
            label = np.log2(label)
        self.estimator.fit(train_data, label)

        if hasattr(self.estimator, "feature_importances_"):
            self.feature_importances_ = self.estimator.feature_importances_

    def predict(self, test_data):
        predict_performance = self.estimator.predict(test_data)
        if self.log_label:
            predict_performance = np.exp2(predict_performance)

        predict_time = test_data[:, self.all_performance_dim].squeeze() - predict_performance

        return predict_time


class LinearRegressor(ModelBase):
    def __init__(self, estimator=LinearRegression()):
        super().__init__()
        self.estimator = estimator

    def fit(self, train_data, train_label):
        if self.estimator:
            self.estimator.fit(train_data, train_label)

    def predict(self, test_data):
        predict_performance = self.estimator.predict(test_data)

        return predict_performance
