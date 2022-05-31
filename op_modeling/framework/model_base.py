# ！/usr/bin/python
import json
from abc import abstractmethod, ABCMeta

from scipy import optimize as opt
import pwlf
import pandas as pd

import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression


class StandardizeDescBase:
    """
    将算子的描述标准化，子类可复写standardize函数
    """

    @classmethod
    def update_desc(cls, desc, format_, shape, dtype):
        desc["shape"] = shape
        desc["format"] = format_
        desc["dtype"] = dtype

    @classmethod
    def update_origin_desc(cls, desc, format_, shape, dtype):
        desc["origin_shape"] = shape
        desc["origin_format"] = format_
        desc["origin_dtype"] = dtype

    def standardize(self, inputs, outputs, attrs):
        for input_ in inputs:
            self.update_origin_desc(input_, input_["format"], input_["shape"], input_["dtype"])
        for output_ in outputs:
            self.update_origin_desc(output_, output_["format"], output_["shape"], output_["dtype"])
        return inputs, outputs, attrs


class FeatureGeneratorBase:
    def __init__(self):
        self.attr = {}

    @abstractmethod
    def cal_feature(self, inputs, outputs, attrs):
        """
        计算特征
        :return: {f1:xx, f2:xx}
        """
        raise Exception("This function is not implement")


class DatasetBase(metaclass=ABCMeta):
    def __init__(self, fea_ge: FeatureGeneratorBase, stand_desc: StandardizeDescBase = StandardizeDescBase()):
        self.fea_ge: FeatureGeneratorBase = fea_ge
        self.stand_desc = stand_desc

    def get_dataset(self, use_block_dim=True):
        return self.get_features(use_block_dim), self.get_labels()

    @abstractmethod
    def get_features(self, use_block_dim=True):
        raise Exception("This function is not implement")

    @abstractmethod
    def get_labels(self):
        raise Exception("This function is not implement")


class CSVDataset(DatasetBase):
    def __init__(self, file, fea_ge: FeatureGeneratorBase, label="aicore_time(us)",
                 stand_desc: StandardizeDescBase = StandardizeDescBase):
        self.data = pd.read_csv(file, keep_default_na=False, na_values=['NA'])

        if type(label) is list:
            self.label = label
        else:
            self.label = [label]
        super().__init__(fea_ge, stand_desc)

    def get_original_data(self):
        return self.data

    def get_features(self, use_block_dim=True) -> pd.DataFrame:
        train_data = []
        for index, row in self.data.iterrows():
            inputs, outputs, attributes = self._parse_row(row)
            # 标准化输入输出及attr
            inputs, outputs, attributes = self.stand_desc.standardize(inputs, outputs, attributes)
            feature = self.fea_ge.cal_feature(inputs, outputs, attributes)
            train_data.append(feature)

        ret = pd.DataFrame(train_data)
        if use_block_dim:
            # 添加Block Dim信息到特征中
            ret['Block Dim'] = self.data['Block Dim']
        return ret

    def get_labels(self) -> pd.DataFrame:
        return self.data[self.label]

    def _parse_row(self, row):
        inputs = self._parse_inout(row['Original Inputs'])
        outputs = self._parse_inout(row['Original Outputs'])

        attr = row['Attributes']
        if attr == 'None' or not attr:
            attributes = None
        else:
            attributes = json.loads(attr)

        return inputs, outputs, attributes

    @classmethod
    def _parse_inout(cls, desc):
        if desc == 'None':
            return None

        objs = json.loads(desc)
        ret = []

        for obj in objs:
            ret.append(obj)

        return ret


class FusedCSVDataset(DatasetBase):
    def __init__(self, fused_file, origin_ops_file, fea_ge: FeatureGeneratorBase,
                 label="aicore_time(us)"):
        self.fused_data = pd.read_csv(fused_file, keep_default_na=False, na_values=['NA'])
        self.origin_ops_data = pd.read_csv(origin_ops_file, keep_default_na=False, na_values=['NA'])

        self.label = label
        super().__init__(fea_ge)

    def get_fused_data(self):
        return self.fused_data

    def get_features(self, use_block_dim=True):
        features = []
        for index, fused_row in self.fused_data.iterrows():
            graph_name = fused_row['Model Name']
            origin_op_rows = self.origin_ops_data[self.origin_ops_data['graph name'] == graph_name]
            inputs, outputs, attributes = self._parse_row(origin_op_rows)
            feature = self.fea_ge.cal_feature(inputs, outputs, attributes)
            features.append(feature)
        return pd.DataFrame(features)

    def get_labels(self):
        return self.fused_data[self.label]

    def _parse_row(self, rows: pd.DataFrame):
        inputs = []
        outputs = []
        origin_attributes = []
        for index, row in rows.iterrows():
            input_ = self._parse_inout(row['Original Inputs'])
            output = self._parse_inout(row['Original Outputs'])
            attr = row['Attributes']
            if attr == 'None' or not attr:
                attr = {}
            else:
                attr = json.loads(attr)
            attribute = dict()
            attribute['time'] = row["aicore_time(us)"]
            attribute['op_type'] = row["OP Type"]
            attribute['inputs'] = input_
            attribute['outputs'] = output
            attribute['attrs'] = attr

            # todo inputs outputs之后再适配
            origin_attributes.append(attribute)
        attribute = {"origin_ops": origin_attributes}

        return inputs, outputs, attribute

    @classmethod
    def _parse_inout(cls, desc):
        if desc == 'None':
            return []

        objs = json.loads(desc)
        ret = []

        for obj in objs:
            ret.append(obj)

        return ret


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
        print(f"The fitted parameters are {self.parameters}")

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
        label = train_data[:, self.flops_dim].squeeze() / (train_label + 1e-6)
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

        predict_time = test_data[:, self.flops_dim].squeeze() / predict_performance

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
