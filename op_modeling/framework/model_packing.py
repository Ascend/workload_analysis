import os
import pickle
from abc import abstractmethod

import pandas as pd

from framework.feature_base import FeatureGeneratorBase
from framework.model_base import ModelBase


def serialize(obj: any, file_path: str) -> None:
    with os.fdopen(os.open(file_path, os.O_WRONLY | os.O_CREAT, 0o440), 'wb') as f:
        pickle.dump(obj, f)


def deserialization(file_path: str) -> any:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class OperatorHandle:
    """
    该类主要用于绑定model和fea_ge, 最终作为打包实体的基本组成单元
    """

    def __init__(self: any, model: ModelBase = None, fea_ge: FeatureGeneratorBase = None) -> None:
        """
        :model: 回归模型
        :fea_ge: 用于根据输入生成特征
        """

        self.model: ModelBase = model
        self.fea_ge: FeatureGeneratorBase = fea_ge


class ModelPackBase:
    """
    算子模型最终打包的实体
    1. 该类完成单算子多模型的组合，多模型建模的依据可能来源于不同的input,output,attr的组合
    2. 该类也会作为产品化侧的直接交付单位
    """
    SUCCESS = 0
    UN_SUCCESS = -1
    ILLEGAL_PERFORMANCE = -1

    def __init__(self: any) -> None:
        self.cur_handle = None
        self.handle_tables = dict()

    @abstractmethod
    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        """
        为支持算子由多模型组合，根据输入的组合获取模型的key
        """
        raise Exception("To be implement")

    @abstractmethod
    def param_check(self: any, inputs: list, outputs: list, attr: dict) -> tuple:
        """
        算子参数校验
        :return: ret_code, msg
        """
        raise Exception("To be implement")

    def add_handle(self: any, key: str, handle: OperatorHandle) -> None:
        """
        添加新的子模型
        :return: ret_code, msg
        """
        if self.handle_tables.get(key, None) is not None:
            raise Exception("Exist handle for key {}".format(key))
        self.handle_tables[key] = handle

    def predict(self: any, inputs: list, outputs: list, attr: dict) -> tuple:
        """
        预测一个算子的性能，该函数用于产品推理侧
        :param inputs: 输入
        :param outputs: 输出
        :param attr: 属性
        :return: ret_code, msg, predict_result
        """
        ret_code, msg = self.param_check(inputs, outputs, attr)
        if ret_code != self.SUCCESS:
            return ret_code, msg, self.ILLEGAL_PERFORMANCE
        key = self.generate_key(inputs, outputs, attr)
        self.cur_handle = self.handle_tables.get(key)
        if not self.cur_handle:
            msg = "No handle found for input: {}, output: {}, attr: {}".format(inputs, outputs, attr)
            return self.UN_SUCCESS, msg, self.ILLEGAL_PERFORMANCE
        features = self.cur_handle.fea_ge.cal_feature(inputs, outputs, attr)
        feature_df = pd.DataFrame([features])
        res = self.cur_handle.model.predict(feature_df.to_numpy())
        return self.SUCCESS, "", res.tolist()[0]
