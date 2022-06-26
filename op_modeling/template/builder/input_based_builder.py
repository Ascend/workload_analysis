#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json

import pandas as pd

from config import env
from framework.model_base import LogRegressor
from framework.model_base import PiecewiseLinFitModel
from template.builder.general_builder import CollectDataDesc
from template.builder.general_builder import GeneralBuilder
from template.builder.general_builder import ModelDesc
from template.builder.general_builder import TrainTestDesc
from template.feature.elementwise_feature import ElementwiseFlopsFeature
from template.generator.random_shape_generator import RandomShapeBinaryInGenerator
from template.generator.random_shape_generator import RandomShapeSingleInOutGenerator


class SingleInOpBuilder(GeneralBuilder):
    """
    单输入算子的建模模板，该模板
    1. 使用ElementwiseFlopsFeature作为算子特征
    2. 使用分段线性模型进行回归训练
    """
    # 数据采集、训练和测试的数据类别
    dtypes = []
    # 训练和测试使用的数据芯片版本
    soc_versions = ["Ascend310"]
    io_generator = RandomShapeSingleInOutGenerator
    op_feature = ElementwiseFlopsFeature
    train_sample = 1000
    test_sample = 200

    @classmethod
    def get_filter(cls, dtype):
        """
        用于筛选符合指定dtype的算子
        :param dtype:
        :return:
        """
        filter_ = lambda data: pd.Series(
            True
            if json.loads(row['Original Inputs'])[0]['dtype'] == dtype
            else
            False
            for _, row in data.iterrows()
        )
        return filter_

    @classmethod
    def init(cls):
        cls.clear()
        models = [
            # 分段线性模型
            ModelDesc("LogPiecewiseLinear", LogRegressor(estimator=PiecewiseLinFitModel(n_break_point=3)),
                      cls.op_feature)
        ]

        # 数据采集过程设置
        # 训练数据
        train_io_generator = cls.io_generator(dtypes=cls.dtypes, n_sample=cls.train_sample, seed=0)
        # 数据采集过程，芯片型号由环境中获取
        train_data_file = cls.get_data_path(cls.op_type, cls.get_data_file(cls.op_type, env.soc_version))
        cls.train_data_collects = [CollectDataDesc(train_io_generator, train_data_file)]
        # 测试数据
        test_io_generator = cls.io_generator(dtypes=cls.dtypes, n_sample=cls.test_sample, seed=1)
        test_data_file = cls.get_data_path(cls.op_type, cls.get_test_data_file(cls.op_type, env.soc_version))
        cls.test_data_collects = [CollectDataDesc(test_io_generator, test_data_file)]

        # 训练、测试过程设置
        for soc_version in cls.soc_versions:
            # 训练和推理过程，芯片型号可由开发者指定
            train_data_file = cls.get_data_path(cls.op_type, cls.get_data_file(cls.op_type, soc_version))
            test_data_file = cls.get_data_path(cls.op_type, cls.get_test_data_file(cls.op_type, soc_version))
            for dtype in cls.dtypes:
                filter_ = cls.get_filter(dtype)
                train_desc = TrainTestDesc(train_data_file, filter_, models, dtype)
                cls.train_infos.append(train_desc)
                test_desc = TrainTestDesc(test_data_file, filter_, models, dtype)
                cls.test_infos.append(test_desc)


class BinaryInOpBuilder(SingleInOpBuilder):
    io_generator = RandomShapeBinaryInGenerator
