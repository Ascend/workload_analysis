#!/usr/bin/python
# -*- coding: UTF-8 -*-
import copy
import json

import pandas as pd

from config import env
from framework.dataset import CSVDataset
from framework.model_base import LogRegressor
from framework.model_base import PiecewiseLinFitModel
from template.builder.general_builder import CollectDataDesc, PackDesc
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
    def init_modeling(cls):
        cls.clear()
        # 此处可以设置多个模型进行对比训练
        models = [
            # 分段线性模型
            ModelDesc("LogPiecewiseLinear", LogRegressor(estimator=PiecewiseLinFitModel(n_break_point=3)),
                      cls.op_feature)
        ]

        for soc_version in cls.soc_versions:
            # 训练和推理过程，芯片型号由开发者指定
            train_data_file = cls.get_data_path(cls.op_type, cls.get_data_file(cls.op_type, soc_version))
            test_data_file = cls.get_data_path(cls.op_type, cls.get_test_data_file(cls.op_type, soc_version))
            pack_path = cls.get_pack_model_path(cls.get_pack_file(cls.op_type, soc_version))
            pack_desc = PackDesc(cls.model_pack, pack_path)
            for dtype in cls.dtypes:
                filter_ = cls.get_filter(dtype)
                # 此处是为了保证生成的模型名不重复
                suffix = f"{dtype}_{soc_version}.pkl"
                # 更新模型的保存路径
                models_ = copy.deepcopy(models)
                for model in models_:
                    model.update_save_path(cls.get_handler_path(cls.op_type, f"{model.model_name}_{suffix}"))

                train_desc = TrainTestDesc(CSVDataset(train_data_file, soc_version=soc_version),
                                           filter_, models_, name=dtype)
                cls.train_infos.append(train_desc)
                test_desc = TrainTestDesc(CSVDataset(test_data_file, soc_version=soc_version),
                                          filter_, models_, name=dtype)
                cls.test_infos.append(test_desc)

                # 打包过程设置
                # 最终打包时每一个训练流程只有一个模型用于打包，故此处特殊处理
                handler_path = models_[0].save_path
                # 此类模型使用dtype作为打包模型的key
                key = dtype
                pack_desc.append(key, handler_path)
            cls.pack_infos.append(pack_desc)

    @classmethod
    def init_data_collect(cls):
        cls.clear()

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


class BinaryInOpBuilder(SingleInOpBuilder):
    io_generator = RandomShapeBinaryInGenerator
