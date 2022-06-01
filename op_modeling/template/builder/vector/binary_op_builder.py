#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
from typing import List

from framework.model_base import LogRegressor
from framework.model_base import PiecewiseLinFitModel, PerformanceRegressor
from framework.op_model_builder import BuilderInfo
from template.builder.general.multi_process_builder import MultiProcessBuilder
from template.generator.vector.binary_op_generator import BinaryVectorIOGenerator
from template.feature.single_op_feature import SingleVectorFeatureBase


class BinaryOpBuilder(MultiProcessBuilder):
    model_names = []
    dtypes = []
    io_generator = BinaryVectorIOGenerator
    _train_data_info: List[BuilderInfo] = list()
    _infer_data_info: List[BuilderInfo] = list()
    train_sample = 1000
    test_sample = 100
    op_model = ""
    model_pack = ""

    @classmethod
    def init(cls):
        """
        初始化定义数据生成策略以及依赖该策略生成的数据存储位置
        :return:
        """
        if cls._train_data_info:
            return
        models = [
            # 分段线性模型
            ("LogPiecewiseLinear", LogRegressor(estimator=PiecewiseLinFitModel(n_break_point=3)),
             SingleVectorFeatureBase),
        ]

        cases = []
        for model_name, dtype in zip(cls.model_names, cls.dtypes):
            cases.append(dict(model_name=model_name, dtype=[dtype], models=models))

        for i_case in cases:
            dtypes = i_case["dtype"]
            sample_feature = f'{cls.key_of_register}_{"_".join(dtypes)}'
            io_generator = cls.io_generator(dtypes, n_sample=cls.train_sample)
            cls._train_data_info.append(
                BuilderInfo(model_name=i_case['model_name'],
                            models=i_case['models'],
                            dtypes=i_case['dtype'],
                            sample_feature=sample_feature,
                            data_file=cls.get_data_path(cls.key_of_register,
                                                        f'{sample_feature}_{cls.ai_metric.name}'),
                            io_generator=io_generator,
                            use_block_dim=False)
            )

        # 针对验证
        if cls._infer_data_info:
            return
        for i_case in cases:
            dtypes = i_case["dtype"]
            sample_feature = f'{cls.key_of_register}_{"_".join(dtypes)}'

            io_generator = cls.io_generator(dtypes, is_training=False, n_sample=cls.test_sample)

            cls._infer_data_info.append(
                BuilderInfo(model_name=i_case['model_name'],
                            models=i_case['models'],
                            dtypes=i_case['dtype'],
                            sample_feature=sample_feature,
                            data_file=cls.get_data_path(cls.key_of_register,
                                                        f'{sample_feature}_{cls.ai_metric.name}_test'),
                            io_generator=io_generator,
                            use_block_dim=False)
            )

    @classmethod
    def modeling(cls):
        cls._modeling(cls._train_data_info)
        if cls.op_model is not None:
            cls.pack_model(cls._train_data_info, [0] * len(cls._train_data_info), cls.op_model,
                           os.path.join(cls.get_data_dir(cls.key_of_register), cls.model_pack))

    @classmethod
    def test(cls):
        cls._test(cls._infer_data_info)
