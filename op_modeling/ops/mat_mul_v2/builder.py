#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
from config import config
from ops.mat_mul_v2.feature import MatMulV2DetailFeature
from ops.mat_mul_v2.mat_mul_v2_op import (
    MatMulV2IOGenerator,
    MatMulV2Standardize,
    MatMulV2linearFeature,
)
from ops.mat_mul_v2.model import MatMulV2OpModel
from framework.curves import linear_curve
from framework.model_base import CurveFitBasedModel, PiecewiseLinFitModel
from framework.op_register import RegisterOfBuilder
from template.builder.general.multi_process_builder import MultiProcessBuilder
from framework.op_model_builder import BuilderInfo
from . import constant
from framework.model_base import LogRegressor, PerformanceRegressor


@RegisterOfBuilder("MatMulV2")
class MatMulV2Builder(MultiProcessBuilder):
    _train_data_info = list()
    _infer_data_info = list()
    model_pack = "MatMulV2_" + config.soc_version + ".pkl"
    op_model = MatMulV2OpModel

    @classmethod
    def init(cls):
        """
        初始化定义数据生成策略以及依赖该策略生成的数据存储位置
        :return:
        """
        if cls._train_data_info:
            return
        # 针对训练
        models = [
            # ("LinearByCurveFit", CurveFitBasedModel(linear_curve), MatMulV2linearFeature),  # 线性模型（基于曲线拟合）
            # ("PiecewiseLinear", PiecewiseLinFitModel(n_break_point=3), MatMulV2linearFeature),  # 分段线性模型
            # ("XGBRegressor", LogRegressor(), MatMulV2DetailFeature),  # 机器学习模型
            (
                "LogXGBRegressor2",
                PerformanceRegressor(flops_dim=0, log_label=True),
                MatMulV2DetailFeature,
            )  # 0.2
        ]
        cases = [
            dict(
                model_name="simple",
                dtype=list(constant.data_type.keys()),
                models=models,
            )
        ]

        for i_case in cases:
            dtypes = i_case["dtype"]
            sample_feature = f'MatMulV2_{"_".join(dtypes)}'
            io_generator = MatMulV2IOGenerator(dtypes)
            stand_desc = MatMulV2Standardize
            cls._train_data_info.append(
                BuilderInfo(
                    model_name=i_case["model_name"],
                    models=i_case["models"],
                    sample_feature=sample_feature,
                    data_file=cls.get_data_path(
                        cls.key_of_register,
                        f"{sample_feature}_{cls.ai_metric.name}",
                    ),
                    io_generator=io_generator,
                    dtypes=i_case["dtype"],
                    use_block_dim=False,
                    stand_desc=stand_desc,
                )
            )

        # 针对验证
        if cls._infer_data_info:
            return
        for i_case in cases:
            dtypes = i_case["dtype"]
            sample_feature = f'MatMulV2_{"_".join(dtypes)}'
            io_generator = MatMulV2IOGenerator(dtypes, is_training=False)
            stand_desc = MatMulV2Standardize
            cls._infer_data_info.append(
                BuilderInfo(
                    model_name=i_case["model_name"],
                    models=i_case["models"],
                    sample_feature=sample_feature,
                    data_file=cls.get_data_path(
                        cls.key_of_register,
                        f"{sample_feature}_{cls.ai_metric.name}_test",
                    ),
                    io_generator=io_generator,
                    dtypes=i_case["dtype"],
                    use_block_dim=False,
                    stand_desc=stand_desc,
                )
            )

    @classmethod
    def modeling(cls):
        cls.init()
        cls._modeling(cls._train_data_info)
        if cls.op_model is not None:
            cls.pack_model(
                cls._train_data_info,
                [0] * len(cls._train_data_info),
                cls.op_model,
                os.path.join(cls.get_data_dir(cls.key_of_register), cls.model_pack),
            )

    @classmethod
    def test(cls):
        cls.init()
        cls._test(cls._infer_data_info)

    @classmethod
    def generate_pack_key(cls, item):
        return "common"
