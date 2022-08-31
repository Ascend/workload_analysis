#!/usr/bin/python
# -*- coding: UTF-8 -*-
import copy
from xgboost import XGBRegressor
from framework.dataset import CSVDataset
from framework.op_register import RegisterOf
from framework.model_base import PerformanceRegressor
from template.builder.general_builder import PackDesc
from template.builder.general_builder import ModelDesc
from template.builder.general_builder import TrainTestDesc
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from ops.binary_cross_entropy.model import BinaryCrossEntropyDetailFeature
from ops.binary_cross_entropy.binary_cross_entropy_op import BinaryCrossEntropyOp
from ops.binary_cross_entropy.binary_cross_entropy_op import BinaryCrossEntropyIOGenerator

@RegisterOf("BinaryCrossEntropy")
class BinaryCrossEntropyBuilder(XGBTrainingOpBuilder):
    dtypes = ['float16', 'float']
    formats = ['ND']
    io_generator = BinaryCrossEntropyIOGenerator
    op = BinaryCrossEntropyOp
    train_sample = 1000
    test_sample = 200

    op_feature = BinaryCrossEntropyDetailFeature
    xgb_estimator = XGBRegressor(
        learning_rate=0.2,
        n_estimators=50,
        max_depth=6,
        min_child_weight=2,
        gamma=0,
        subsample=1,
        colsample_bytree=0.9,
    )
    @classmethod
    def init_modeling(cls):
        cls.clear()
        models = [
            ModelDesc("XGBRegressor",
                      PerformanceRegressor(estimator=cls.xgb_estimator, flops_dim=0, log_label=True), cls.op_feature),
        ]
        cls.init_modeling_by_models(models)

    @classmethod
    def init_modeling_by_models(cls, models):
        for soc_version in cls.soc_versions:
            train_data_file = cls.get_data_path(cls.op_type, cls.get_data_file(cls.op_type, soc_version))
            test_data_file = cls.get_data_path(cls.op_type, cls.get_test_data_file(cls.op_type, soc_version))
            suffix = f"{soc_version}.pkl"
            # 更新模型的保存路径
            models_ = copy.deepcopy(models)
            for model in models_:
                model.update_save_path(cls.get_handler_path(cls.op_type, f"{model.model_name}_{suffix}"))

            train_desc = TrainTestDesc(CSVDataset(train_data_file, soc_version=soc_version), None, models_)
            cls.train_infos.append(train_desc)
            test_desc = TrainTestDesc(CSVDataset(test_data_file, soc_version=soc_version), None, models_)
            cls.test_infos.append(test_desc)

            # 打包流程设置
            pack_path = cls.get_pack_model_path(cls.get_pack_file(cls.op_type, soc_version))
            pack_desc = PackDesc(cls.model_pack, pack_path)
            handler_path = models_[0].save_path
            pack_desc.append("common", handler_path)
            cls.pack_infos.append(pack_desc)