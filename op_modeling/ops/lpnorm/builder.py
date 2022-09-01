import copy
import json
import pandas as pd
from ops.lpnorm.lpnorm_op import LpNormOp
from ops.lpnorm.lpnorm_op import LpNormIOGenerator
from ops.lpnorm.model import LpNormFeature
from ops.lpnorm.model import LpNormOpModel
from framework.dataset import CSVDataset
from framework.op_register import RegisterOf
from template.builder.general_builder import TrainTestDesc
from template.builder.general_builder import PackDesc
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from xgboost import XGBRegressor


@RegisterOf("LpNorm")
class LpNormBuilder(XGBTrainingOpBuilder):
    dtypes = ['float16', 'float']
    formats = ['ND']
    train_sample = 10000
    test_sample = 2000
    p = [0, 1, 2]
    io_generator = LpNormIOGenerator
    model_pack = LpNormOpModel
    op = LpNormOp
    op_feature = LpNormFeature
    xgb_estimator = XGBRegressor(
        learning_rate=0.16,
        n_estimators=40,
        max_depth=20,
        subsample=0.7,
        colsample_bytree=0.9,
    )

    @classmethod
    def get_filter(cls, p):
        """
        用于筛选符合指定p值的算子
        :param p:
        :return:
        """
        filter_ = lambda data: pd.Series(
            True
            if json.loads(row['Attributes'])['p'] == p
            else
            False
            for _, row in data.iterrows()
        )
        return filter_

    @classmethod
    def init_modeling_by_models(cls, models):
        for soc_version in cls.soc_versions:
            train_data_file = cls.get_data_path(cls.op_type, cls.get_data_file(cls.op_type, soc_version))
            test_data_file = cls.get_data_path(cls.op_type, cls.get_test_data_file(cls.op_type, soc_version))
            for i in cls.p:
                suffix = f"{i}_{soc_version}.pkl"
                # filter
                filter_ = cls.get_filter(i)
                # 更新模型的保存路径
                models_ = copy.deepcopy(models)
                for model in models_:
                    model.update_save_path(cls.get_handler_path(cls.op_type, f"{model.model_name}_{suffix}"))
                train_desc = TrainTestDesc(CSVDataset(train_data_file, soc_version=soc_version), filter_, models_)
                cls.train_infos.append(train_desc)
                test_desc = TrainTestDesc(CSVDataset(test_data_file, soc_version=soc_version), filter_, models_)
                cls.test_infos.append(test_desc)

                # 打包流程设置
                pack_path = cls.get_pack_model_path(cls.get_pack_file(cls.op_type, soc_version))
                pack_desc = PackDesc(cls.model_pack, pack_path)
                handler_path = models_[0].save_path
                # 此类模型使用p作为打包模型的key
                pack_desc.append(i, handler_path)
            cls.pack_infos.append(pack_desc)
