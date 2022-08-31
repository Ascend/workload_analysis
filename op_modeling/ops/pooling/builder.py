import copy
import json
import pandas as pd
from xgboost import XGBRegressor
from ops.pooling.pooling_op import PoolingOp
from ops.pooling.pooling_op import PoolingIOGenerator
from ops.pooling.model import PoolingDetailFeature
from ops.pooling.model import PoolingOpModel

from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from framework.op_register import RegisterOf

from framework.dataset import CSVDataset
from template.builder.general_builder import PackDesc
from template.builder.general_builder import TrainTestDesc


@RegisterOf("Pooling")
class PoolingBuilder(XGBTrainingOpBuilder):
    def __init__(self):
        super().__init__()
    dtypes = ['float16']
    formats = ['NCHW']
    io_generator = PoolingIOGenerator
    model_pack = PoolingOpModel
    op = PoolingOp
    train_sample = 16000
    test_sample = 8000

    op_feature = PoolingDetailFeature
    xgb_estimator = XGBRegressor(
        learning_rate=0.06,
        n_estimators=350,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=1,
    )
    filters_conditions ={'Avg_Global': [1, True], 'Avg_unGlobal': [1, False], 'Max_Global': [0, True], \
                         'Max_unGlobal': [0, False]}

    @classmethod
    def get_filter(cls, filters_condition):

        filter_ = lambda data: pd.Series(
            True
            if json.loads(row['Attributes'])['mode'] == filters_condition[0] and \
               json.loads(row['Attributes'])['global_pooling'] == filters_condition[1]
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
            # 打包流程设置
            pack_path = cls.get_pack_model_path(cls.get_pack_file(cls.op_type, soc_version))
            pack_desc = PackDesc(cls.model_pack, pack_path)

            for key, value in cls.filters_conditions.items():
                filter_ = cls.get_filter(value)
                suffix = f"{key}_{soc_version}.pkl"
                # 更新模型的保存路径
                models_ = copy.deepcopy(models)
                for model in models_:
                    model.update_save_path(cls.get_handler_path(cls.op_type, f"{model.model_name}_{suffix}"))
                train_desc = TrainTestDesc(CSVDataset(train_data_file, soc_version=soc_version), filter_, models_)
                cls.train_infos.append(train_desc)
                test_desc = TrainTestDesc(CSVDataset(test_data_file, soc_version=soc_version), filter_, models_)
                cls.test_infos.append(test_desc)

                handler_path = models_[0].save_path
                pack_desc.append(key, handler_path)
            cls.pack_infos.append(pack_desc)


