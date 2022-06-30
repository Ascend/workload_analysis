import copy

from xgboost import XGBRegressor

from config import env
from framework.dataset import CSVDataset
from framework.model_base import PerformanceRegressor
from template.builder.general_builder import CollectDataDesc
from template.builder.general_builder import ModelDesc
from template.builder.general_builder import GeneralBuilder
from template.builder.general_builder import PackDesc
from template.builder.general_builder import TrainTestDesc


class XGBTrainingOpBuilder(GeneralBuilder):
    dtypes = []
    formats = []
    io_generator = None
    op_feature = None
    train_sample = 2000
    test_sample = 400
    xgb_estimator = XGBRegressor()
    soc_versions = ['Ascend310']

    @classmethod
    def init(cls):
        cls.clear()
        models = [
            ModelDesc("XGBRegressor",
                      PerformanceRegressor(estimator=cls.xgb_estimator, flops_dim=0, log_label=True), cls.op_feature),
        ]

        train_io_generator = cls.io_generator(cls.dtypes, cls.formats, n_sample=cls.train_sample, seed=0)
        train_data_file = cls.get_data_path(cls.op_type, cls.get_data_file(cls.op_type, env.soc_version))
        cls.train_data_collects = [CollectDataDesc(train_io_generator, train_data_file)]
        test_io_generator = cls.io_generator(cls.dtypes, cls.formats, n_sample=cls.test_sample, seed=1)
        test_data_file = cls.get_data_path(cls.op_type, cls.get_test_data_file(cls.op_type, env.soc_version))
        cls.test_data_collects = [CollectDataDesc(test_io_generator, test_data_file)]

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
