import os
import copy
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from termcolor import cprint

from framework.model_process import ModelProcess
from framework.model_save import OperatorHandle
from framework.model_save import Serialization
from framework.op_model_builder import OpModelBuilder, BuilderInfo
from framework.model_base import CSVDataset


class TrainTestBuilder(OpModelBuilder, ABC):

    @classmethod
    @abstractmethod
    def init(cls):
        pass

    @classmethod
    def _data_collect_from_list(cls, data_info_list):
        for item in data_info_list:
            data_save_path = item.data_file
            print(data_save_path)
            io_generator = item.io_generator
            cls._data_collect(cls.op_type, io_generator, data_save_path)

    @classmethod
    def filter_feature(cls, feature, labels, origin_data, is_training=True):
        return feature, labels, origin_data

    @classmethod
    def _init(cls, data_info, model_infos, generator_fcn, is_training=True):
        if data_info:
            return

        for i_case in model_infos:
            dtypes = i_case["dtype"]
            sample_feature = f'{cls.key_of_register}_{"_".join(dtypes)}'

            if is_training:
                data_file = cls.get_data_path(cls.key_of_register, f'{sample_feature}_{cls.ai_metric.name}')
            else:
                data_file = cls.get_data_path(cls.key_of_register, f'{sample_feature}_{cls.ai_metric.name}_test')

            io_generator = generator_fcn(i_case)
            data_info.append(
                BuilderInfo(model_name=i_case['model_name'],
                            models=i_case['models'],
                            sample_feature=sample_feature,
                            data_file=data_file,
                            io_generator=io_generator)
            )

    @classmethod
    def _modeling(cls, train_data_info: List[BuilderInfo]):
        cls.init()

        for item in train_data_info:
            for model_name, model, FeatureGenerator in item.models:
                # 模型训练+测试准度分析
                cprint(f"Using regression model: {model_name} for {item.sample_feature}", on_color='on_red')
                sample_feature = item.sample_feature
                model_save_path = cls.get_model_path(cls.key_of_register, f"{sample_feature}_{model_name}")
                feature_generator = FeatureGenerator()
                op_handle = OperatorHandle(model, feature_generator)
                dataset = CSVDataset(item.data_file, feature_generator, stand_desc=item.stand_desc())
                original_data = dataset.get_original_data()
                feature, labels = dataset.get_dataset(item.use_block_dim)
                feature, labels, dataset.data = cls.filter_feature(feature, labels, dataset.data, is_training=True)

                model_process = ModelProcess(op_handle)
                model_process.cross_validation(feature.to_numpy(), labels.to_numpy(), feature.columns)
                Serialization.serialize(op_handle, model_save_path)

                data = copy.deepcopy(feature)
                predict_time = op_handle.model.predict(feature.to_numpy())
                data["aicore_time(us)"] = labels
                data["predict_time(us)"] = predict_time
                data = pd.concat((data, original_data.iloc[:, 11:]), axis=1)
                data.to_csv(f"{os.path.dirname(model_save_path)}/{sample_feature}_data.csv", index=False)

    @classmethod
    def generate_pack_key(cls, item):
        """
        获取打包的key，不同的算子和建模策略可以使用不同的key
        :param item:
        :return:
        """
        return item.dtypes[0]

    @classmethod
    def pack_model(cls, train_data_info: List[BuilderInfo], pack_index, model_type, model_path):
        """
        将handle打包进一个model
        :param train_data_info: 训练数据
        :param pack_index: 最终打包的模型在train_data_info各case的models中的索引
        :param model_type: 打包的最终模型类型
        :param model_path: 模型的保存路径
        :return:
        """
        op_model = model_type()
        for i, item in enumerate(train_data_info):
            model_name, _, _ = item.models[pack_index[i]]
            sample_feature = item.sample_feature
            model_save_path = cls.get_model_path(cls.key_of_register, f"{sample_feature}_{model_name}")
            handle = Serialization.deserialization(model_save_path)
            key = cls.generate_pack_key(item)
            op_model.add_handle(key, handle)

        Serialization.serialize(op_model, model_path)

    @classmethod
    def _test(cls, infer_data_info: List[BuilderInfo]):
        cls.init()

        for item in infer_data_info:
            test_data_save_path = item.data_file

            cprint("Test strange samples", on_color='on_red')
            for model_name, _, _ in item.models:
                cprint(f"Using regression model: {model_name}", on_color='on_red')

                # 模拟预测一个算子性能
                sample_feature = item.sample_feature
                model_save_path = cls.get_model_path(cls.key_of_register, f"{sample_feature}_{model_name}")

                op_handle = Serialization.deserialization(model_save_path)
                feature_generator = op_handle.fea_ge
                data_base = CSVDataset(test_data_save_path, feature_generator, stand_desc=item.stand_desc())
                test_data = data_base.get_features(item.use_block_dim)
                test_label = data_base.get_labels()
                test_data, test_label, data_base.data = cls.filter_feature(test_data, test_label, data_base.data,
                                                                           is_training=False)
                predict_time = op_handle.model.predict(test_data.to_numpy())

                data = data_base.data
                data['predict_time'] = predict_time
                real_time = data['aicore_time(us)'].to_numpy().astype(float)
                data['relative_error'] = abs(real_time - predict_time) / (real_time + 1e-6)
                data = data.sort_values(by=['relative_error'])
                pd.set_option('display.max_columns', None, 'display.width', 5000, 'display.max_rows', None)
                cprint(data[['Input Shapes', 'Input Data Types', 'Input Formats', 'Attributes',
                             'aicore_time(us)', 'predict_time', 'relative_error']],
                       color='green')

                model_process = ModelProcess(op_handle)
                evaluate_ret = model_process.evaluate_stage(test_data.to_numpy(), test_label.to_numpy())
                cprint(pd.DataFrame([evaluate_ret]), color='green')

                data2 = copy.deepcopy(test_data)
                data2["Block Dim"] = data["Block Dim"]
                data2["aicore_time(us)"] = test_label
                data2["predict_time(us)"] = predict_time
                # data2 = pd.concat((data2, data.iloc[:, 15:]), axis=1)
                data2 = pd.concat((data2, data), axis=1)

                data2.to_csv(f"{os.path.dirname(model_save_path)}/{sample_feature}_data.csv", index=False)
