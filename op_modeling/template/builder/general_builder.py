import logging
import os
import copy
import stat
import time
from abc import ABC, abstractmethod
from multiprocessing import Process
from typing import List
import pandas as pd
import numpy as np
from termcolor import cprint

from framework.dataset import CSVDataset
from framework.model_process import ModelProcess
from framework.model_packing import OperatorHandle
from framework.model_packing import serialize, deserialization
from framework.op_model_builder import OpModelBuilder


class CollectDataDesc:
    """
    定义一个数据采集过程的基本元素
    """

    def __init__(self, io_generator, data_file):
        """
        :param io_generator: 样本生成器
        :param data_file: profiling数据保存的路径
        """
        self.io_generator = io_generator
        self.data_file = data_file


class ModelDesc:
    """
    定义一个模型的基本元素
    """

    def __init__(self, model_name, regressor, fea_ge, save_path=""):
        """
        :param model_name: 模型名
        :param regressor: 训练使用的回归器
        :param fea_ge: 使用的特征生成类(FeatureGeneratorBase的子类)
        :param save_path: 模型保存的路径
        """
        self.model_name = model_name
        self.regressor = regressor
        self.fea_ge = fea_ge
        self.save_path = save_path

    def get(self):
        return self.model_name, self.regressor, self.fea_ge

    def update_save_path(self, path):
        self.save_path = path


class TrainTestDesc:
    """
    定义一个训练或测试过程的基本元素
    """

    def __init__(self, dataset: CSVDataset, filter_, model_descs: List[ModelDesc], name=""):
        """
        :param dataset: 训练使用的数据集
        :param filter_: 用于筛选数据集的筛选器，为一个lamda表达式，返回值为pd.Series
        :param model_descs: 训练使用的模型
        """
        self.dataset = dataset
        self.filter = filter_
        self.model_descs = model_descs
        self.name=name


class PackDesc:
    """
    定义一个打包过程的基本元素
    """

    def __init__(self, model_pack, file_path):
        """
        :param model_pack:  使用的打包类
        :param file_path: 打包后模型存储的路径
        """
        # handler_path的列表
        self.handler_paths = list()
        # 与handler_paths对应的key列表
        self.handler_keys = list()
        self.model_pack = model_pack
        self.file_path = file_path

    def append(self, handler_key, handler_path):
        self.handler_keys.append(handler_key)
        self.handler_paths.append(handler_path)


class GeneralBuilder(OpModelBuilder, ABC):
    """
    该模板定义了一套标准的数据采集、训练、测试，模型打包流程。
    对于大部分算子的建模动作只需要继承该模板，并重写init函数
    """

    train_data_collects: List[CollectDataDesc] = list()  # 定义训练数据采集
    test_data_collects: List[CollectDataDesc] = list()  # 定义测试数据采集
    train_infos: List[TrainTestDesc] = list()  # 定义训练过程
    test_infos: List[TrainTestDesc] = list()  # 定义测试过程
    pack_infos: List[PackDesc] = list()  # 定义打包过程
    model_pack = None

    @classmethod
    @abstractmethod
    def init_data_collect(cls):
        """
        init函数中应定义的内容包括
        1. 数据生成策略（IOStrategy）
        2. 采集数据存储位置
        """
        raise Exception("init_data_collect is not impl")

    @classmethod
    @abstractmethod
    def init_modeling(cls):
        """
        init函数中应定义的内容包括
        1. 训练、测试使用的回归器、特征、模型保存的路径
        2. 需要打包的模型各路径及其在最终模型中的key
        3. 最终保存的模型类别及其路径
        """
        raise Exception("init_modeling is not impl")

    @classmethod
    def clear(cls):
        cls.train_data_collects.clear()
        cls.test_data_collects.clear()
        cls.train_infos.clear()
        cls.test_infos.clear()

    @classmethod
    def data_collect(cls, mode='all'):
        """
        请在此处完成训练和测试用的数据集的处理
        :return:
        """
        cls.init_data_collect()
        if mode == 'all' or mode == 'train':
            cls.data_collect_by_multi_process(cls.train_data_collects)

        if mode == 'all' or mode == 'infer':
            cls.data_collect_by_multi_process(cls.test_data_collects)

    @classmethod
    def modeling(cls, mode="experiment"):
        cls.init_modeling()
        cls._modeling(cls.train_infos, mode)

    @classmethod
    def test(cls):
        cls.init_modeling()
        cls._test(cls.test_infos)

    @classmethod
    def pack(cls):
        cls.init_modeling()
        if cls.model_pack is None:
            return
        cls._pack(cls.pack_infos)

    @classmethod
    def chunks(cls, lst, n):
        """Yield successive n-sized chunks from lst."""
        if n == 0:
            raise Exception("n must > 0")
        n_size = int(len(lst) / n) + 1
        for start in range(0, len(lst), n_size):
            yield lst[start:start + n_size]

    @classmethod
    def data_collect_by_multi_process(cls, data_collects: List[CollectDataDesc]):
        """
        多设备、多进程采集数据
        """

        class SimpleGenerator:
            def __init__(self, cases=None):
                if cases is None:
                    self.cases = []
                else:
                    self.cases = cases

            def __call__(self, *args, **kwargs):
                for item_ in self.cases:
                    yield item_

        device_num = len(cls.device_ids)

        def clear_tmp_file(paths):
            for path_ in paths:
                if os.path.exists(path_):
                    os.remove(path_)

        for item in data_collects:
            # 最终数据的文件路径
            data_save_path = item.data_file
            # 各device保存的临时文件路径
            tmp_device_data_paths = []
            for device_id in cls.device_ids:
                # 避免竞争，每个device单独使用一个落盘文件，最后一起合并
                tmp_device_data_path = data_save_path.replace(".csv", f"_device{device_id}.csv")
                tmp_device_data_paths.append(tmp_device_data_path)

            # 将所有结果合并，上一次可能异常结束，所以先合并以方便查看已经生成了多少样本
            cls._combine_all_dfs(tmp_device_data_paths, data_save_path)

            # 获取所有还未被处理的样本数据
            io_generator = item.io_generator
            data = cls._get_not_done_samples(data_save_path, io_generator)
            if not data:
                clear_tmp_file(tmp_device_data_paths)
                continue

            sub_generators = list(cls.chunks(data, device_num))

            process_list = []
            for i, device_id in enumerate(cls.device_ids):
                if i >= len(sub_generators):
                    continue

                sub_generator = SimpleGenerator(sub_generators[i])
                p = Process(target=cls._data_collect,
                            args=(cls.op, sub_generator, tmp_device_data_paths[i], device_id))
                p.start()
                logging.info(f"Launching process of device_id={device_id}, data_save_path={tmp_device_data_paths[i]}")
                time.sleep(5)
                process_list.append(p)

            for p in process_list:
                p.join()

            # 将所有结果合并
            cls._combine_all_dfs(tmp_device_data_paths, data_save_path)
            clear_tmp_file(tmp_device_data_paths)

    @classmethod
    def filter_data(cls, filter_, data: pd.DataFrame, feature: pd.DataFrame, label: pd.DataFrame):
        if filter_ is None:
            return data, feature, label
        # 从原始数据中进行筛选
        filter_series = filter_(data)
        return data[filter_series], feature[filter_series], label[filter_series]

    @classmethod
    def _modeling(cls, train_infos: List[TrainTestDesc], mode="experiment"):
        """
        依据train_info进行建模
        :param train_infos: 训练过程的描述
        :param mode: experiment/build. 分别对应交叉验证的关闭和开启
        :return:
        """
        for train_info in train_infos:
            dataset = train_info.dataset
            for model_desc in train_info.model_descs:
                model_name, model, feature_ge = model_desc.get()
                soc_version = dataset.soc_version
                cprint(f"Train {train_info.name}: using regression model {model_name} for {cls.op_type} in platform"
                       f" {soc_version}", on_color='on_red')

                # 获取训练数据
                feature_generator = feature_ge()
                dataset.set_feature_generator(feature_generator)
                original_data = dataset.get_original_data()
                feature, labels = dataset.get_dataset()
                # 部分算子需要依据不同场景进行建模，在此处筛选相关数据
                original_data, feature, labels = cls.filter_data(train_info.filter, original_data, feature, labels)
                # 训练
                op_handle = OperatorHandle(model, feature_generator)
                model_process = ModelProcess(op_handle)
                if mode == "experiment":
                    model_process.train(feature.to_numpy(), labels.to_numpy(), n_splits=5,
                                        feature_names=feature.columns)
                elif mode == "build":
                    model_process.train(feature.to_numpy(), labels.to_numpy(), n_splits=0,
                                        feature_names=None)
                else:
                    raise Exception("Unsupported modeling mode")
                # 模型保存
                model_save_path = model_desc.save_path
                serialize(op_handle, model_save_path)
                # 保存特征用于分析
                data = copy.deepcopy(feature)
                predict_time = op_handle.model.predict(feature.to_numpy())
                data["aicore_time(us)"] = labels
                data["predict_time(us)"] = predict_time
                data = pd.concat((data, dataset.data.iloc[:, 11:]), axis=1)
                model_prefix = os.path.splitext(model_save_path)[0]
                data.to_csv(f"{model_prefix}_data.csv", index=False)
                cprint(f"Train done: using regression model {model_name} for {cls.op_type} in platform"
                       f" {soc_version}", on_color='on_green')

    @classmethod
    def _test(cls, test_infos: List[TrainTestDesc]):
        for test_info in test_infos:
            dataset = test_info.dataset
            for model_desc in test_info.model_descs:
                model_name, _, _ = model_desc.get()
                soc_version = dataset.soc_version
                cprint(f"Test {test_info.name}: using regression model {model_name} for {cls.op_type} in platform"
                       f" {soc_version}", on_color='on_red')

                # 加载模型
                model_save_path = model_desc.save_path
                op_handle = deserialization(model_save_path)
                # 获取测试数据
                feature_generator = op_handle.fea_ge
                dataset.set_feature_generator(feature_generator)
                original_data = dataset.get_original_data()
                feature, labels = dataset.get_dataset()
                original_data, feature, labels = cls.filter_data(test_info.filter, original_data, feature, labels)
                predict_time = op_handle.model.predict(feature.to_numpy())

                screen = original_data
                screen['predict_time'] = predict_time
                real_time = screen['aicore_time(us)'].to_numpy().astype(float)
                screen['relative_error'] = abs(real_time - predict_time) / (real_time + np.finfo(float).eps)
                screen = screen.sort_values(by=['relative_error'])
                pd.set_option('display.max_columns', None, 'display.width', 5000, 'display.max_rows', None)
                cprint(screen[['Input Shapes', 'Input Data Types', 'Input Formats',
                               'aicore_time(us)', 'predict_time', 'relative_error']],
                       color='green')

                model_process = ModelProcess(op_handle)
                evaluate_ret = model_process.evaluate_stage(feature.to_numpy(), labels.to_numpy())
                cprint(pd.DataFrame([evaluate_ret]), color='green')

                data = copy.deepcopy(feature)
                data["aicore_time(us)"] = labels
                data["predict_time(us)"] = predict_time
                data = pd.concat((data, screen), axis=1)

                model_prefix = os.path.splitext(model_save_path)[0]
                data.to_csv(f"{model_prefix}_test_data.csv", index=False)

    @classmethod
    def _pack(cls, pack_infos: List[PackDesc]):
        """
        将该算子的不同handler打包进最终的模型
        :param pack_infos: 每一个PackDesc对应一个最终保存的模型
        :return:
        """

        for pack_desc in pack_infos:
            op_model = pack_desc.model_pack()
            for handler_path, key in zip(pack_desc.handler_paths, pack_desc.handler_keys):
                handle = deserialization(handler_path)
                op_model.add_handle(key, handle)
            serialize(op_model, pack_desc.file_path)

    @classmethod
    def _combine_all_dfs(cls, tmp_device_data_paths, data_save_path):
        """
        合并各个device产生的临时文件以及最终文件中的数据，保存至 data_save_path
        :param tmp_device_data_paths: 各device产生的临时文件列表
        :param data_save_path: 保存数据的最终文件
        :return:
        """
        if not tmp_device_data_paths:
            return

        if os.path.exists(data_save_path):
            dfs = [pd.read_csv(data_save_path)]
        else:
            dfs = []

        for path in tmp_device_data_paths:
            if os.path.exists(path) and os.path.getsize(path) != 0:
                dfs.append(pd.read_csv(path))
                os.remove(path)
            flags = os.O_RDWR | os.O_CREAT
            modes = stat.S_IWUSR | stat.S_IRUSR
            fd = os.fdopen(os.open(path, flags, modes), 'w')
            fd.close()

        if dfs:
            df = pd.concat(dfs)
            df.drop_duplicates(ignore_index=True, inplace=True)
            df.to_csv(data_save_path, index=False)

    @classmethod
    def _get_not_done_samples(cls, data_save_path, io_generator):
        def done_before(done_prof, desc):
            match_item = done_prof.loc[(done_prof['Original Inputs'] == desc['Original Inputs']) &
                                       (done_prof['Attributes'] == desc['Attributes'])]
            if match_item.empty:
                return False
            return True

        if os.path.exists(data_save_path) and os.path.getsize(data_save_path):
            done_profile = pd.read_csv(data_save_path)
            cprint(f"Load {len(done_profile)} samples from {data_save_path}!", on_color='on_green')
        else:
            done_profile = pd.DataFrame()
        not_done_samples_list = []

        for io in io_generator():
            op_task = cls.op(None, *io)
            op_desc = op_task.get_unique_desc()

            if done_profile.empty or not done_before(done_profile, op_desc):
                not_done_samples_list.append(io)

        cprint(f"Found {len(not_done_samples_list)} samples haven't been done!", on_color='on_red')
        return not_done_samples_list
