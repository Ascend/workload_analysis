import os
from abc import ABC
from typing import List
from multiprocessing import Process
import pandas as pd
from termcolor import cprint

from framework.data_manager import DataManager
from framework.op_model_builder import BuilderInfo
from framework.op_register import OpManager
from framework.util import chunks
from template.builder.general.train_test_builder import TrainTestBuilder


class SimpleGenerator:
    """
    广义策略类
    """

    def __init__(self, cases=None):
        if cases is None:
            self.cases = []
        else:
            self.cases = cases

    def __call__(self, *args, **kwargs):
        for item in self.cases:
            yield item


class MultiProcessBuilder(TrainTestBuilder, ABC):
    _train_data_info: List[BuilderInfo] = list()
    _infer_data_info: List[BuilderInfo] = list()

    @classmethod
    def data_collect_by_multi_process(cls, model_info_list: List[BuilderInfo]):
        """
        请在此处完成训练和测试用的数据集的处理
        :return:
        """
        device_num = len(cls.device_ids)

        for item in model_info_list:
            data_save_path = item.data_file
            data_save_paths = []
            for i, device_id in enumerate(cls.device_ids):
                # 重命名
                i_data_save_path = data_save_path.replace(".csv", f"_device{device_id}.csv")
                data_save_paths.append(i_data_save_path)

            # 将所有结果合并， 上一次可能异常结束，所以先合并以方便查看已经生成了多少样本
            cls._combine_all_dfs(data_save_paths, data_save_path)

            # 获取所有还未被处理的样本数据
            io_generator = item.io_generator
            data = cls._get_not_done_samples(data_save_path, io_generator)
            if not data:
                continue

            sub_generators = list(chunks(data, device_num))

            process_list = []
            for i, device_id in enumerate(cls.device_ids):
                if i >= len(sub_generators):
                    continue

                sub_generator = SimpleGenerator(sub_generators[i])
                p = Process(target=cls._data_collect,
                            args=(cls.op_type, sub_generator, data_save_paths[i], data_save_path, device_id))
                p.start()
                print(f"Launching process of device_id={device_id}, data_save_path={data_save_paths[i]}")
                os.system("sleep 5")
                process_list.append(p)

            for p in process_list:
                p.join()

            # 将所有结果合并
            cls._combine_all_dfs(data_save_paths, data_save_path)
            for i_data_save_path in data_save_paths:
                os.system(f"rm -rf {i_data_save_path}")

    @classmethod
    def _combine_all_dfs(cls, data_save_paths, data_save_path):
        if not data_save_paths:
            return

        if os.path.exists(data_save_path):
            dfs = [pd.read_csv(data_save_path)]
        else:
            dfs = []

        for path in data_save_paths:
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))

        if dfs:
            df = pd.concat(dfs)
            df.drop_duplicates(ignore_index=True, inplace=True)
            df.to_csv(data_save_path, index=False)

    @classmethod
    def data_collect(cls, mode='all'):
        """
        请在此处完成训练和测试用的数据集的处理
        :return:
        """
        cls.init()
        if mode == 'all' or mode == 'train':
            cls.data_collect_by_multi_process(cls._train_data_info)

        if mode == 'all' or mode == 'infer':
            cls.data_collect_by_multi_process(cls._infer_data_info)

    @classmethod
    def _get_not_done_samples(cls, data_save_path, io_generator):
        Op = OpManager.get(cls.key_of_register)
        data_manager = DataManager(data_save_path)
        not_done_samples_list = []

        for io in io_generator():
            op_task = Op(None, *io)
            if not data_manager.check_sample_exists(cls.op_type, *op_task.get_unique_desc()):
                not_done_samples_list.append(io)

        cprint(f"Found {len(not_done_samples_list)} samples haven't been done!", on_color='on_red')
        return not_done_samples_list

    @classmethod
    def init(cls):
        pass
