#!/usr/bin/python
# -*- coding: UTF-8 -*-
import logging
import os
import time
import traceback
from abc import ABCMeta, abstractmethod

from termcolor import cprint

from config import config
from framework.ai_metric import AICoreMetric
from framework.device_manager import Device
from framework.profile import Profile


class OpModelBuilder(metaclass=ABCMeta):
    op = None  # 手动设置，算子的实现类
    op_type = ''  # 自动获取
    ai_metric = AICoreMetric.PIPE_UTILIZATION
    sample_interval = config.sample_interval
    device_ids = config.device_ids
    data_dir = "data"  # 用于保存采集数据的文件夹
    output_dir = "output"  # 用于暂存训练中间结果的文件夹
    pack_dir = "model"  # 用于保存最终打包的算子模型的文件夹

    @classmethod
    @abstractmethod
    def data_collect(cls):
        raise Exception("data_collect is not impl.")

    @classmethod
    @abstractmethod
    def modeling(cls):
        raise Exception("modeling is not impl.")

    @classmethod
    @abstractmethod
    def pack(cls):
        raise Exception("pack is not impl.")

    @classmethod
    @abstractmethod
    def test(cls):
        raise Exception("test is not impl.")

    @classmethod
    def get_data_path(cls, op_type, filename, sub_dir="ops"):
        op_data_dir = cls.get_data_dir(op_type, sub_dir)
        return os.path.join(op_data_dir, filename)

    @classmethod
    def get_pack_model_path(cls, filename):
        pack_dir = cls.get_pack_dir()
        return os.path.join(pack_dir, f"{filename}")

    @classmethod
    def get_handler_path(cls, op_type, filename, sub_dir="ops"):
        op_data_dir = cls.get_output_dir(op_type, sub_dir)
        return os.path.join(op_data_dir, f"{filename}")

    @classmethod
    def get_pack_dir(cls):
        if not os.path.exists(cls.pack_dir):
            os.makedirs(cls.pack_dir)
        return cls.pack_dir

    @classmethod
    def get_data_file(cls, op_type, soc_version):
        return op_type + "_" + soc_version + ".csv"

    @classmethod
    def get_test_data_file(cls, op_type, soc_version):
        return op_type + "_" + soc_version + "_test.csv"

    @classmethod
    def get_pack_file(cls, op_type, soc_version):
        return op_type + "_" + soc_version + ".pkl"

    @classmethod
    def get_data_dir(cls, op_type, sub_dir="ops"):
        if not os.path.exists(cls.data_dir):
            os.makedirs(cls.data_dir)
        op_data_dir = os.path.join(cls.data_dir, sub_dir, op_type)
        if not os.path.exists(op_data_dir):
            os.makedirs(op_data_dir)
        return op_data_dir

    @classmethod
    def get_output_dir(cls, op_type, sub_dir="ops"):
        if not os.path.exists(cls.output_dir):
            os.makedirs(cls.output_dir)
        op_output_dir = os.path.join(cls.output_dir, sub_dir, op_type)
        if not os.path.exists(op_output_dir):
            os.makedirs(op_output_dir)
        return op_output_dir

    @classmethod
    def _data_collect(cls, op, io_generator, data_save_path, device_id):
        if device_id is None:
            device_id = cls.device_ids[0]

        with Device(device_id) as d:
            with Profile(d, data_save_path, cls.op_type, interval=cls.sample_interval) as prof:
                cls._run_op_tasks(op, io_generator, d, prof)

    @classmethod
    def _run_op_tasks(cls, op, io_generator, d, prof):
        for k, io in enumerate(io_generator()):
            start = time.time()
            op_task = op(d, *io)
            prof.add_op_origin_desc(op_task.get_unique_desc())

            try:
                op_task.run(prof)
            except Exception as error:
                cprint(f"{error}", on_color="on_red")
                traceback.print_exc()
                prof.pop_op_origin_desc()
            finally:
                op_task.force_del()
                # 算子执行成功，data_manager添加内容
                logging.info(f"{k} device_id={d.id}: {str(op_task)}, {time.time() - start}")
