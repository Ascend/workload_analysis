#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import time
import traceback
from abc import ABCMeta, abstractmethod

from termcolor import cprint

from config import config
from framework.ai_metric import AICoreMetric
from framework.data_manager import DataManager
from framework.device_manager import Device
from framework.model_base import StandardizeDescBase
from framework.op_register import OpManager
from framework.profile import Profile


class BuilderInfo:
    def __init__(self, model_name, models, sample_feature, data_file, io_generator, dtypes=None, use_block_dim=True,
                 stand_desc=StandardizeDescBase):
        self.model_name = model_name
        self.models = models
        self.dtypes = dtypes
        self.sample_feature = sample_feature
        self.data_file = data_file
        self.io_generator = io_generator
        self.use_block_dim = use_block_dim
        self.stand_desc = stand_desc


class OpModelBuilder(metaclass=ABCMeta):
    op_type = ''  # 手动设置，通常op_type和op_key一样
    key_of_register = ''  # 自动获取
    ai_metric = AICoreMetric.PIPE_UTILIZATION
    sample_interval = config.sample_interval
    device_ids = config.device_ids
    output_dir = config.output_dir

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
    def test(cls):
        raise Exception("test is not impl.")

    @classmethod
    def get_data_path(cls, op_type, filename):
        op_data_dir = cls.get_data_dir(op_type)
        return os.path.join(op_data_dir, f"{filename}.csv")

    @classmethod
    def get_model_path(cls, op_type, filename):
        op_data_dir = cls.get_data_dir(op_type)
        return os.path.join(op_data_dir, f"{filename}.pkl")

    @classmethod
    def get_data_dir(cls, op_type):
        if not os.path.exists(cls.output_dir):
            os.makedirs(cls.output_dir)
        op_data_dir = os.path.join(cls.output_dir, "ops", op_type)
        if not os.path.exists(op_data_dir):
            os.makedirs(op_data_dir)
        return op_data_dir

    @classmethod
    def _data_collect(cls, op_type, iogenerator, data_save_path, already_dump_file=None, device_id=None):
        if device_id is None:
            device_id = cls.device_ids[0]

        data_manager = DataManager(data_save_path, already_dump_file)

        with Device(device_id) as d:
            with Profile(d, data_manager, ai_metric=cls.ai_metric, interval=cls.sample_interval) as prof:
                Op = OpManager.get(cls.key_of_register)
                for k, io in enumerate(iogenerator()):
                    start = time.time()
                    op_task = Op(d, *io)

                    if prof.check_sample_exists(op_type, *op_task.get_unique_desc(), op_name=op_task.get_op_type()):
                        # 该样本已经执行过
                        continue

                    try:
                        op_task.run(prof)
                        op_task.force_del()
                        # 算子执行成功，data_manager添加内容
                        print(f"{k} device_id={device_id}:", str(op_task), time.time() - start)

                    except Exception as error:
                        print(f"[ERROR]: {k}, device_id={device_id}: ", str(op_task), time.time() - start)
                        cprint(f"{error}", on_color="on_red")
                        traceback.print_exc()
                        op_task.force_del()
