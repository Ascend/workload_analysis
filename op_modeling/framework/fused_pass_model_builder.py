#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
from abc import ABCMeta, abstractmethod

from workload.config import config
from workload.framework.ai_metric import AICoreMetric


class FusedBuilderInfo:
    def __init__(self, model, sample_feature, fused_file, origin_ops_file):
        self.model = model
        self.sample_feature = sample_feature
        self.fused_file = fused_file
        self.origin_ops_file = origin_ops_file


class FusedPassModelBuilder(metaclass=ABCMeta):
    op_type = ""  # 手动设置，通常op_type和融合规则名一样
    key_of_register = ''  # auto
    output_dir = config.output_dir
    ai_metric = AICoreMetric.PIPE_UTILIZATION

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
    def get_output_dir(cls, op_type):
        if not os.path.exists(cls.output_dir):
            os.makedirs(cls.output_dir)
        fusion_pass_dir = os.path.join(cls.output_dir, "fusion_pass")
        if not os.path.exists(fusion_pass_dir):
            os.makedirs(fusion_pass_dir)
        fusion_pass_data_dir = os.path.join(fusion_pass_dir, op_type)
        if not os.path.exists(fusion_pass_data_dir):
            os.makedirs(fusion_pass_data_dir)
        return fusion_pass_data_dir

    @classmethod
    def get_model_dir(cls, op_type, filename):
        op_data_dir = cls.get_output_dir(op_type)
        return os.path.join(op_data_dir, f"{filename}.pkl")

    @classmethod
    def get_ops_running_dir(cls, op_type):
        output_dir = cls.get_output_dir(op_type)
        ops_running_dir = os.path.join(output_dir, "ops_running")
        if not os.path.exists(ops_running_dir):
            os.makedirs(ops_running_dir)
        return ops_running_dir
