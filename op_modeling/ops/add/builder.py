#!/usr/bin/python
# -*- coding: UTF-8 -*-
from workload.config import config
from workload.framework.ai_metric import AICoreMetric
from workload.framework.op_register import RegisterOfBuilder
from workload.ops.add.model import AddOpModel
from workload.template.builder.vector.binary_op_builder import BinaryOpBuilder


@RegisterOfBuilder("Add")
class AddBuilder(BinaryOpBuilder):
    ai_metric = AICoreMetric.MEMORY_BANDWIDTH
    device_ids = [0]
    model_names = ['float16_model', "float_model", "int32_model"]
    dtypes = ['float16', 'float', 'int32']
    _train_data_info = list()
    _infer_data_info = list()
    op_model = AddOpModel
    model_pack = 'Add_' + config.soc_version + '.pkl'

