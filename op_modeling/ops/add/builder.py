#!/usr/bin/python
# -*- coding: UTF-8 -*-
from config import config
from framework.ai_metric import AICoreMetric
from framework.op_register import RegisterOfBuilder
from ops.add.model import AddOpModel
from template.builder.vector.binary_op_builder import BinaryOpBuilder


@RegisterOfBuilder("Add")
class AddBuilder(BinaryOpBuilder):
    ai_metric = AICoreMetric.MEMORY_BANDWIDTH
    model_names = ['float16_model', "float_model", "int32_model"]
    dtypes = ['float16', 'float', 'int32']
    _train_data_info = list()
    _infer_data_info = list()
    op_model = AddOpModel
    model_pack = 'Add_' + config.soc_version + '.pkl'

