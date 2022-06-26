#!/usr/bin/python
# -*- coding: UTF-8 -*-
from framework.op_register import RegisterOf
from ops.add.add_op import AddOp
from template.builder.input_based_builder import BinaryInOpBuilder
from template.model_packing.input_based_model import DoubleInDtypeOpModel


@RegisterOf("Add")
class AddBuilder(BinaryInOpBuilder):
    dtypes = ['float16', 'float', 'int32']
    model_pack = DoubleInDtypeOpModel
    op = AddOp
