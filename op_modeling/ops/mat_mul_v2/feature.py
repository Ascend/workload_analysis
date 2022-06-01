import math
from abc import ABC

import numpy as np

from framework.model_base import FeatureGeneratorBase


class MatMulV2DetailFeature(FeatureGeneratorBase, ABC):
    def swap(self, shape):
        temp_value = shape[0]
        shape[0] = shape[1]
        shape[1] = temp_value
        return shape

    def cal_feature(self, inputs, outputs, attrs: dict):
        inputs_x1 = inputs[0]["shape"]
        inputs_x2 = inputs[1]["shape"]
        if inputs[0]["format"] == "FRACTAL_NZ":
            inputs_x1 = self.swap(inputs_x1)
        if inputs[1]["format"] == "FRACTAL_NZ":
            inputs_x2 = self.swap(inputs_x2)
        if attrs["transpose_x1"]:
            inputs_x1 = self.swap(inputs_x1)
        if attrs["transpose_x2"]:
            inputs_x2 = self.swap(inputs_x2)
        input_x1_0 = inputs_x1[0]
        input_x1_1 = inputs_x1[1]
        input_x2_0 = inputs_x2[0]
        input_x2_1 = inputs_x2[1]
        flops = input_x1_0 * input_x1_1 * input_x2_1
        return dict(
            flops=flops,
            input_x1_value=np.prod(inputs_x1),
            input_x2_value=np.prod(inputs_x2),
            input_x1_0=input_x1_0,
            input_x1_1=input_x1_1,
            input_x2_0=input_x2_0,
            input_x2_1=input_x2_1,
            is_float=1 if inputs[0]["dtype"] == "float" else 0,
            is_float16=1 if inputs[0]["dtype"] == "float16" else 0,
            is_int32=1 if inputs[0]["dtype"] == "int32" else 0,
            is_int8=1 if inputs[0]["dtype"] == "int8" else 0,
            transpose_x1=1 if attrs["transpose_x1"] else 0,
            transpose_x2=1 if attrs["transpose_x2"] else 0,
            is_ND=1 if inputs[0]["format"] == "FORMAT_ND" else 0,
            offset_x=attrs["offset_x"],
            # offset_w=attrs["offset_w"],
        )
