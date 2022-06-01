from typing import List
import math
import numpy as np
import pandas as pd
from framework.model_base import FeatureGeneratorBase, StandardizeDescBase
from framework.op_base import OpBase
from framework.io_generater import IOGenerator
from framework.tensor_strategy import GeneralStrategy
from framework.op_register import RegisterOfOp
from . import constant
import random
import copy


@RegisterOfOp("MatMulV2")
class MatMulV2Op(OpBase):
    """
    该类主要定义算子的形态
    """

    def __init__(
            self,
            device,
            input_x1,
            input_x2,
            bias,
            transpose_x1,
            transpose_x2,
            offset_x,
            output,
    ):
        super().__init__(device, "MatMulV2")
        self.attr("transpose_x1", transpose_x1)
        self.attr("transpose_x2", transpose_x2)
        self.attr("offset_x", offset_x)
        self.input("x1", input_x1)
        self.input("x2", input_x2)
        self.input("bias", bias)
        self.output("y", output)


class MatMulV2IOGenerator(IOGenerator):
    """
    该类主要返回算子执行需要的输入输出内容，并支持被算子定义所使用
    该类需要融合Tensor策略以及组合的工作
    """

    mode = "non_product"

    def __init__(self, dtype, is_training=True):
        self.dtype = dtype
        self.size_of_2GB = 1.5 * 1024 * 1024 * 1024 / 4
        # 设置随机数种子，保证每一次生成的训练样本集一致
        random.seed(0)
        self.list_of_control_params = []
        self.templates = []

        if is_training:
            for i in range(5000):
                for dtype_ in self.dtype:
                    attr = self.get_attr(dtype_)
                    input_x1, input_x2, bias = self.get_input(dtype_, attr)
                    self.list_of_control_params.append(
                        dict(
                            **{
                                "input_x1": input_x1,
                                "input_x2": input_x2,
                                "bias": bias,
                            },
                            **attr
                        )
                    )
            self.gen_templates_by_cps()
        else:
            self.set_test_templates()
        super(MatMulV2IOGenerator, self).__init__(self.gen_strategies_by_cases())

    def get_input(self, dtype, attr, x1_input_shape=None):
        if x1_input_shape is None:
            x1_input_shape = [random.randint(1, 1000), random.randint(1, 1000)]
        x1_input = {"format": "ND", "dtype": dtype, "shape": x1_input_shape}
        # x2
        if attr["transpose_x1"]:
            x1_input_shape = [x1_input_shape[1], x1_input_shape[0]]
        x2_input = copy.deepcopy(x1_input)
        x2_input["shape"] = [x1_input_shape[1], random.randint(1, constant.max_value)]
        if random.randint(1, 5) == 1 and dtype == "float16":  # 此时能保持ND格式
            x2_input["shape"] = [
                max(int(int(val / 16) * 16), 16) for val in x2_input["shape"]
            ]
            x1_input["shape"] = [
                max(int(int(val / 16) * 16), 16) for val in x1_input["shape"]
            ]

        x2_input_shape = x2_input["shape"]
        if attr["transpose_x2"]:
            x2_input["shape"] = [x2_input_shape[1], x2_input_shape[0]]
        # bias
        bias = copy.deepcopy(x2_input)
        bias["shape"] = (x2_input_shape[1],)
        if bias["dtype"].lower() == "int8":
            bias["dtype"] = "int32"
        return x1_input, x2_input, bias

    def get_attr(cls, dtype_):
        # attr
        transpose_list = [(False, False), (False, True), (True, False), (True, True)]
        if dtype_ == "int8":
            transpose_x1, transpose_x2 = transpose_list[0]
        if dtype_ == "float16":
            transpose_x1, transpose_x2 = transpose_list[random.randint(0, 3)]
        if dtype_ == "int32" or dtype_ == "float":
            transpose_x1, transpose_x2 = transpose_list[random.randint(0, 2)]
        offset_x = random.randint(-128, 127)
        attr = {
            "transpose_x1": transpose_x1,
            "transpose_x2": transpose_x2,
            "offset_x": offset_x,
        }
        return attr

    def gen_strategies_by_cases(self):
        input_x1_strategy = GeneralStrategy().append(
            [item[0] for item in self.templates]
        )
        input_x2_strategy = GeneralStrategy().append(
            [item[1] for item in self.templates]
        )
        bias_strategy = GeneralStrategy().append([item[2] for item in self.templates])
        transpose_x1_strategy = GeneralStrategy().append(
            [item[3] for item in self.templates]
        )
        transpose_x2_strategy = GeneralStrategy().append(
            [item[4] for item in self.templates]
        )
        offset_x_strategy = GeneralStrategy().append(
            [item[5] for item in self.templates]
        )
        return [
            input_x1_strategy,
            input_x2_strategy,
            bias_strategy,
            transpose_x1_strategy,
            transpose_x2_strategy,
            offset_x_strategy,
        ]

    def gen_templates_by_cps(self):
        for item in self.list_of_control_params:
            self.templates.append(
                (
                    item["input_x1"],
                    item["input_x2"],
                    item["bias"],
                    item["transpose_x1"],
                    item["transpose_x2"],
                    item["offset_x"],
                )
            )

    def _validate_cps(self, item):
        pass

    def compute_output(
            self,
            input_x1,
            input_x2,
            bias,
            transpose_x1,
            transpose_x2,
            offset_x,
    ):
        # x2
        x1_shape = input_x1["shape"]
        if transpose_x1:
            x1_shape = [x1_shape[1], x1_shape[0]]
        x2_shape = input_x2["shape"]
        if transpose_x2:
            x2_shape = [x2_shape[1], x2_shape[0]]
        y_shape = [x1_shape[0], x2_shape[1]]
        output = {
            "format": input_x1["format"],
            "dtype": input_x1["dtype"],
            "shape": y_shape,
        }
        if output["dtype"].lower() == "int8":
            output["dtype"] = "int32"
        return output

    def get_io_strategys(self, *assemble_strategy):
        # output的计算还是有问题，虽然代码已经是开发给的
        output = self.compute_output(*assemble_strategy)
        return (*assemble_strategy, output)

    def set_test_templates(self):
        test_shapes_x = [
            (300, 400),
            (400, 500),
            (600, 800),
            (50, 200),
            (50, 50),
        ]
        for input_shape in test_shapes_x:
            for dtype_ in self.dtype:
                attr = self.get_attr(dtype_)
                input_x1, input_x2, bias = self.get_input(dtype_, attr, input_shape)
                self.list_of_control_params.append(
                    dict(
                        **{
                            "input_x1": input_x1,
                            "input_x2": input_x2,
                            "bias": bias,
                        },
                        **attr
                    )
                )
        self.gen_templates_by_cps()


class MatMulV2Standardize(StandardizeDescBase):
    def standardize(self, inputs, outputs, attrs):
        # float int32 FORMAT_ND;FORMAT_ND;FORMAT_ND
        # float16 FRACTAL_NZ;FRACTAL_NZ;FORMAT_ND
        # int8 FRACTAL_NZ;FRACTAL_Z;FORMAT_ND
        # 进行格式转换，MatMulV2算子各输入最终在昇腾芯片上的运行数据格式为FRACTAL_NZ（可以通过采集到的profiling结果查看）
        # 更多格式介绍请阅 https://support.huaweicloud.com/TBEdevg-cann202training1/atlaste_10_0006.html
        if inputs[0]["dtype"] == "float16":
            input_x1 = inputs[0]["shape"]
            real_x1 = copy.deepcopy(input_x1)
            real_x1[0] = math.ceil(input_x1[1] / 16)
            real_x1[1] = math.ceil(input_x1[0] / 16)
            real_x1 = real_x1 + [16, 16]
            self.update_desc(inputs[0], "FRACTAL_NZ", real_x1, inputs[0]["dtype"])
            input_x2 = inputs[1]["shape"]
            real_x2 = copy.deepcopy(input_x2)
            real_x2[0] = math.ceil(input_x2[1] / 16)
            real_x2[1] = math.ceil(input_x2[0] / 16)
            real_x2 = real_x2 + [16, 16]
            self.update_desc(inputs[1], "FRACTAL_NZ", real_x2, inputs[0]["dtype"])
        elif inputs[0]["dtype"] == "int8":
            input_x1 = inputs[0]["shape"]
            real_x1 = copy.deepcopy(input_x1)
            real_x1[0] = math.ceil(input_x1[1] / 32)
            real_x1[1] = math.ceil(input_x1[0] / 16)
            real_x1 = real_x1 + [16, 32]
            self.update_desc(inputs[0], "FRACTAL_NZ", real_x1, inputs[0]["dtype"])
            input_x2 = inputs[1]["shape"]
            real_x2 = copy.deepcopy(input_x2)
            real_x2[0] = math.ceil(input_x2[0] / 32)
            real_x2[1] = math.ceil(input_x2[1] / 16)
            real_x2 = real_x2 + [16, 32]
            self.update_desc(inputs[1], "FRACTAL_Z", real_x2, inputs[0]["dtype"])
        else:
            self.update_desc(
                inputs[0], "FORMAT_ND", inputs[0]["shape"], inputs[0]["dtype"]
            )
            self.update_desc(
                inputs[1], "FORMAT_ND", inputs[1]["shape"], inputs[0]["dtype"]
            )
        return inputs, outputs, attrs


class MatMulV2linearFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        input_x1_0 = inputs[0].shape[0]
        input_x1_1 = inputs[0].shape[1]
        input_x2_0 = inputs[1].shape[0]
        input_x2_1 = inputs[1].shape[1]
        if attrs["transpose_x1"]:
            input_x1_0 = inputs[0].shape[1]
            input_x1_1 = inputs[0].shape[0]
        if attrs["transpose_x2"]:
            input_x2_0 = inputs[1].shape[1]
            input_x2_1 = inputs[1].shape[0]
        flops = input_x1_0 * input_x1_1 * input_x2_1
        return dict(
            flops=flops,
        )
