import numpy as np

from framework.feature_base import FeatureGeneratorBase
from framework.model_packing import ModelPackBase


class ExpandDimsOpModel(ModelPackBase):
    def param_check(self, inputs, outputs, attr):
        # 当前此处只对输入数目做校验，未来可能拓展
        if len(inputs) not in [2]:
            return self.UN_SUCCESS, "FullyConnection op need inputs with size in [2], get {}".format(len(inputs))
        return self.SUCCESS, ""

    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        return "common"


class ExpandDimsDetailFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        x = inputs[0]
        axis = inputs[1]
        y = outputs[0]

        x_shape = x['shape']
        y_shape = y['shape']
        y_origin_shape = y['origin_shape']

        return dict(
            x=np.prod(x_shape),
            axis=axis,
            y=np.prod(y_shape),

            is_float16=int(x['dtype'].lower() == "float16"),
            is_int8=int(x['dtype'].lower() == "int8"),
            is_y_float16=int(y['dtype'].lower() == "float16"),
            is_y_float=int(y['dtype'].lower() == "float"),
            is_y_int32=int(y['dtype'].lower() == "int32"),
        )
