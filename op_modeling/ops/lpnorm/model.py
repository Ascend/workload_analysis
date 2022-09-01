import numpy as np
from framework.feature_base import FeatureGeneratorBase
from framework.model_packing import ModelPackBase


class LpNormOpModel(ModelPackBase):
    def param_check(self, inputs, outputs, attr):
        if len(inputs) not in [1]:
            return self.UN_SUCCESS, "LpNorm op need inputs with size in [1], get {}".format(len(inputs))
        return self.SUCCESS, ""

    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        return attr.get("p")


class LpNormFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        x = inputs[0]
        y = outputs[0]
        x_shape = x['shape']
        y_shape = y['shape']
        fc_shape1 = np.prod(x_shape) / np.prod(y_shape)
        return dict(
            x = np.prod(x_shape),
            fc_shape1 = fc_shape1,
            y = np.prod(y_shape),
            is_float16=int(x['dtype'].lower() == "float16"),
            is_float=int(x['dtype'].lower() == "float"))
