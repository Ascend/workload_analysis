import numpy as np

from framework.feature_base import FeatureGeneratorBase
from framework.model_packing import ModelPackBase


class InstanceNormOpModel(ModelPackBase):
    def param_check(self, inputs, outputs, attr):
        if len(inputs) not in [1]:
            return self.UN_SUCCESS, "LpNorm op need inputs with size in [1], get {}".format(len(inputs))
        if len(attr) not in [4]:
            return self.UN_SUCCESS, "LpNorm op need attr with size in [4], get {}".format(len(attr))
        return self.SUCCESS, ""

    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        return "common"


class InstanceNormFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        x = inputs[0]
        y = outputs[0]
        mean_shape = [x['shape'][0],x['shape'][1],1,1]
        variance_shape = [x['shape'][0],x['shape'][1],1,1]
        x_shape = x['shape']
        y_shape = y['shape']


        return dict(
            x = np.prod(x_shape),
            y = np.prod(y_shape),
            means = np.prod(mean_shape),
            variances = 3*np.prod(variance_shape),
            x_mean = np.prod(x_shape),
            variance_e = np.prod(variance_shape),
            de = np.prod(x_shape),
            mul = np.prod(x_shape),
            add = np.prod(x_shape),

            is_float16=int(x['dtype'].lower() == "float16"),
            is_float=int(x['dtype'].lower() == "float"))
