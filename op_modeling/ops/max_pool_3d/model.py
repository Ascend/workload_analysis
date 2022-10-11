import numpy as np

from framework.feature_base import FeatureGeneratorBase
from framework.model_packing import ModelPackBase


class MaxPool3DOpModel(ModelPackBase):
    def param_check(self, inputs, outputs, attr):
        if len(inputs) not in [4]:
            return self.UN_SUCCESS, "MaxPool3D op need inputs with size in [4], get {}".format(len(inputs))
        return self.SUCCESS, ""
    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        return "common"


class MaxPool3DDetailFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        x = inputs[0]
        y = outputs[0]
        x_shape = x['shape']
        k_shape = attrs.get('ksize')
        s_shape = attrs.get('strides')
        ceil_mode = attrs.get('ceil_mode', 0)
        y_shape = y['shape']
        x_d = x_shape[1]
        x_h = x_shape[3]
        x_w = x_shape[4]
        y_d = y_shape[1]
        y_h = y_shape[3]
        y_w = y_shape[4]
        Cin = x_shape[2] * x_shape[5]
        flops = np.prod([x_shape[0], Cin, k_shape[0], k_shape[1], k_shape[2], y_d, y_h, y_w])

        return dict(
            flops = flops,
            N = x_shape[0],
            Cin = Cin,
            x=np.prod(x_shape),
            k_shape=np.prod(k_shape),
            s_shape=np.prod(s_shape),
            
            x_d = x_d,
            x_h = x_h,
            x_w = x_w,
            y_d = y_d,
            y_h = y_h,
            y_w = y_w,

            k_d = k_shape[1],
            k_h = k_shape[2],
            k_w = k_shape[3],


            s_d = s_shape[1],
            s_h = s_shape[2],
            s_w = s_shape[3],
            y=np.prod(y_shape),

            ceil_mode = ceil_mode,
        )
