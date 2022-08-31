import numpy as np

from framework.feature_base import FeatureGeneratorBase
from framework.model_packing import ModelPackBase


class PoolingOpModel(ModelPackBase):
    def param_check(self, inputs, outputs, attr):
        # 当前此处只对输入数目做校验，未来可能拓展
        return self.SUCCESS, ""
    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        filters_conditions ={'Avg_Global': [1, True], 'Avg_unGlobal': [1, False], 'Max_Global': [0, True], \
                             'Max_unGlobal': [0, False]}
        for key, values in filters_conditions.items():
            if attr['mode'] == values[0] and attr['global_pooling'] == values[1]:
                return key
        return "common"


class PoolingDetailFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        # 获得字典
        x = inputs[0]
        y = outputs[0]
        global_pooling = attrs.get('global_pooling', False)
        mode = attrs.get('mode', 0)
        window = attrs.get('window', [1, 1])
        stride = attrs.get('stride', [1, 1])
        padding =  attrs.get('pad', [0, 0, 0, 0])
        ceil_mode = attrs.get('ceil_mode', 0)
        y_shape = y['shape']
        x_shape = x['shape']
        if(global_pooling):
            window[0] = x_shape[2]
            window[1] = x_shape[3]

        Cin = x_shape[1]*x_shape[4]
        flops = np.prod([x_shape[0], Cin, window[0], window[1], y_shape[2], y_shape[3]])

        return dict(
            flops = flops,
            N = x_shape[0],
            Cin = Cin,
            x_h = x_shape[2],
            x_w = x_shape[3],
            y_h = y_shape[2],
            y_w = y_shape[3],

            window_h = window[0],
            window_w = window[1],

            stride_h = stride[0],
            stride_w = stride[1],

            pad_top = padding[0],
            pad_left = padding[2],

            ceil_mode = ceil_mode,
        )