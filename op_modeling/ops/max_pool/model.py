import numpy as np
from framework.feature_base import FeatureGeneratorBase
from framework.model_packing import ModelPackBase

class MaxPoolOpModel(ModelPackBase):
    def param_check(self, inputs, outputs, attr):
        # 当前此处只对输入数目做校验，未来可能拓展
        if len(inputs) not in [4]:
            return self.UN_SUCCESS, "MaxPool op need inputs with size in [4], get {}".format(len(inputs))
        return self.SUCCESS, ""
    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        return "common"
class MaxPoolDetailFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        x = inputs[0]
        y = outputs[0]
        ksize = attrs.get('ksize')
        strides = attrs.get('strides')
        padding = attrs.get('padding', 0)
        x_shape = x['origin_shape']
        y_shape = y['origin_shape']
        flops = np.prod([x_shape[0], x_shape[3], ksize[1], ksize[2],y_shape[1], y_shape[2]])
        return dict(
            flops = flops,
            sum_strides = x_shape[1]/strides[1]* x_shape[2]/strides[2],
            H_x = x_shape[1],
            W_x = x_shape[2],
            N_y = y_shape[0],
            H_y = y_shape[1],
            W_y = y_shape[2],
            C_y = y_shape[3],
            H_ksize = ksize[1],
            W_ksize = ksize[2],
            H_strides = strides[1],
            W_strides = strides[2],
            is_SAME = int(attrs['padding'] == 'VALID'),
        )





