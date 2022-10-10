import numpy as np
from framework.feature_base import FeatureGeneratorBase
from framework.model_packing import ModelPackBase
class MaxPoolGradOpModel(ModelPackBase):
    def param_check(self, inputs, outputs, attr):
        # 当前此处只对输入数目做校验，未来可能拓展
        if len(inputs) not in [3]:
            return self.UN_SUCCESS, "MaxPoolGrad op need inputs with size in [4], get {}".format(len(inputs))
        return self.SUCCESS, ""

    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        return "common"
class MaxPoolGradDetailFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        x1 = inputs[0]
        x2 = inputs[1]
        ksize = attrs.get('ksize')
        strides = attrs.get('strides')
        padding = attrs.get('padding', 0)
        x1_shape = x1['origin_shape']
        x2_shape = x2['origin_shape']
        flops = np.prod([x1_shape[0], x1_shape[3], ksize[1], ksize[2],x2_shape[1], x2_shape[2]])
        return dict(
            flops = flops,
            sum_strides = x1_shape[1]/strides[1]* x1_shape[2]/strides[2],
            H_x1 = x1_shape[1],
            W_x1 = x1_shape[2],
            N_x2 = x2_shape[0],
            H_x2 = x2_shape[1],
            W_x2 = x2_shape[2],
            C_x2 = x2_shape[3],
            H_ksize = ksize[1],
            W_ksize = ksize[2],
            H_strides = strides[1],
            W_strides = strides[2],
            is_SAME = int(attrs['padding'] == 'VALID'),
        )