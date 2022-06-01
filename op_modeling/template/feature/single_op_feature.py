import numpy as np

from framework.model_base import FeatureGeneratorBase


class SingleVectorFeatureBase(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        """
        Relu算子的计算值理论上只与计算量相关
        """
        input_shape = inputs[0]["shape"]
        flops = np.prod(input_shape)

        return dict(flops_log2=np.log2(flops))
