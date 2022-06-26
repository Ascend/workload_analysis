import numpy as np

from framework.feature_base import FeatureGeneratorBase


class ElementwiseFlopsFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        """
        计算量
        """
        input_shape = inputs[0]["shape"]
        flops = np.prod(input_shape)

        return dict(flops_log2=np.log2(flops))
