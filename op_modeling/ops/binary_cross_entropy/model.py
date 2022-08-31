import numpy as np
from framework.model_packing import ModelPackBase
from framework.feature_base import FeatureGeneratorBase

class BinaryCrossEntropyOpModel(ModelPackBase):
    def param_check(self, inputs, outputs, attr):
        if len(inputs) not in [2, 3]:
            return self.UN_SUCCESS, "BinaryCrossEntropy op need inputs with size in [2,3], get {}".format(len(inputs))
        return self.SUCCESS, ""

    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        return "common"

class BinaryCrossEntropyDetailFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        shape = inputs[0]['shape']
        reduction = attrs.get('reduction', 'none')
        reduction_type = 0
        if reduction == "none":
            reduction_type = 0
        elif reduction == "mean":
            reduction_type = 1
        elif reduction == "sum":
            reduction_type = 2
        dtype = 0
        if inputs[0]['dtype'] == "float16":
            dtype = 0
        elif inputs[0]['dtype'] == "float":
            dtype = 1

        return dict(
            flops_x = np.prod(shape),
            size = len(inputs),
            reduction_type = reduction_type,
            dtype = dtype
        )