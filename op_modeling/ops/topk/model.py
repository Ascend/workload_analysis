import numpy as np

from framework.feature_base import FeatureGeneratorBase
from framework.model_packing import ModelPackBase

class TopKOpModel(ModelPackBase):
    def param_check(self, inputs, outputs, attr):
        # 当前此处只对输入数目做校验，未来可能拓展
        if len(inputs) not in [2]:
            return self.UN_SUCCESS, "TopK op need inputs with size in [2], get {}".format(len(inputs))
        return self.SUCCESS, ""

    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        return "common"

class TopKDetailFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        x = inputs[0]
        values = outputs[0]
        x_shape = x['shape']
        values_shape = values['shape']
        return dict(
            x=np.prod(x_shape),
            x_front=np.prod(x_shape[:-1]),
            x_behind=np.prod(x_shape[-1:]),
            values=np.prod(values_shape),
            values_behind=np.prod(values_shape[-1:]),
        )
