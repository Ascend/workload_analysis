import numpy as np
from framework.feature_base import FeatureGeneratorBase


class L2LossDetailFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        x = inputs[0]
        y = outputs[0]
        x_shape = x['shape']
        y_shape = y['shape']

        return dict(
            x=np.prod(x_shape),
            y=np.prod(y_shape),
            is_float16=int(x['dtype'].lower() == "float16"),
            is_float=int(x['dtype'].lower() == "float")
        )