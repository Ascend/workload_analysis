import numpy as np
from framework.feature_base import FeatureGeneratorBase
from framework.model_packing import ModelPackBase

class PadV3OpModel(ModelPackBase):
    def param_check(self, inputs, outputs, attr):
        if len(inputs) not in [2, 3]:
            return self.UN_SUCCESS, "PadV3 op need inputs with size in [2,3], get {}".format(len(inputs))
        return self.SUCCESS, ""
    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        return "common"
    pass
class PadV3DetailFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        x = inputs[0]
        paddings = inputs[1]
        y = outputs[0]
        constant_values = attrs.get('constant_values', 0)
        mode = attrs.get('mode', 'constant')
        paddings_contiguous = attrs.get('paddings_contiguous', False)
        x_shape = x['shape']
        paddings_shape = paddings['shape']
        y_shape = y['shape']
        return dict(
            x = np.prod(x_shape),
            x_front=np.prod(x_shape[:-1]),
            x_behind=np.prod(x_shape[-1:]),
            y = np.prod(y_shape),
            y_front=np.prod(y_shape[:-1]),
            y_behind=np.prod(y_shape[-1:]),
            y_x = np.prod(y_shape)-np.prod(x_shape),
            is_float16 = int(x['dtype'].lower() == "float16"),
            is_float = int(x['dtype'].lower() == 'float'),
            is_y_float16 = int(y['dtype'].lower() == "float16"),
            is_y_float = int(y['dtype'].lower() == 'float'),
        )