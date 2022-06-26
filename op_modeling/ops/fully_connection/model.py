import numpy as np

from framework.feature_base import FeatureGeneratorBase
from framework.model_packing import ModelPackBase


class FullyConnectionOpModel(ModelPackBase):
    def param_check(self, inputs, outputs, attr):
        # 当前此处只对输入数目做校验，未来可能拓展
        if len(inputs) not in [2, 3, 4]:
            return self.UN_SUCCESS, "FullyConnection op need inputs with size in [2,3,4], get {}".format(len(inputs))
        return self.SUCCESS, ""

    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        return "common"


class FullyConnectionDetailFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        x = inputs[0]
        w = inputs[1]
        y = outputs[0]
        transpose = attrs.get('transpose', False)
        axis = attrs.get('axis', 1)
        x_shape = x['shape']
        w_shape = w['shape']
        y_shape = y['shape']
        y_origin_shape = y['origin_shape']
        b_shape = [0]
        if len(inputs) >= 3:
            b = inputs[2]
            b_shape = b['shape']

        return dict(
            x=np.prod(x_shape),
            w=np.prod(w_shape),
            b=np.prod(b_shape),
            y=np.prod(y_shape),
            fc_prod=np.prod(y_shape[:axis]),
            fc_num=np.prod(y_origin_shape[:axis]),

            is_transpose=int(transpose),
            axis=axis,

            is_float16=int(x['dtype'].lower() == "float16"),
            is_int8=int(x['dtype'].lower() == "int8"),
            is_y_float16=int(y['dtype'].lower() == "float16"),
            is_y_float=int(y['dtype'].lower() == "float"),
            is_y_int32=int(y['dtype'].lower() == "int32"),
        )
