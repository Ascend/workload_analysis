import numpy as np

from framework.feature_base import FeatureGeneratorBase
from framework.model_packing import ModelPackBase


class IouOpModel(ModelPackBase):
    def param_check(self, inputs, outputs, attr):
        # 当前此处只对输入数目做校验，未来可能拓展
        if len(inputs) != 2:
            return self.UN_SUCCESS
        return self.SUCCESS, ""

    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        return "common"

class IouDetailFeature(FeatureGeneratorBase):
    def cal_feature(self, inputs, outputs, attrs: dict):
        bboxes = inputs[0]
        gtboxes = inputs[1]
        overlap = outputs[0]

        bboxes_shape = bboxes['shape']
        gtboxes_shape = gtboxes['shape']
        overlap_shape = overlap['shape']


        return dict(
            bboxes=np.prod(bboxes_shape),
            gtboxes=np.prod(gtboxes_shape),
            overlap=np.prod(overlap_shape),
            mode=int(attrs['mode'] == 'iou'),
        )
