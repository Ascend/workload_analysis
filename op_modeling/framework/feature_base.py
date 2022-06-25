from abc import abstractmethod


class FeatureGeneratorBase:
    def __init__(self):
        self.attr = {}

    @abstractmethod
    def cal_feature(self, inputs, outputs, attrs):
        """
        计算特征
        :return: {f1:xx, f2:xx}
        """
        raise Exception("This function is not implement")
