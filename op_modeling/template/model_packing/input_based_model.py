from abc import ABC
from abc import abstractmethod

from framework.model_packing import ModelPackBase


class DtypeOpModel(ModelPackBase, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def param_check(self: any, inputs: list, outputs: list, attr: dict) -> tuple:
        """
        算子参数校验
        :return: ret_code, msg
        """
        raise Exception("To be implement")

    def generate_key(self: any, inputs: list, outputs: list, attr: dict) -> str:
        """
        此类模型统一使用dtype作为子模型的索引
        """
        return inputs[0]['dtype']


class SingleInDtypeOpModel(DtypeOpModel):
    def __init__(self):
        super().__init__()

    def param_check(self: any, inputs: list, outputs: list, attr: dict) -> tuple:
        """
        算子参数校验
        :return: ret_code, msg
        """
        if len(inputs) != 1:
            return self.UN_SUCCESS, "Single in op need 1 inputs, get {}".format(len(inputs))
        return self.SUCCESS, ""


class DoubleInDtypeOpModel(DtypeOpModel):
    def __init__(self):
        super().__init__()

    def param_check(self: any, inputs: list, outputs: list, attr: dict) -> tuple:
        """
        算子参数校验
        :return: ret_code, msg
        """
        if len(inputs) != 2:
            return self.UN_SUCCESS, "Double in op need 2 inputs, get {}".format(len(inputs))
        return self.SUCCESS, ""
