import itertools
from abc import abstractmethod, ABCMeta

from framework.tensor import HostTensor
from framework.tensor import DeviceTensor


class IOGenerator(metaclass=ABCMeta):
    """
    该类主要以算子维度进行策略的解析，并输出实例化后的系列Tensor
    策略之间会有多种关系，比如：
    1. 多个输入输出使用同一策略
    2. 某个输入依赖另一个输入的策略
    3. 各个输入输出间没有关系
    """

    # mode表示策略的关联类型，当前已有的关联类型包括
    # 1. product 求策略的笛卡尔积（该类型可能导致样本数过大采集困难）
    # 2. zip 打包策略，对各策略的case采取zip的一对一打包
    mode = "zip"

    def __init__(self, strategys):
        self.strategys = strategys

    def __call__(self, *args, **kwargs):
        concretization_strategys = []
        for strategy in self.strategys:
            concretization_strategys.append(strategy.get())

        groups = []
        if self.mode == "zip":
            for group in zip(*concretization_strategys):
                groups.append(group)
        # product
        else:
            groups = itertools.product(*concretization_strategys)

        for item in groups:
            io_strategys = self.get_io_strategys(*item)
            # 对于部分算子可能在get_io_strategys中发现策略的组合不可用，在此处直接丢弃
            if len(io_strategys) == 0:
                continue
            io = []
            for io_strategy in io_strategys:
                if self._is_tensor_strategy(io_strategy):
                    tensor = self._generate_tensor(io_strategy)
                    io.append(tensor)
                else:
                    io.append(io_strategy)

            yield io

    @classmethod
    def _generate_tensor(cls, tensor_strategy):
        is_host = tensor_strategy.get("is_host")
        if is_host:
            return HostTensor(tensor_strategy)
        else:
            return DeviceTensor(tensor_strategy)

    @classmethod
    def _is_tensor_strategy(cls, io_strategy):
        if type(io_strategy) is dict:
            return True
        return False

    @abstractmethod
    def get_io_strategys(self, *assemble_strategy):
        """
        该组成方法与具体算子有关，需要由算子自己定义
        由于算子本身的输入与输出间存在一定的关系，依据策略生成一系列内容
        此处需要维护策略实例化后，如何与算子的输入输出对应起来
        此处的输出会作为算子全量输入输出的生成策略
        :return:
        """
        pass
