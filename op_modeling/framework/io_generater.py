import itertools
from abc import abstractmethod, ABCMeta
from framework.tensor import Tensor


class IOGenerator(metaclass=ABCMeta):
    """
    该类主要以算子维度进行策略的解析，并输出实例化后的系列Tensor
    策略之间会有多种关系，比如：
    1. 多个输入输出使用同一策略
    2. 某个输入依赖另一个输入的策略
    3. 各个输入输出间没有关系
    """
    mode = "product"

    def __init__(self, strategys):
        self.strategys = strategys

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

    def __call__(self, *args, **kwargs):
        concretization_strategys = []
        for strategy in self.strategys:
            concretization_strategys.append(strategy.get())

        if self.mode == "product":
            groups = itertools.product(*concretization_strategys)
        else:
            groups = []
            n_cases = len(concretization_strategys[0])
            for i in range(n_cases):
                case = [item[i] for item in concretization_strategys]
                groups.append(case)

        for item in groups:
            io_strategys = self.get_io_strategys(*item)
            if len(io_strategys) == 0:
                continue
            tensors = []
            for io_strategy in io_strategys:
                if type(io_strategy) is dict:
                    if "value" not in io_strategy.keys():
                        # 目前看来host的tensor一定是const
                        tensors.append(Tensor(io_strategy, mode='random', is_host=False))
                    else:
                        is_host = io_strategy.get("host", None)
                        if is_host:
                            tensors.append(Tensor(io_strategy, mode='const', is_host=True))
                        else:
                            tensors.append(Tensor(io_strategy, mode='const', is_host=False))
                else:
                    tensors.append(io_strategy)

            yield tensors
