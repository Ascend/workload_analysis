#!/usr/bin/python
import random

from workload.framework.tensor import Tensor


class TensorGenerate:
    """
    该类主要以Tensor和Tensor的生成策略作为输入，输出Tensor序列。
    """

    def __init__(self, strategy):
        self.cases = strategy.to_realize()
        self.index = -1

    def __iter__(self):
        return self

    def __next__(self):
        # if self.index >= len(self.cases):
        #     return
        # self.index += 1
        # TODO: fix bug
        # tensor = TensorBase(self.cases[self.index])
        tensor = Tensor()
        return tensor
