import numpy as np
import random
from functools import reduce

from framework.io_generater import IOGenerator
from framework.tensor_strategy import GeneralStrategy
from framework.tensor_strategy import TensorStrategy


class SingleVectorIOGenerator(IOGenerator):
    """
    单输入的vector算子的IO生成类，该类适用的算子应满足以下要求：
    1. formats无限制
    2. 输入的维度最大为5维
    """
    mode = "non_product"

    def __init__(self, dtype, is_training=True, n_sample=0):
        self.dtype = dtype
        self.n_sample = n_sample
        self.size_of_2GB = 1.5 * 1024 * 1024 * 1024 / 4
        if is_training:
            x_strategy = self.get_train_shapes()
        else:
            x_strategy = self.get_test_shapes()
        dims_strategy = self.get_dims_strategy(len(x_strategy))
        super().__init__([x_strategy, dims_strategy])

    @staticmethod
    def get_dims_strategy(size):
        dims = np.random.choice([1] + [2] * 3 + [3] * 3 + [4] * 2 + [5], size)
        dims_strategy = GeneralStrategy()
        dims_strategy.append([dim for dim in dims])
        return dims_strategy

    def get_train_shapes(self):
        np.random.seed(0)
        x = list(range(2, 514, 2)) * 2 \
            + [1] * 20 \
            + np.exp2(np.linspace(8, 12, 50)).astype(int).tolist() * 5 \
            + list(np.linspace(1024, 2048, 20).astype(np.int32)) \
            + list(np.linspace(2048, 2048 * 10, 50).astype(np.int32))
        x = np.array(x)
        x_shapes = np.random.choice(x, (self.n_sample, 5)).tolist()

        x_strategy = TensorStrategy()
        x_strategy.format(['ND']).shape(x_shapes).dtype(self.dtype)
        return x_strategy

    def get_test_shapes(self):
        np.random.seed(1)
        x = []
        for i in range(self.n_sample):
            x_i = np.random.randint(10, 10000, size=5).tolist()
            x.append(x_i)
        x_strategy = TensorStrategy()
        x_strategy.format(['ND']).shape(x).dtype(self.dtype)

        return x_strategy

    def get_io_strategys(self, *assemble_strategy):
        x, dim = assemble_strategy
        shape = x['shape']
        shape = shape[: dim]

        while self.get_size(shape) * 2 > self.size_of_2GB:
            index = random.randint(0, dim - 1)
            shape[index] = max(int(shape[index] / 2), 1)

        x['shape'] = shape

        y = x

        return [x, y]

    @staticmethod
    def get_size(shape):
        return reduce(lambda x, y: x * y, shape)


