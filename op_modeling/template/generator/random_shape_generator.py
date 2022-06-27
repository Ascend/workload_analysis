from abc import ABC
from functools import reduce
import random
import numpy as np

from framework.io_generater import IOGenerator
from framework.tensor_strategy import TensorStrategy, ValueStrategy


class RandomShapeGenerator(IOGenerator, ABC):
    """
    单输入算子的IO生成类，该模板生成的策略满足：
    1. dtypes可以指定多种，format恒定为ND
    2. shape支持1-{max_dim}维,(在昇腾算子场景下，常用的max_dim为4或5)
    3. shape各dim取值范围固定
    """

    def __init__(self, dtypes, max_dim=5, n_sample=0, seed=0):
        self.dtypes = dtypes
        self.formats = ["ND"]
        self.n_sample = n_sample
        self.max_dim = max_dim
        self.size_of_2gb = 1.5 * 1024 * 1024 * 1024 / 4
        x_strategy = self.get_random_shape_strategy(seed)
        super().__init__([x_strategy])

    @staticmethod
    def get_size(shape):
        return reduce(lambda x, y: x * y, shape)

    def get_random_shape_strategy(self, seed=0):
        # 设置随机数种子，保证每一次生成的样本集一致
        np.random.seed(seed)
        x = list(range(2, 514, 2)) * 2 \
            + [1] * 20 \
            + np.exp2(np.linspace(8, 12, 50)).astype(int).tolist() * 5 \
            + list(np.linspace(1024, 2048, 20).astype(np.int32)) \
            + list(np.linspace(2048, 2048 * 10, 50).astype(np.int32))
        x = np.array(x)
        # 首先统一生成最大dim的x
        x_shapes = np.random.choice(x, (self.n_sample, self.max_dim)).tolist()

        # 获取x的最终shape
        dims = np.random.choice(list(range(1, self.max_dim + 1)), self.n_sample).tolist()
        x_shapes = list(shape[:dim] for shape, dim in zip(x_shapes, dims))

        for shape in x_shapes:
            # 避免内存不足
            while self.get_size(shape) * 2 > self.size_of_2gb:
                index = np.random.randint(0, len(shape) - 1)
                shape[index] = max(int(shape[index] / 2), 1)
        x_strategy = TensorStrategy()
        # 使用product的模式实现随机shape和dtype的组合
        x_strategy.format(self.formats).shape(x_shapes).dtype(self.dtypes)
        return x_strategy


class RandomShapeSingleInOutGenerator(RandomShapeGenerator):
    """
    该类生成单输入、单输出、且输入输出shape相同的策略
    """

    def get_io_strategys(self, *assemble_strategy):
        x, = assemble_strategy
        y = x

        return [x, y]


class RandomShapeBinaryInGenerator(RandomShapeGenerator):
    """
    该类生成双输入，单输出，且输入输出shape相同的策略
    """

    def get_io_strategys(self, *assemble_strategy):
        x, = assemble_strategy
        y = x
        return [x, x, y]


class RandomShapeValueGenerator(IOGenerator, ABC):
    """
     算子的shape和value随机生成类，该模板生成的策略满足：
     1. 支持多种的format和dtype，format支持"NCHW", "NHWC" 和 "ND"
     2. shape生成支持"product" 和 "zip" 两种模式
     3. value生成支持不同的随机数生成器 rand_func
    """
    def __init__(self, sample_dtype, sample_format, n_sample=0):
        self.dtype = sample_dtype
        self.format = sample_format
        self.n_sample = n_sample
        self.size_of_1gb = 1024 * 1024 * 1024 / 4
        strategies = []
        super().__init__(strategies)

    @staticmethod
    def get_size(shape):
        return reduce(lambda x, y: x * y, shape)

    @staticmethod
    def get_value_strategy(*args, size=None, rand_func=None):
        """
        随机生成value/list类型的数据，主要用于attr的生成
        :param args: 可变参数，args=[rand_range1, (rand_range2,...,rand_rangen, rang_func)]
        rand_range表示随机值的生成的范围，包括两类：
          1. 如果rand_range=[rand_range1,rand_range2],表示随机数生成的上下界(适用于random.randint和random.uniform等函数)
          2. 如果rand_range=[rand_range1]，适用于random.choice等一个参数的生成器
        :param size: list长度，attr的数据量
        :param rand_func: 随机值生成器，比如random.randint，random.uniform, random.choice
        :return:
        """
        rand_range = args
        if rand_func is None and len(args) >= 3:
            rand_func = args[-1]
            size = args[-2]
            rand_range = args[:-2]
        elif size is None and len(args) >= 2:
            size = args[-1]
            rand_range = args[:-1]
        if size is None or rand_func is None:
            # 不生成数据
            size = 0
        value_strategy = ValueStrategy()
        value_strategy.append([rand_func(*rand_range) for _ in range(size)])
        return value_strategy

    def get_shape_strategy(self, mode='product'):
        x_shapes = []
        x_formats = []
        x_dtypes = []
        for i in range(self.n_sample):
            for x_dtype in self.dtype:
                x_format = self.format[i % len(self.format)]
                n_value = random.randint(1, 100)
                c_value = random.randint(3, 64)
                h_value = random.randint(16, 2048)
                w_value = random.randint(16, 2048)
                if x_format == 'NHWC':
                    shape = [n_value, h_value, w_value, c_value]
                elif x_format == 'NCHW':
                    shape = [n_value, c_value, h_value, w_value]
                else:
                    # ND
                    dim = random.randint(1, 5)
                    shape = [random.randint(2, 2048) for _ in range(dim)]
                while self.get_size(shape) * 2 > self.size_of_1gb:
                    index = random.randint(0, len(shape) - 1)
                    shape[index] = max(int(shape[index] / 2), 1)
                x_shapes.append(shape)
                x_formats.append(x_format)
                x_dtypes.append(x_dtype)
        x_strategy = TensorStrategy()
        if x_shapes is not None:
            x_strategy.set_mode(mode)
            x_strategy.format(x_formats).shape(x_shapes).dtype(x_dtypes)
        return x_strategy
