import itertools
import random
from workload.framework.tensor_strategy import TensorStrategy


class SampleTensorStrategy(TensorStrategy):
    """
    该类的作用是部分算子参数过多，为防止最终组合策略样本量过大，进行采样
    """

    def __init__(self, sample_interval, mode='random'):
        self.sample_interval = sample_interval
        self.mode = mode
        self.seed = 10
        super(SampleTensorStrategy, self).__init__()

    def _instantiate(self):
        """
        依据策略规则，在整体策略的穷举列表中进行采样
        整体步骤如下：
        1. 将dtype,format, shape进行全排列组合
        2. 由于format和shape之间有对应关系，所以组合后需要移除不合理内容
        3. 依据采样间隔进行采样
        :return: list
        """

        # 去重
        self.formats = list(set(self.formats))
        self.dtypes = list(set(self.dtypes))

        if self._is_instantiated:
            return
        if self.mode == 'random':
            strategies = list(itertools.product(self.formats, self.dtypes, self.shapes))
            random.seed(self.seed)
            samples = random.sample(strategies, len(strategies) // self.sample_interval)

            for item in samples:
                format, dtype, shape = item
                self.strategys.append({"format": format, "dtype": dtype, "shape": shape})
        else:
            pass
