import itertools
from abc import abstractmethod, ABCMeta


class BaseStrategy(metaclass=ABCMeta):
    @abstractmethod
    def append(self, obj):
        pass

    @abstractmethod
    def get(self):
        pass


class TensorStrategy(BaseStrategy):
    """
    该类在使用过程中主要维护的是Tensor自身的策略,在最终使用时，最终调用instantiate实现所有组合关系
    针对该类中的具体功能函数，需要包含如下功能：
    1. 提供构成Tensor的各类属性的策略设置功能，自身作为集合体
    2. 调用属性组合生成策略，实例化Tensor的属性要求
    3. 支持依据Tensor属性要求，生成可以作为算子输入的Tensor内容
    """

    def __init__(self):
        """
        定义输入的数据结构
        """
        self.dtypes = []

        # relate params
        self.formats = list()
        self.origin_format = [None]
        self.shapes = list()
        # TODO: 针对数值暂未实现
        self.max_num = 1.0

        self._is_instantiated = False
        self.strategys = list()

    def __len__(self):
        return len(self.formats) * len(self.shapes) * len(self.dtypes)

    def format(self, formats):
        self.formats += formats
        return self

    def dtype(self, dtypes):
        self.dtypes += dtypes
        return self

    def shape(self, shapes: list(tuple())):
        """
        :param shapes:
        :return:
        """
        self.shapes += shapes
        return self

    def _instantiate(self):
        """
        依据策略规则，形成整体策略的穷举列表
        整体步骤如下：
        1. 将dtype,format, shape进行全排列组合
        2. 由于format和shape之间有对应关系，所以组合后需要移除不合理内容
        3.
        :return: list
        """

        # 去重
        self.formats = list(set(self.formats))
        self.dtypes = list(set(self.dtypes))

        if self._is_instantiated:
            return
        for item in itertools.product(self.shapes, self.formats, self.dtypes):
            shape, format, dtype = item
            self.strategys.append({"format": format, "dtype": dtype, "shape": shape})

    def append(self, obj):
        if not self._is_instantiated:
            self._instantiate()
        self.strategys += obj.get()
        return self

    def get(self):
        if not self._is_instantiated:
            self._instantiate()
        return self.strategys


class ConstTensorStrategy(BaseStrategy):
    def __init__(self):
        self.formats = ['ND']
        self.shapes = list()

        self._is_instantiated = False
        self.dtypes = list()
        self.values = list()

        self.strategies = list()

    def __len__(self):
        return len(self.formats) * len(self.shapes) * len(self.dtypes) * len(self.values)

    def value(self, values):
        self.values += values
        return self

    def shape(self, shapes):
        """
        :param shapes:
        :return:
        """
        self.shapes += shapes
        return self

    def append(self, obj):
        if not self._is_instantiated:
            self._instantiate()
        self.strategies += obj.get()
        return self

    def dtype(self, dtypes):
        self.dtypes += dtypes
        return self

    def _instantiate(self):
        # 去重
        self.dtypes = list(set(self.dtypes))

        if self._is_instantiated:
            return
        for item in itertools.product(self.shapes, self.formats, self.dtypes, self.values):
            shape, format_, dtype, value = item
            self.strategies.append({"format": format_, "dtype": dtype, "shape": shape, "value": value})

    def get(self):
        if not self._is_instantiated:
            self._instantiate()
        return self.strategies


class GeneralStrategy(BaseStrategy):
    """
    广义策略类
    """

    def __init__(self, cases=None):
        if cases is None:
            self.cases = []
        else:
            self.cases = cases
        self.index = -1

    def __len__(self):
        return len(self.cases)

    def append(self, cases):
        self.cases += cases
        return self

    def get(self):
        return self.cases
