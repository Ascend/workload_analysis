import itertools
from abc import abstractmethod, ABCMeta


class BaseStrategy(metaclass=ABCMeta):
    """
    该类是策略的基础定义，这里的策略可以理解为一个对象各成员不同取值的排列组合，对于一个策略，其基本行为包括
    1. 扩充策略
    2. 获取当前策略中的全部排列组合
    """

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
        self.max_num = 1.0
        self.is_host = False
        self.mode = 'product'

        self._is_instantiated = False
        self.strategys = list()

    def set_host(self):
        self.is_host = True

    def set_mode(self, mode):
        self.mode = mode

    def format(self, formats: list):
        self.formats += formats
        return self

    def dtype(self, dtypes: list):
        self.dtypes += dtypes
        return self

    def shape(self, shapes: list):
        """
        :param shapes:
        :return:
        """
        self.shapes += shapes
        return self

    def append(self, obj):
        if not self._is_instantiated:
            self._instantiate()
        self.strategys += obj.get()
        return self

    def get(self):
        if not self._is_instantiated:
            self._instantiate()
        return self.strategys

    def param_check(self):
        if len(self.shapes) != len(self.formats) or len(self.formats) != len(self.dtypes):
            raise Exception('Size of shapes: {}, formats: {}, dtypes: {}, must be same.'
                            .format(len(self.shapes), len(self.formats), len(self.dtypes)))

    def _instantiate(self):
        """
        依据策略规则，形成整体策略的穷举列表
        整体步骤如下：
        1. 将dtype,format, shape进行全排列组合
        2. 由于format和shape之间有对应关系，所以组合后需要移除不合理内容
        :return: list
        """

        if self._is_instantiated:
            return
        if self.mode == 'zip':
            self.param_check()
            groups = zip(self.shapes, self.formats, self.dtypes)
        else:
            # product
            # 去重
            self.formats = list(set(self.formats))
            self.dtypes = list(set(self.dtypes))
            groups = itertools.product(self.shapes, self.formats, self.dtypes)
        for item in groups:
            shape, format_, dtype = item
            self.strategys.append({"format": format_, "dtype": dtype, "shape": shape, "is_host": self.is_host})


class ValueStrategy(BaseStrategy):
    """
    该类在使用过程中主要维护的是纯数值的策略
    """

    def __init__(self, cases=None):
        if cases is None:
            self.cases = []
        else:
            self.cases = cases

    def __len__(self):
        return len(self.cases)

    def append(self, cases):
        self.cases += cases
        return self

    def get(self):
        return self.cases
