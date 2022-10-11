import random
import math
from framework.op_base import OpBase
from framework.tensor_strategy import ValueStrategy, TensorStrategy
from template.generator.random_shape_generator import RandomShapeValueGenerator

class MaxPoolOp(OpBase):
    def __init__(self, device, x, ksize, strides, padding, data_format, y):
        super().__init__(device, "MaxPool")
        self.input("x", x) \
            .attr("ksize", ksize)\
            .attr("strides", strides)\
            .attr("padding", padding)\
            .attr("data_format", data_format) \
            .output("y", y)\

class MaxPoolIOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)
        random.seed(seed)
        x_strategy = self.get_shape_strategy(mode='zip')
        size = len(x_strategy.shapes)
        padding_strategy = self.get_value_strategy(['SAME', 'VALID'], size = size, rand_func = random.choice)
        [y_strategy, ksize_strategy, strides_strategy] = self.gen_strategies(x_strategy, padding_strategy)
        self.strategys = [
            x_strategy,
            y_strategy,
            ksize_strategy,
            strides_strategy,
            padding_strategy
            ]
    def gen_strategies(self, x_strategy, padding_strategy):
        x_shapes = x_strategy.shapes
        x_dtypes = x_strategy.dtypes
        paddings = padding_strategy.get()
        y_shapes = []
        y_dtypes = []
        ksize_lists = []
        strides_lists = []
        for x_shape, x_dtype, padding in zip(x_shapes, x_dtypes, paddings):
            strides_h_value = random.randint(1, 5)
            strides_w_value = random.randint(1, 5)
            strides_value = [1, strides_h_value, strides_w_value, 1]
            strides_lists.append(strides_value)
            ksize_h_value = random.randint(1, 5)
            ksize_w_value = random.randint(1, 5)
            if padding == 'SAME':   # x边缘用0填充
                ksize_value = [1, ksize_h_value, ksize_w_value, 1]
                ksize_lists.append(ksize_value)
                y_h_value = math.ceil(x_shape[1] / strides_h_value)
                y_w_value = math.ceil(x_shape[2] / strides_w_value)
                y_shape = [x_shape[0], y_h_value, y_w_value, x_shape[3]]
                y_shapes.append(y_shape)
            else:  # x边缘不用0填充
                # VALID模式下，ksize_h_value, ksize_w_value不能超过x_shape[1],x_shape[2]
                ksize_h_value = min(ksize_h_value, x_shape[1])
                ksize_w_value = min(ksize_w_value, x_shape[2])
                ksize_value = [1, ksize_h_value, ksize_w_value, 1]
                ksize_lists.append(ksize_value)
                y_h_value = math.ceil((x_shape[1] - ksize_h_value + 1) / strides_h_value)
                y_w_value = math.ceil((x_shape[2] - ksize_w_value + 1) / strides_w_value)
                y_shape = [x_shape[0], y_h_value, y_w_value, x_shape[3]]
                y_shapes.append(y_shape)
            y_dtype = x_dtype
            y_dtypes.append(y_dtype)
        # value类型
        ksize_strategy = ValueStrategy()
        ksize_strategy.append(ksize_lists)
        strides_strategy = ValueStrategy()
        strides_strategy.append(strides_lists)
        # tensor类型
        formats = ['NHWC'] * len(y_shapes)
        y_strategy = TensorStrategy()
        if y_shapes is not None:
            y_strategy.format(formats).shape(y_shapes).dtype(y_dtypes)
            y_strategy.set_mode('zip')
        return [y_strategy, ksize_strategy, strides_strategy]
    def get_io_strategys(self, *assemble_strategy):
        [x, y, ksize, strides, padding] = assemble_strategy
        # data_format设置为默认值
        data_format = 'NHWC'
        return [x, ksize, strides, padding, data_format, y]