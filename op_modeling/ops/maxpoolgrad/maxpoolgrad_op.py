import random
import math
from framework.op_base import OpBase
from framework.tensor_strategy import ValueStrategy, TensorStrategy
from template.generator.random_shape_generator import RandomShapeValueGenerator

class MaxPoolGradOp(OpBase):
    def __init__(self, device, x1, x2, grad, ksize, strides, padding, data_format, y):
        super().__init__(device, "MaxPoolGrad")
        self.input("x1", x1) \
            .input("x2", x2) \
            .input("grad", grad) \
            .attr("ksize", ksize) \
            .attr("strides", strides) \
            .attr("padding", padding) \
            .attr("data_format", data_format) \
            .output("y", y)

class MaxPoolGradIOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)
        random.seed(seed)
        x1_strategy = self.get_shape_strategy(mode='zip')
        y_strategy = x1_strategy
        size = len(x1_strategy.shapes)
        padding_strategy = self.get_value_strategy(['SAME', 'VALID'], size = size, rand_func = random.choice)
        [x2_strategy, grad_strategy, ksize_strategy, strides_strategy] = self.gen_strategies(x1_strategy, padding_strategy)
        self.strategys = [
            x1_strategy,
            x2_strategy,
            grad_strategy,
            y_strategy,
            ksize_strategy,
            strides_strategy,
            padding_strategy,
        ]
    def gen_strategies(self, x1_strategy, padding_strategy):
        x1_shapes = x1_strategy.shapes
        x1_dtypes = x1_strategy.dtypes
        paddings = padding_strategy.get()
        x2_dtypes = []
        x2_shapes = []
        grad_shapes = []
        grad_dtypes = []
        ksize_lists = []
        strides_lists = []
        for x1_shape, x1_dtype, padding in zip(x1_shapes, x1_dtypes, paddings):
            strides_h_value = random.randint(1,64)
            strides_w_value = random.randint(1,64)
            strides_value = [1, strides_h_value, strides_w_value, 1]
            strides_lists.append(strides_value)
            ksize_h_value = random.randint(1, 21)
            ksize_w_value = random.randint(1, 21)
            while ksize_h_value * ksize_w_value > 255 :
                ksize_h_value = random.randint(1, int(ksize_h_value / 2))
                ksize_w_value = random.randint(1, int(ksize_w_value / 2))
            if padding == 'SAME':   # x边缘用0填充
                ksize_value = [1, ksize_h_value, ksize_w_value, 1]
                ksize_lists.append(ksize_value)
                x2_h_value = math.ceil(x1_shape[1] / strides_h_value)
                x2_w_value = math.ceil(x1_shape[2] / strides_w_value)
                x2_shape = [x1_shape[0], x2_h_value, x2_w_value, x1_shape[3]]
                x2_shapes.append(x2_shape)
                grad_shapes.append(x2_shape)
            else:  # x边缘不用0填充
                # VALID模式下，ksize_h_value, ksize_w_value不能超过x_shape[1],x_shape[2]
                ksize_h_value = min(ksize_h_value, x1_shape[1])
                ksize_w_value = min(ksize_w_value, x1_shape[2])
                ksize_value = [1, ksize_h_value, ksize_w_value, 1]
                ksize_lists.append(ksize_value)
                x2_h_value = math.ceil((x1_shape[1] - ksize_h_value + 1) / strides_h_value)
                x2_w_value = math.ceil((x1_shape[2] - ksize_w_value + 1) / strides_w_value)
                x2_shape = [x1_shape[0], x2_h_value, x2_w_value, x1_shape[3]]
                x2_shapes.append(x2_shape)
                grad_shapes.append(x2_shape)
            x2_dtype = x1_dtype
            x2_dtypes.append(x2_dtype)
            grad_dtype = x1_dtype
            grad_dtypes.append(grad_dtype)
        # value类型
        ksize_strategy = ValueStrategy()
        ksize_strategy.append(ksize_lists)
        strides_strategy = ValueStrategy()
        strides_strategy.append(strides_lists)
        # tensor类型
        formats = ['NHWC'] * len(x2_shapes)
        x2_strategy = TensorStrategy()
        grad_strategy = TensorStrategy()
        if x2_shapes is not None:
            x2_strategy.format(formats).shape(x2_shapes).dtype(x2_dtypes)
            x2_strategy.set_mode('zip')
        if grad_shapes is not None:
            grad_strategy.format(formats).shape(x2_shapes).dtype(x2_dtypes)
            grad_strategy.set_mode('zip')
        return [x2_strategy, grad_strategy, ksize_strategy, strides_strategy]

    def get_io_strategys(self, *assemble_strategy):
        [x1, x2, grad, y, ksize, strides, padding] = assemble_strategy
        # data_format设置为默认值
        data_format = 'NHWC'
        return [x1, x2, grad, ksize, strides, padding, data_format, y]