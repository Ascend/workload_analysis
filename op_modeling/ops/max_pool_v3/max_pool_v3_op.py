import random
import math
from framework.op_base import OpBase
from framework.tensor_strategy import ValueStrategy, TensorStrategy
from template.generator.random_shape_generator import RandomShapeValueGenerator

class MaxPoolV3Op(OpBase):
    def __init__(self, device, x, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode, y):
        super().__init__(device, "MaxPoolV3")
        self.input("x", x) \
            .attr("ksize", ksize) \
            .attr("strides", strides) \
            .attr("padding_mode", padding_mode) \
            .attr("pads", pads) \
            .attr("data_format", data_format) \
            .attr("global_pooling", global_pooling) \
            .attr("ceil_mode", ceil_mode) \
            .output("y", y)

class MaxPoolV3IOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)
        random.seed(seed)
        x_strategy = self.get_shape_strategy(mode = 'zip')
        size = len(x_strategy.shapes)
        global_pooling_strategy = self.get_value_strategy([True, False], size = size, rand_func=random.choice)
        [y_startegy, ksize_strategy, strides_strategy, pads_strategy, padding_mode_strategy,
         ceil_mode_strategy] = self.gen_strategies(x_strategy, global_pooling_strategy)
        self.strategys = [
            x_strategy,
            ksize_strategy,
            strides_strategy,
            padding_mode_strategy,
            pads_strategy,
            global_pooling_strategy,
            ceil_mode_strategy,
            y_startegy
        ]

    # 生成CACULATED模式下的pad
    def gen_pad(self, ceil_mode, ksize, strides):
        pad_top = random.randint(1, 5)
        pad_bottom = random.randint(1, 5)
        pad_left = random.randint(1, 5)
        pad_right = random.randint(1, 5)
        if ceil_mode:
            pad_top = min(pad_top, ksize[1] - 1)
            pad_bottom = min(pad_bottom, ksize[1] - strides[1] )
            pad_left = min(pad_left, ksize[2] - 1)
            pad_right = min(pad_right, ksize[2] - strides[2] )
        else:
            pad_top = min(pad_top, ksize[1] - 1)
            pad_bottom = min(pad_bottom, ksize[1] - 1)
            pad_left = min(pad_left, ksize[2] - 1)
            pad_right = min(pad_right, ksize[2] - 1)
        pad = [pad_top, pad_bottom, pad_left, pad_right]
        return pad

    # 生成padding_mode, ksize 和 strides
    def gen_mode_ksize_strides(self, x_shape):
        padding_mode = random.choice(['SAME', 'VALID', 'CALCULATED'])
        ksize_h_value = min(random.randint(1, 10), x_shape[1])
        ksize_w_value = min(random.randint(1, 10), x_shape[2])
        ksize_value = [1, ksize_h_value, ksize_w_value, 1]
        strides_h_value = min(random.randint(1, 5), ksize_h_value)
        strides_w_value = min(random.randint(1, 5), ksize_w_value)
        strides_value = [1, strides_h_value, strides_w_value, 1]
        return [padding_mode, ksize_value, strides_value]

    # 生成CACULATED模式下的y_shape
    def gen_calculated_y(self, ceil_mode, x_shape, ksize, strides, pad):
        ceil_mode_pad_h = 0
        ceil_mode_pad_w = 0
        if ceil_mode:
            ceil_mode_pad_h = strides[1] - 1
            ceil_mode_pad_w = strides[2] - 1
        y_h_value = math.floor((x_shape[1] + (pad[0] + pad[1]) -
                                ksize[1] + ceil_mode_pad_h) / strides[1] + 1)
        y_w_value = math.floor((x_shape[2] +  (pad[2] + pad[3]) -
                                ksize[2] + ceil_mode_pad_w) / strides[2] + 1)
        y_shape = [x_shape[0], y_h_value, y_w_value, x_shape[3]]
        return y_shape

    # 生成strategy需要的lists
    def gen_lists(self, x_strategy, global_pooling_strategy):
        x_shapes = x_strategy.shapes
        x_dtypes = x_strategy.dtypes
        global_poolings = global_pooling_strategy.get()
        y_shapes = []
        y_dtypes = []
        ksize_lists = []
        strides_lists = []
        pads_lists = []
        padding_mode_lists = []
        ceil_mode_lists = []
        for x_shape, x_dtype, global_pooling in zip(x_shapes, x_dtypes, global_poolings):
            pad = [0, 0, 0, 0]
            if global_pooling:
                padding_mode_lists.append('VALID')
                ceil_mode_lists.append(False)
                ksize_lists.append([1, x_shape[1], x_shape[2], 1])
                strides_lists.append([1, 1, 1, 1])
                y_shape = [x_shape[0], 1, 1, x_shape[3]]
            else:
                [padding_mode, ksize, strides] = self.gen_mode_ksize_strides(x_shape)
                padding_mode_lists.append(padding_mode)
                ksize_lists.append(ksize)
                strides_lists.append(strides)
                if padding_mode == 'SAME':   # x边缘用0填充
                    ceil_mode_lists.append(random.choice([True, False]))
                    y_h_value = math.ceil(x_shape[1] / strides[1])
                    y_w_value = math.ceil(x_shape[2] / strides[2])
                    y_shape = [x_shape[0], y_h_value, y_w_value, x_shape[3]]
                elif padding_mode == 'VALID':  # x边缘不用0填充
                    ceil_mode_lists.append(False)
                    y_h_value = math.ceil((x_shape[1]) / strides[1])
                    y_w_value = math.ceil((x_shape[2] ) / strides[2])
                    y_shape = [x_shape[0], y_h_value, y_w_value, x_shape[3]]
                elif padding_mode == 'CALCULATED':
                    ceil_mode = random.choice([True, False])
                    ceil_mode_lists.append(ceil_mode)
                    pad = self.gen_pad(ceil_mode, ksize, strides)
                    y_shape = self.gen_calculated_y( ceil_mode, x_shape, ksize, strides, pad)
            pads_lists.append(pad)
            y_shapes.append(y_shape)
            y_dtype = x_dtype
            y_dtypes.append(y_dtype)
            return [ksize_lists, strides_lists, pads_lists, padding_mode_lists, ceil_mode_lists, y_dtypes, y_shapes]

    # 得到y、ksize、strides等策略
    def gen_strategies(self, x_strategy, global_pooling_strategy):
        # value类型
        ksize_lists, strides_lists, pads_lists, padding_mode_lists, ceil_mode_lists, y_dtypes, y_shapes\
            = self.gen_lists(x_strategy, global_pooling_strategy)
        ksize_strategy = ValueStrategy()
        ksize_strategy.append(ksize_lists)
        strides_strategy = ValueStrategy()
        strides_strategy.append(strides_lists)
        pads_strategy = ValueStrategy()
        pads_strategy.append(pads_lists)
        padding_mode_strategy = ValueStrategy()
        padding_mode_strategy.append(padding_mode_lists)
        ceil_mode_strategy = ValueStrategy()
        ceil_mode_strategy.append(ceil_mode_lists)
        # tensor类型
        formats = ['NHWC'] * len(y_dtypes)
        y_strategy = TensorStrategy()
        if y_shapes is not None:
            y_strategy.format(formats).shape(y_shapes).dtype(y_dtypes)
            y_strategy.set_mode('zip')
        return [y_strategy, ksize_strategy, strides_strategy, pads_strategy,
                padding_mode_strategy, ceil_mode_strategy]

    def get_io_strategys(self, *assemble_strategy):
        [x, ksize, strides, padding_mode, pads, global_pooling, ceil_mode, y] = assemble_strategy
        data_format = 'NHWC'
        return [x, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode, y]