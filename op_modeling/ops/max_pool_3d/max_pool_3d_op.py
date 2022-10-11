import random
import math
import numpy as np
from framework.op_base import OpBase
from framework.tensor_strategy import TensorStrategy, ValueStrategy
from template.generator.random_shape_generator import RandomShapeValueGenerator


class MaxPool3DOp(OpBase):
    def __init__(self, device, x, y, ksize, strides, padding, pads, dilation, ceil_mode, data_format):
        super().__init__(device, "MaxPool3D")
        self.input("x", x) \
            .output("y", y) \
            .attr("ksize", ksize) \
            .attr("strides", strides) \
            .attr("padding", padding) \
            .attr("pads", pads) \
            .attr("dilation", dilation) \
            .attr("ceil_mode", ceil_mode) \
            .attr("data_format", data_format)


class MaxPool3DIOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)

        random.seed(seed)
        np.random.seed(seed)
        x_strategy = self.get_shape_strategy(mode='zip')
        size = len(x_strategy.shapes)
        padding_strategy = self.get_value_strategy(['SAME', 'VALID'], size = size, rand_func = random.choice)
        ceil_mode_strategy = self.get_value_strategy([0, 1], size=size, rand_func=random.choice)
        [ksize_strategy, strides_strategy] = self.get_k_s_strategies(x_strategy, padding_strategy, ceil_mode_strategy)
        y_strategy = self.get_y_strategies(x_strategy, ksize_strategy, strides_strategy, padding_strategy)
        self.strategys = [
            x_strategy,
            ksize_strategy,
            strides_strategy,
            padding_strategy,
            ceil_mode_strategy,
            y_strategy,
        ]

    def get_shape_strategy(self, mode='product'):
        x_shapes = []
        x_formats = []
        x_dtypes = []
        for i in range(self.n_sample):
            for x_dtype in self.dtype:
                x_format = self.format[i % len(self.format)]
                n_value = random.randint(1, 100)
                c_value = random.randint(3, 64)
                d_value = random.randint(16, 100)
                h_value = random.randint(16, 100)
                w_value = random.randint(16, 100)
                
                shape = [n_value, d_value, h_value, w_value, c_value]
                    
                while self.get_size(shape) * 2 > self.size_of_half_gb:
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

    def get_k_s_strategies(self, x_strategy, padding_strategy, ceil_mode_strategy):
        x_shapes = x_strategy.shapes
        paddings = padding_strategy.get()
        ceil_modes = ceil_mode_strategy.get()
        ksize_lists = []
        strides_lists = []
        for x_shape, padding, ceil_mode in zip(x_shapes, paddings, ceil_modes):

            ksize_n_value = 1
            ksize_d_value = np.random.randint(2, 8)
            ksize_h_value = np.random.randint(2, 8)
            ksize_w_value = np.random.randint(2, 8)
            ksize_c_value = 1

            if padding == 'SAME':
                ksize = [ksize_n_value, ksize_d_value, ksize_h_value, ksize_w_value, ksize_c_value]
                ksize_lists.append(ksize)

                strides_n_value = 1
                strides_d_value = np.random.randint(1, ksize_d_value)
                strides_h_value = np.random.randint(1, ksize_h_value)
                strides_w_value = np.random.randint(1, ksize_w_value)
                strides_c_value = 1
                strides = [strides_n_value, strides_d_value, strides_h_value, strides_w_value, strides_c_value]
                strides_lists.append(strides)
            else:
                ksize_d_value = min(ksize_d_value, x_shape[1])
                ksize_h_value = min(ksize_h_value, x_shape[2])
                ksize_w_value = min(ksize_w_value, x_shape[3])
                ksize = [ksize_n_value, ksize_d_value, ksize_h_value, ksize_w_value, ksize_c_value]
                ksize_lists.append(ksize)

                strides_n_value = 1
                strides_d_value = np.random.randint(1, ksize_d_value)
                strides_h_value = np.random.randint(1, ksize_h_value)
                strides_w_value = np.random.randint(1, ksize_w_value)
                strides_c_value = 1
                strides = [strides_n_value, strides_d_value, strides_h_value, strides_w_value, strides_c_value]
                strides_lists.append(strides)
        
        ksize_strategy = ValueStrategy()
        ksize_strategy.append(ksize_lists)

        strides_strategy = ValueStrategy()
        strides_strategy.append(strides_lists)

        return [ksize_strategy, strides_strategy]


    def get_y_strategies(self, x_strategy, ksize_strategy, strides_strategy, padding_strategy):
        x_shapes = x_strategy.shapes
        x_dtypes = x_strategy.dtypes
        ksizes = ksize_strategy.get()
        m_strides = strides_strategy.get()
        paddings = padding_strategy.get()
        y_formats = []
        y_shapes = []
        y_dtypes = []
        for x_shape, x_dtype, ksize, strides, padding in zip(x_shapes, x_dtypes, ksizes, m_strides, paddings):

            if padding == 'SAME':
                y_d = math.ceil(x_shape[1] / strides[1])
                y_h = math.ceil(x_shape[2] / strides[2])
                y_w = math.ceil(x_shape[3] / strides[3])
                y_shape = [x_shape[0], y_d, y_h, y_w, x_shape[4]]
                y_formats.append('NDHWC')
                y_shapes.append(y_shape)
                y_dtypes.append(x_dtype)
            else:
                y_d = math.ceil((x_shape[1] - ksize[1] + 1) / strides[1])
                y_h = math.ceil((x_shape[2] - ksize[2] + 1) / strides[2])
                y_w = math.ceil((x_shape[3] - ksize[3] + 1) / strides[3])
                y_shape = [x_shape[0], y_d, y_h, y_w, x_shape[4]]
                y_formats.append('NDHWC')
                y_shapes.append(y_shape)
                y_dtypes.append(x_dtype)

        y_strategy = TensorStrategy()
        if y_shapes is not None:
            y_strategy.format(y_formats).shape(y_shapes).dtype(y_dtypes)
            y_strategy.set_mode('zip')

        return y_strategy

    def get_io_strategys(self, *assemble_strategy):
        pads = [0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1]
        data_format = 'NDHWC'
        [x, ksize, strides, padding, ceil_mode, y] = assemble_strategy
        return [x, y, ksize, strides, padding, pads, dilation, ceil_mode, data_format]

