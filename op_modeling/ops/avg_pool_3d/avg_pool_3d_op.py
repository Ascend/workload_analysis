import random
import numpy as np
from framework.op_base import OpBase
from framework.tensor_strategy import TensorStrategy, ValueStrategy
from template.generator.random_shape_generator import RandomShapeValueGenerator


class AvgPool3DOp(OpBase):
    def __init__(self, device, x, y, ksize, strides, pads, ceil_mode, count_include_pad, divisor_override, data_format):
        super().__init__(device, "AvgPool3D")
        self.input("x", x) \
            .output("y", y) \
            .attr("ksize", ksize) \
            .attr("strides", strides) \
            .attr("pads", pads) \
            .attr("ceil_mode", ceil_mode) \
            .attr("count_include_pad", count_include_pad) \
            .attr("divisor_override", divisor_override) \
            .attr("data_format", data_format)


class AvgPool3DIOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)

        random.seed(seed)
        np.random.seed(seed)
        x_strategy = self.get_shape_strategy(mode='zip')
        size = len(x_strategy.shapes)
        ksize_strategy = self.get_value_strategy([1, 3], size=size, rand_func=random.choice)
        strides_strategy = self.get_value_strategy([1, 3], size=size, rand_func=random.choice)
        [k_strategy, s_strategy, data_format_strategy, y_strategy] = self.gen_strategies(x_strategy, 
                                                                    ksize_strategy, strides_strategy)
        self.strategys = [
            x_strategy,
            k_strategy,
            s_strategy,
            data_format_strategy,
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
                if x_format == 'NCDHW':
                    shape = [n_value, c_value, d_value, h_value, w_value]
                else:
                    # NDHWC
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

    def gen_strategies(self, x_strategy, ksize_strategy, strides_strategy):
        x_shapes = x_strategy.shapes
        x_dtypes = x_strategy.dtypes
        m_ksize = ksize_strategy.get()
        m_strides = strides_strategy.get()
        m_data_format = x_strategy.formats
        k_sizes = []
        k_shapes = []
        k_dtypes = []
        s_sizes = []
        s_shapes = []
        s_dtypes = []
        data_formats = []
        y_shapes = []
        y_dtypes = []
        for x_shape, x_dtype, ksize, stride, data_format in zip(x_shapes, x_dtypes, m_ksize, m_strides, m_data_format):
            if ksize == 1:
                ksizes = np.random.randint(1, 7)
                d_ksizes = h_ksizes = w_ksizes = ksizes
                if data_format == 'NCDHW':
                    if x_shape[2] < d_ksizes:
                        d_ksizes = np.random.randint(1, x_shape[2] + 1)
                    if x_shape[3] < h_ksizes:
                        h_ksizes = np.random.randint(1, x_shape[3] + 1)
                    if x_shape[4] < w_ksizes:
                        w_ksizes = np.random.randint(1, x_shape[4] + 1)
                    k_shape = [d_ksizes, h_ksizes, w_ksizes]
                    k_shapes.append(k_shape)
                    k_dtypes.append('int32')
                else:
                    if x_shape[1] < d_ksizes:
                        d_ksizes = np.random.randint(1, x_shape[1] + 1)
                    if x_shape[2] < h_ksizes:
                        h_ksizes = np.random.randint(1, x_shape[2] + 1)
                    if x_shape[3] < w_ksizes:
                        w_ksizes = np.random.randint(1, x_shape[3] + 1)
                    k_shape = [d_ksizes, h_ksizes, w_ksizes]
                    k_shapes.append(k_shape)
                    k_dtypes.append('int32')
            else:
                d_ksizes = np.random.randint(1, 7)
                h_ksizes = np.random.randint(1, 7)
                w_ksizes = np.random.randint(1, 7)
                if data_format == 'NCDHW':
                    if x_shape[2] < d_ksizes:
                        d_ksizes = np.random.randint(1, x_shape[2] + 1)
                    if x_shape[3] < h_ksizes:
                        h_ksizes = np.random.randint(1, x_shape[3] + 1)
                    if x_shape[4] < w_ksizes:
                        w_ksizes = np.random.randint(1, x_shape[4] + 1)
                    k_shape = [d_ksizes, h_ksizes, w_ksizes]
                    k_shapes.append(k_shape)
                    k_dtypes.append('int32')
                else:
                    if x_shape[1] < d_ksizes:
                        d_ksizes = np.random.randint(1, x_shape[1] + 1)
                    if x_shape[2] < h_ksizes:
                        h_ksizes = np.random.randint(1, x_shape[2] + 1)
                    if x_shape[3] < w_ksizes:
                        w_ksizes = np.random.randint(1, x_shape[3] + 1)
                    k_shape = [d_ksizes, h_ksizes, w_ksizes]
                    k_shapes.append(k_shape)
                    k_dtypes.append('int32')

            if stride == 1:
                strides = np.random.randint(1, 7)
                d_strides = h_strides = w_strides = strides
                if d_ksizes < d_strides:
                    d_strides = np.random.randint(1, d_ksizes + 1)
                if h_ksizes < h_strides:
                    h_strides = np.random.randint(1, h_ksizes + 1)
                if w_ksizes < w_strides:
                    w_strides = np.random.randint(1, w_ksizes + 1)
                s_shape = [d_strides, h_strides, w_strides]
                s_shapes.append(s_shape)
                s_dtypes.append('int32')
            else:
                d_strides = np.random.randint(1, 7)
                h_strides = np.random.randint(1, 7)
                w_strides= np.random.randint(1, 7)
                if d_ksizes < d_strides:
                    d_strides = np.random.randint(1, d_ksizes + 1)
                if h_ksizes < h_strides:
                    h_strides = np.random.randint(1, h_ksizes + 1)
                if w_ksizes < w_strides:
                    w_strides = np.random.randint(1, w_ksizes + 1)
                s_shape = [d_strides, h_strides, w_strides]
                s_shapes.append(s_shape)
                s_dtypes.append('int32')

            data_formats.append(data_format)

            y_shape = [0, 0, 0, 0, 0]
            if data_format == 'NCDHW':
                y_shape[0] = x_shape[0]
                y_shape[1] = x_shape[1]
                y_shape[2] = (x_shape[2] - d_ksizes) // d_strides + 1
                y_shape[3] = (x_shape[3] - h_ksizes) // h_strides + 1
                y_shape[4] = (x_shape[4] - w_ksizes) // w_strides + 1
            else:
                y_shape[0] = x_shape[0]
                y_shape[1] = (x_shape[1] - d_ksizes) // d_strides + 1
                y_shape[2] = (x_shape[2] - h_ksizes) // h_strides + 1
                y_shape[3] = (x_shape[3] - w_ksizes) // w_strides + 1
                y_shape[4] = x_shape[4]
            y_shape = [y_shape[0], y_shape[1], y_shape[2], y_shape[3], y_shape[4]]
            y_shapes.append(y_shape)
            y_dtypes.append(x_dtype)

        data_format_strategy = ValueStrategy()
        data_format_strategy.append(data_formats)

        k_strategy = ValueStrategy()
        k_strategy.append(k_shapes)

        s_strategy = ValueStrategy()
        s_strategy.append(s_shapes)

        y_strategy = TensorStrategy()
        if y_shapes is not None:
            y_strategy.format(data_formats).shape(y_shapes).dtype(y_dtypes)
            y_strategy.set_mode('zip')

        return [k_strategy, s_strategy, data_format_strategy, y_strategy]

    def get_io_strategys(self, *assemble_strategy):
        pads = [0, 0, 0, 0, 0, 0]
        ceil_mode = False
        count_include_pad = True
        divisor_override = False
        [x, ksize, strides, data_format, y] = assemble_strategy
        return [x, y, ksize, strides, pads, ceil_mode, count_include_pad, divisor_override, data_format]

