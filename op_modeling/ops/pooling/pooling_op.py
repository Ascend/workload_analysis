import random
import math
from framework.op_base import OpBase
from framework.tensor_strategy import TensorStrategy, ValueStrategy
from template.generator.random_shape_generator import RandomShapeValueGenerator


class PoolingOp(OpBase):
    def __init__(self, device, x, y, mode, global_pooling, window, stride, pad, dilation, ceil_mode, data_format):
        super().__init__(device, "Pooling")
        self.input("x", x) \
            .output("y", y) \
            .attr("mode", mode) \
            .attr("global_pooling", global_pooling) \
            .attr("window", window) \
            .attr("stride", stride) \
            .attr("pad", pad) \
            .attr("dilation", dilation) \
            .attr("ceil_mode", ceil_mode) \
            .attr("data_format", data_format)


class PoolingIOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)

        random.seed(seed)
        x_strategy = self.get_x_strategy(mode='zip')
        size = len(x_strategy.shapes)
        mode_strategy = self.get_value_strategy([0, 1], size=size, rand_func=random.choice)
        global_pooling_strategy = self.get_value_strategy([False, True], size=size, rand_func=random.choice)
        [window_strategy, pad_strategy, stride_strategy, \
        dilation_strategy] = self.gen_w_s_p_d_strategies(x_strategy, global_pooling_strategy)
        ceil_mode_strategy = self.get_value_strategy([0, 1], size=size, rand_func=random.choice)
        data_format_strategy = self.get_value_strategy(['NCHW'], size=size, rand_func=random.choice)
        y_strategy =  self.gen_y_strategy(x_strategy, global_pooling_strategy, window_strategy, 
        pad_strategy, stride_strategy, \
        dilation_strategy, data_format_strategy, ceil_mode_strategy)

        self.strategys = [
            x_strategy,
            y_strategy,
            mode_strategy,
            global_pooling_strategy,
            window_strategy,
            stride_strategy,
            pad_strategy,
            dilation_strategy,
            ceil_mode_strategy,
            data_format_strategy,
        ]


    def gen_y_strategy(self, x_strategy, global_pooling_strategy, window_strategy, pad_strategy, \
    stride_strategy, dilation_strategy, data_format_strategy, ceil_mode_strategy):
        x_shapes = x_strategy.shapes
        x_dtypes = x_strategy.dtypes
        windows = window_strategy.get()
        pads = pad_strategy.get()
        strides = stride_strategy.get()
        dilations = dilation_strategy.get()
        data_formats = data_format_strategy.get()
        global_pools = global_pooling_strategy.get()
        ceil_modes = ceil_mode_strategy.get()
        y_shapes = []
        y_dtypes = []
        y_formats = []
        for x_shape, x_dtype, window, pad, stride, dilation, simply_format, global_pool, \
        ceil_mode in zip(x_shapes, x_dtypes, windows, pads, strides, dilations, data_formats, global_pools, ceil_modes):
            if x_dtype == 'int8':
                y_dtype = 'int32'
            else:
                y_dtype = x_dtype

            y_format = simply_format

            if(global_pool):
                y_shape = [x_shape[0], x_shape[1], 1, 1]
            else:
                if(ceil_mode == 0):
                    y_shape = [x_shape[0], x_shape[1], math.ceil(((
                        x_shape[2]+pad[0]+pad[1]-dilation[0]*(window[0]-1)-1) / stride[0]))+1,\
                    math.ceil(((x_shape[3]+pad[2]+pad[3]-dilation[2]*(window[1]-1)-1) / stride[1]))+1]
                else:
                    y_shape = [x_shape[0], x_shape[1], int(((x_shape[2]+pad[0]+pad[1]-dilation[0]*
                        (window[0]-1)-1) / stride[0]))+1, \
                    int(((x_shape[3]+pad[2]+pad[3]-dilation[2]*(window[1]-1)-1) / stride[1]))+1]
            y_shapes.append(y_shape)
            y_dtypes.append(y_dtype)
            y_formats.append(y_format)

        y_strategy = TensorStrategy()
        y_strategy.format(y_formats).shape(y_shapes).dtype(y_dtypes)
        y_strategy.set_mode('zip')
        return  y_strategy


    def gen_w_s_p_d_strategies(self, x_strategy, global_pooling_strategy):
        x_shapes = x_strategy.shapes
        global_poolings =  global_pooling_strategy.get()
        window_strategy = ValueStrategy()
        pad_strategy = ValueStrategy()
        stride_strategy = ValueStrategy()
        dilation_strategy = ValueStrategy()
        for x_shape, gl_pool in zip(x_shapes, global_poolings):
            if(gl_pool):
                window_h = random.randint(1, 15)
                window_w = random.randint(1, 15)
                window_strategy.append([[window_h, window_w]])
                pad_strategy.append([[0, 0, 0, 0]])
                dilation_strategy.append([[1, 1, 1, 1]])
                stride_strategy.append([[1, 1]])

            else:
                h_num = min(x_shape[2], 15)
                w_num = min(x_shape[3], 15)
                window_h = random.randint(1, h_num)
                window_w = random.randint(1, w_num)
                window_strategy.append([[window_h, window_w]])

                pad_up = random.randint(0, int((window_h-1)/2))
                pad_bottom = pad_up
                pad_left = random.randint(0, int((window_w-1)/2))
                pad_right = pad_left
                pad_strategy.append([[pad_up, pad_bottom, pad_left, pad_right]])

                dilation_strategy.append([[1, 1, 1, 1]])

                stride_h = random.randint(1, window_h)
                stride_w = random.randint(1, window_w)
                stride_strategy.append([[stride_h, stride_w]])

        return [window_strategy, pad_strategy, stride_strategy, dilation_strategy]


    def get_x_strategy(self, mode='product'):
        x_shapes = []
        x_formats = []
        x_dtypes = []
        for i in range(self.n_sample):
            n_value = random.randint(1, 100)
            c_value = random.randint(3, 64)
            h_value = random.randint(1, 4096)
            w_value = random.randint(1, 4096)
            shape = [n_value, c_value, h_value, w_value]
            while self.get_size(shape) * 2 > self.size_of_half_gb:
                index = random.randint(0, len(shape) - 1)
                shape[index] = max(int(shape[index] / 2), 1)
            x_shapes.append(shape)
            x_formats.append(self.format[0])
            x_dtypes.append(self.dtype[0])
        x_strategy = TensorStrategy()
        if x_shapes is not None:
            x_strategy.set_mode(mode)
            x_strategy.format(x_formats).shape(x_shapes).dtype(x_dtypes)
        return x_strategy


    def get_io_strategys(self, *assemble_strategy):
        return assemble_strategy
