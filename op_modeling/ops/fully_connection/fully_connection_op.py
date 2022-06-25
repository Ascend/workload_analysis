import random
from framework.op_base import OpBase
from framework.tensor_strategy import TensorStrategy
from template.generator.random_shape_generator import RandomShapeValueGenerator


class FullyConnectionOp(OpBase):
    def __init__(self, device, x, w, b, offset_w, y, num_output, transpose, axis, offset_x):
        super().__init__(device, "FullyConnection")
        self.input("x", x) \
            .input("w", w) \
            .output("y", y) \
            .attr("num_output", num_output) \
            .attr("transpose", transpose) \
            .attr("axis", axis) \
            .attr("offset_x", offset_x)
        if b is not None:
            self.input("b", b)
        if offset_w is not None:
            self.input("offset_w", offset_w)


class FullyConnectionIOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)

        random.seed(seed)
        x_strategy = self.get_shape_strategy(mode='zip')
        size = len(x_strategy.shapes)
        axis_strategy = self.get_value_strategy(1, 2, size=size, rand_func=random.randint)
        transpose_strategy = self.get_value_strategy([False, True], size=size, rand_func=random.choice)
        [y_strategy, w_strategy, b_strategy] = self.gen_strategies(x_strategy, axis_strategy, transpose_strategy)
        self.strategys = [
            x_strategy,
            w_strategy,
            b_strategy,
            y_strategy,
            transpose_strategy,
            axis_strategy,
        ]

    def gen_strategies(self, x_strategy, axis_strategy, transpose_strategy):
        x_shapes = x_strategy.shapes
        x_dtypes = x_strategy.dtypes
        axes = axis_strategy.get()
        transposes = transpose_strategy.get()
        y_shapes = []
        y_dtypes = []
        w_shapes = []
        b_shapes = []
        for x_shape, x_dtype, axis, transpose in zip(x_shapes, x_dtypes, axes, transposes):
            # 此处算子在实现上存在限制，输入的dtype为int8则输出的dtype为int32
            if x_dtype == 'int8':
                y_dtype = 'int32'
            else:
                y_dtype = random.choice(['float', 'float16'])
            y_shape = x_shape[:axis] + [random.randint(16, 2048)]
            while y_shape[axis] * self.get_size(x_shape[axis:]) * 2 > self.size_of_2gb:
                y_shape[axis] = max(int(y_shape[-1] / 2), 1)
            y_shapes.append(y_shape)
            y_dtypes.append(y_dtype)

            w_shape = [y_shape[axis], self.get_size(x_shape[axis:])]

            if transpose:
                w_shape = [self.get_size(x_shape[axis:]), y_shape[axis]]
            w_shapes.append(w_shape)
            b_shapes.append([y_shape[axis]])
        formats = ['ND'] * len(y_shapes)
        y_strategy = TensorStrategy()
        if y_shapes is not None:
            y_strategy.format(formats).shape(y_shapes).dtype(y_dtypes)
            y_strategy.set_mode('zip')

        w_strategy = TensorStrategy()
        if w_shapes is not None:
            w_strategy.format(formats).shape(w_shapes).dtype(x_dtypes)
            w_strategy.set_mode('zip')

        b_strategy = TensorStrategy()
        if b_shapes is not None:
            b_strategy.format(formats).shape(b_shapes).dtype(y_dtypes)
            b_strategy.set_mode('zip')
        return [y_strategy, w_strategy, b_strategy]

    def get_io_strategys(self, *assemble_strategy):
        [x, w, b, y, transpose, axis] = assemble_strategy
        # offset_w must be None
        offset_w = None
        num_output = 1
        offset_x = 0
        return [x, w, b, offset_w, y, num_output, transpose, axis, offset_x]
