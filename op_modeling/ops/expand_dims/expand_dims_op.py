import random
from framework.op_base import OpBase
from framework.tensor_strategy import TensorStrategy, ValueStrategy
from template.generator.random_shape_generator import RandomShapeValueGenerator


class ExpandDimsOp(OpBase):
    def __init__(self, device, x, axis, y):
        super().__init__(device, "ExpandDims")
        self.input("x", x) \
            .input("axis", axis) \
            .output("y", y)


class ExpandDimsIOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)

        random.seed(seed)
        x_strategy = self.get_shape_strategy(mode='zip')

        [axis_values_strategy, axis_tensor_strategy, y_strategy] = self.gen_strategies(x_strategy)

        self.strategys = [
            x_strategy,
            axis_values_strategy,
            axis_tensor_strategy,
            y_strategy,
        ]

    def gen_strategies(self, x_strategy):
        x_shapes = x_strategy.shapes
        x_dtypes = x_strategy.dtypes

        axis_tensor_shapes = []
        axis_tensor_dtypes = []
        axis_value_lists = []

        y_shapes = []
        y_dtypes = []

        for x_shape, x_dtype in zip(x_shapes, x_dtypes):
            axis_value_list = random.randint(0, len(x_shape))  # 随机生成0到len(xshape)的数 插入axis数组
            axis_value_lists.append(axis_value_list)

            axis_tensor_shape = [1]

            axis_tensor_shapes.append(axis_tensor_shape)
            axis_tensor_dtypes.append('int32')

            if x_dtype == 'int8':
                y_dtype = 'int32'
            else:
                y_dtype = x_dtype

            y_shape = x_shape[0:axis_value_list] + [1] + x_shape[axis_value_list:]  # 生成y的shape 这里是list操作
            y_shapes.append(y_shape)
            y_dtypes.append(y_dtype)

        # Value类型
        axis_value_strategy = ValueStrategy()
        axis_value_strategy.append(axis_value_lists)
        # tensor类型
        formats = ['ND'] * len(y_shapes)
        axis_tensor_strategy = TensorStrategy()
        axis_tensor_strategy.format(formats).shape(axis_tensor_shapes).dtype(axis_tensor_dtypes)
        axis_tensor_strategy.set_host()
        axis_tensor_strategy.set_mode('zip')

        y_strategy = TensorStrategy()
        if y_shapes is not None:
            y_strategy.format(formats).shape(y_shapes).dtype(y_dtypes)
            y_strategy.set_mode('zip')

        return [axis_value_strategy, axis_tensor_strategy, y_strategy]

    def get_io_strategys(self, *assemble_strategy):
        [x, axis_value, axis_tensor, y] = assemble_strategy
        # 整合value进tensor
        axis_dict = {"value": 0}
        axis_dict["value"] = axis_value
        axis_tensor.update(axis_dict)
        return [x, axis_tensor, y]
