import random
from framework.op_base import OpBase
from framework.tensor_strategy import TensorStrategy, ValueStrategy
from template.generator.random_shape_generator import RandomShapeValueGenerator
from functools import reduce



class TopKOp(OpBase):
    # 初始化
    def __init__(self, device, x, k, values, indices, sorted, largest, dim):
        super().__init__(device, "TopK")
        self.input("x", x) \
            .input("k", k) \
            .output("values", values) \
            .output("indices", indices) \
            .attr("sorted", sorted) \
            .attr("largest", largest) \
            .attr("dim", dim)


class TopKIOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)

        random.seed(seed)
        # 根据TopK算子输入特性，重写生成策略
        x_strategy = self.get_shape_strategy(mode='zip')
        # 根据输入的特征，依据约束条件生成其他策略
        [k_values_strategy, k_tensor_strategy, values_strategy, indices_strategy] = self.gen_strategies(x_strategy)
        self.strategys = [
            x_strategy,
            k_values_strategy,
            k_tensor_strategy,
            values_strategy,
            indices_strategy,
        ]

    def get_shape_strategy(self, mode='product'):
        x_shapes = []
        x_formats = []
        x_dtypes = []
        for i in range(self.n_sample):
            for x_dtype in self.dtype:
                dim = random.randint(1, 5)
                shape = [random.randint(2, 2048) for _ in range(dim)]
                if self.get_size(shape) > 32768 and shape[dim-1] <= 16:
                    shape[dim-1] = random.randint(16, 2048)
                while self.get_size(shape) * 2 > self.size_of_half_gb:
                    index = random.randint(0, len(shape) - 1)
                    if index == len(shape) - 1 and self.get_size(shape) > 32768:
                        shape[index] = max(int(shape[index] / 2), 16)
                    else:
                        shape[index] = max(int(shape[index] / 2), 1)
                x_shapes.append(shape)
                x_formats.append("ND")
                x_dtypes.append(x_dtype)
        x_strategy = TensorStrategy()
        if x_shapes is not None:
            x_strategy.set_mode(mode)
            x_strategy.format(x_formats).shape(x_shapes).dtype(x_dtypes)
        return x_strategy

    def gen_strategies(self, x_strategy):
        x_shapes = x_strategy.shapes
        x_dtypes = x_strategy.dtypes

        values_shapes = []
        values_dtypes = []
        indices_shapes = []
        indices_dtypes = []

        k_tensor_shapes = []
        k_tensor_dtypes = []
        k_value_lists = []
        for x_shape, x_dtype in zip(x_shapes, x_dtypes):
            # 获取最后的维度，以便获取k值的上限
            last_dim=x_shape[len(x_shape)-1]
            size=reduce(lambda x, y: x*y, x_shape)
            # 约束条件，生成k值
            if last_dim >= 16 and size >32768:
                k_value_list=random.randint(16, last_dim)
            else:
                k_value_list=random.randint(1, last_dim)
            k_value_lists.append(k_value_list)
            # k的tensor类型
            k_tensor_shape = [1]
            k_tensor_shapes.append(k_tensor_shape)
            k_tensor_dtypes.append('int32')

            values_shape = x_shape[:len(x_shape)-1] + [k_value_list]
            values_shapes.append(values_shape)
            values_dtypes.append(x_dtype)

            indices_shape = x_shape[:len(x_shape)-1] + [k_value_list]
            indices_shapes.append(indices_shape)
            indices_dtypes.append('int32')
        # k值Value类型
        k_value_strategy = ValueStrategy()
        k_value_strategy.append(k_value_lists)

        formats = ['ND'] * len(values_shapes)
        # k的tensor类型
        k_tensor_strategy = TensorStrategy()
        k_tensor_strategy.format(formats).shape(k_tensor_shapes).dtype(k_tensor_dtypes)
        # k的is_host全部设置为True
        k_tensor_strategy.set_host()
        k_tensor_strategy.set_mode('zip')

        values_strategy = TensorStrategy()
        if values_shapes is not None:
            values_strategy.format(formats).shape(values_shapes).dtype(values_dtypes)
            values_strategy.set_mode('zip')

        indices_strategy = TensorStrategy()
        if indices_shapes is not None:
            indices_strategy.format(formats).shape(indices_shapes).dtype(indices_dtypes)
            indices_strategy.set_mode('zip')


        return [k_value_strategy, k_tensor_strategy, values_strategy, indices_strategy]


    def get_io_strategys(self, *assemble_strategy):
        # attr只能采用默认值
        dim=-1
        sorted=True
        largest=True
        [x, k_value, k_tensor, values, indices] = assemble_strategy
        # 创建字典并更新
        k_dict={"value":k_value}
        k_tensor.update(k_dict)
        return [x, k_tensor, values, indices, sorted, largest, dim]