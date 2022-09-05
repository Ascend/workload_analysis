import random
from framework.op_base import OpBase
from framework.tensor_strategy import TensorStrategy, ValueStrategy
from template.generator.random_shape_generator import RandomShapeValueGenerator

class PadV3Op(OpBase):
    def __init__(self, device, x, paddings, constant_values, y, mode, paddings_contiguous):
        super().__init__(device, "PadV3")
        self.input("x", x) \
            .input("paddings", paddings) \
            .attr("constant_values", constant_values) \
            .output("y", y) \
            .attr("mode", mode) \
            .attr("paddings_contiguous", paddings_contiguous)

class PadV3IOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)
        random.seed(seed)
        x_strategy = self.get_shape_strategy(mode='zip')
        [paddings_value_strategy, paddings_tensor_strategy, y_strategy] = self.gen_strategies(x_strategy)
        self.strategys = [
            x_strategy,
            paddings_value_strategy,
            paddings_tensor_strategy,
            y_strategy,
        ]
    def get_size(self, shape):
        ans = 1
        for i in range(len(shape)):
            ans = shape[i]*ans
        return ans
    def gen_strategies(self, x_strategy):
        x_shapes = x_strategy.shapes
        x_dtypes = x_strategy.dtypes
        y_shapes = []
        y_dtypes = []
        paddings_shapes = []
        paddings_dtypes = []
        pd_value_list=[]
        for x_shape, x_dtype in zip(x_shapes, x_dtypes):
            x_rank = len(x_shape)  # 计算x的秩
            paddings_shapes.append([x_rank, 2])
            paddings_dtype = random.choice(['int32', 'int64'])
            paddings_dtypes.append(paddings_dtype)
            pd_tmp = []
            shape = []
            tmp = 0
            # 计算在每个维度的 前面和后面 能扩充的行数
            n_value = max(1, (100 - x_shape[0])//2)
            c_value = max(1, (64 - x_shape[1])//2)
            h_value = max(1, (2048 - x_shape[2])//2)
            w_value = max(1, (2048 - x_shape[3])//2)
            # 随机生成在每个维度上扩充的行数
            pd_tmp.append([random.randint(1, n_value), random.randint(1, n_value)])
            pd_tmp.append([random.randint(1, c_value), random.randint(1, c_value)])
            pd_tmp.append([random.randint(1, h_value), random.randint(1, h_value)])
            pd_tmp.append([random.randint(1, w_value), random.randint(1, w_value)])
            for i in range(x_rank):
                shape.append(x_shape[i] + pd_tmp[i][0] + pd_tmp[i][1])
            # 判断维度是否超过限制，超过则减半
            while self.get_size(shape) * 2 > self.size_of_half_gb:
                index = tmp%4
                tmp = tmp + 1
                pd_tmp[index][0] = max(int(pd_tmp[index][0] / 2), 0)
                pd_tmp[index][1] = max(int(pd_tmp[index][1] / 2), 0)
                shape = []
                for i in range(x_rank):
                    shape.append(x_shape[i] + pd_tmp[i][0] + pd_tmp[i][1])
            pd_value_list.append(pd_tmp)
            y_shape = []
            for i in range(x_rank):
                y_shape.append(x_shape[i] + pd_tmp[i][0] + pd_tmp[i][1])
            y_shapes.append(y_shape)
            y_dtype = x_dtype
            y_dtypes.append(y_dtype)
        pd_value_strategy = ValueStrategy()
        pd_value_strategy.append(pd_value_list)
        formats = ['ND'] * len(x_shapes)
        pd_tensor_strategy = TensorStrategy()
        if paddings_shapes is not None:
            pd_tensor_strategy.format(formats).shape(paddings_shapes).dtype(paddings_dtypes)
            pd_tensor_strategy.set_mode('zip')
        pd_tensor_strategy.set_host()
        y_strategy = TensorStrategy()
        if y_shapes is not None:
            y_strategy.format(formats).shape(y_shapes).dtype(y_dtypes)
            y_strategy.set_mode('zip')
        return [pd_value_strategy, pd_tensor_strategy, y_strategy]

    def get_io_strategys(self, *assemble_strategy):
        mode = 'constant'
        paddings_contiguous = 'true'
        [x, paddings_v, paddings, y] = assemble_strategy
        constant_values = 0
        p_dict = {"value":0}
        p_dict["value"] = paddings_v
        paddings.update(p_dict)
        return [x, paddings, constant_values, y, mode, paddings_contiguous]