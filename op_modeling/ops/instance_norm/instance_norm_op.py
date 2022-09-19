import random
from framework.op_base import OpBase
from framework.tensor_strategy import TensorStrategy, ValueStrategy
from template.generator.random_shape_generator import RandomShapeValueGenerator


class InstanceNormOp(OpBase):
    def __init__(self, device, x, gamma, beta, y, mean, variance, data_format, epsilon):
        super().__init__(device, "InstanceNorm")
        self.input("x", x) \
            .input("gamma", gamma) \
            .input("beta", beta) \
            .output("y", y) \
            .output("mean", mean) \
            .output("variance", variance) \
            .attr("data_format", data_format) \
            .attr("epsilon", epsilon) \



class InstanceNormIOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)
        random.seed(123)
        x_strategy = self.get_shape_strategy(mode='zip')
        size = len(x_strategy.shapes)

        data_format_strategy = self.get_value_strategy(["ND"], size=size, rand_func=random.choice)

        [gamma_strategy, y_strategy, beta_strategy, epsilon_strategy, mean_strategy, 
        variance_strategy] = self.gen_strategies(x_strategy, data_format_strategy)
        self.strategys = [
            x_strategy,
            gamma_strategy,
            beta_strategy,
            y_strategy,
            mean_strategy,
            variance_strategy,
            data_format_strategy,
            epsilon_strategy
        ]

    def gen_strategies(self, input_strategy, data_format_strategy):
        input_shapes = input_strategy.shapes
        input_dtypes = input_strategy.dtypes
        gamma_shapes = []
        mean_shapes = []
        epsilon_lists = []
        for input_shape, input_dtype in zip(input_shapes, input_dtypes):
            input_rank = len(input_shape)
            gamma_shapes.append([input_shape[1]])
            epsilon_lists.append(1e-6)

        epsilon_strategy = ValueStrategy()
        epsilon_strategy.append(epsilon_lists)

        formats = data_format_strategy.get()
        y_strategy = TensorStrategy()
        gamma_strategy = TensorStrategy()
        beta_strategy = TensorStrategy()
        mean_strategy = TensorStrategy()
        variance_strategy = TensorStrategy()

        y_strategy.format(formats).shape(input_shapes).dtype(input_dtypes)
        y_strategy.set_mode('zip')
        gamma_strategy.format(formats).shape(gamma_shapes).dtype(input_dtypes)
        gamma_strategy.set_mode('zip')
        beta_strategy.format(formats).shape(gamma_shapes).dtype(input_dtypes)
        beta_strategy.set_mode('zip')
        mean_strategy.format(formats).shape(input_shapes).dtype(input_dtypes)
        mean_strategy.set_mode('zip')
        variance_strategy.format(formats).shape(input_shapes).dtype(input_dtypes)
        variance_strategy.set_mode('zip')
        return [gamma_strategy, y_strategy, beta_strategy, epsilon_strategy, mean_strategy, variance_strategy]


    def get_io_strategys(self, *assemble_strategy):
        [x, gamma, beta, y, mean, variance, data_format, epsilon] = assemble_strategy
        return [x, gamma, beta, y, mean, variance, data_format, epsilon]
