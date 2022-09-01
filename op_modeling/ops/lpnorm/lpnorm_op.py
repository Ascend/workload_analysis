import random
from framework.op_base import OpBase
from framework.tensor_strategy import TensorStrategy, ValueStrategy
from template.generator.random_shape_generator import RandomShapeValueGenerator


class LpNormOp(OpBase):
    def __init__(self, device, input, output, axes, p, keep_dims, epsilon):
        super().__init__(device, "LpNorm")
        self.input("input", input) \
            .output("output", output) \
            .attr("axes", axes) \
            .attr("p", p) \
            .attr("keep_dims", keep_dims) \
            .attr("epsilon", epsilon) \


class LpNormIOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)
        random.seed(seed)
        input_strategy = self.get_shape_strategy(mode='zip')

        size = len(input_strategy.shapes)
        p_strategy = self.get_value_strategy(0, 2, size=size, rand_func=random.randint)
        keep_dims_strategy = self.get_value_strategy([False, True], size=size, rand_func=random.choice)


        [axes_strategy, epsilon_strategy, output_strategy] = self.gen_strategies(input_strategy, keep_dims_strategy)
        self.strategys = [
            input_strategy,
            output_strategy,
            axes_strategy,
            p_strategy,
            keep_dims_strategy,
            epsilon_strategy
        ]

    def gen_strategies(self, input_strategy, keep_dims_strategy):
        input_shapes = input_strategy.shapes
        input_dtypes = input_strategy.dtypes
        keep_dims = keep_dims_strategy.get()

        axes_lists = []
        output_shapes = []
        output_dtypes = []
        epsilon_lists = []
        for input_shape, input_dtype, keep_dim in zip(input_shapes, input_dtypes, keep_dims):
            input_rank = len(input_shape)
            # get axes
            axes_rank = random.randint(1, input_rank)
            axes_list = []
            for _ in range(axes_rank):
                axes_list.append(random.randint(0, input_rank - 1))
            axes_list = list(sorted(set(axes_list)))
            axes_lists.append(axes_list)
            #get output
            output_shape = []
            output_dtype = input_dtype
            for i in range(input_rank):
                if i in axes_list:
                    if keep_dim:
                        output_shape.append(1)
                else:
                    output_shape.append(input_shape[i])
            epsilon_lists.append(1e-12)
            output_shapes.append(output_shape)
            output_dtypes.append(output_dtype)
        axes_strategy = ValueStrategy()
        axes_strategy.append(axes_lists)
        epsilon_strategy = ValueStrategy()
        epsilon_strategy.append(epsilon_lists)

        formats = ['ND'] * len(output_shapes)
        output_strategy = TensorStrategy()
        if output_shapes is not None:
            output_strategy.format(formats).shape(output_shapes).dtype(output_dtypes)
            output_strategy.set_mode('zip')
        return [axes_strategy, epsilon_strategy, output_strategy]


    def get_io_strategys(self, *assemble_strategy):
        return assemble_strategy
