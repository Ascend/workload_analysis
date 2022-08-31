import random
from framework.op_base import OpBase
from framework.tensor_strategy import TensorStrategy
from template.generator.random_shape_generator import RandomShapeValueGenerator

class BinaryCrossEntropyOp(OpBase):
    def __init__(self, device, x, y, weight, reduction, output):
        super().__init__(device, "BinaryCrossEntropy")
        self.input("x", x) \
            .input("y", y) \
            .attr("reduction", reduction) \
            .output("output", output)
        if weight is not None:
            self.input("weight", weight)

class BinaryCrossEntropyIOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)

        random.seed(seed)
        x_strategy = self.get_shape_strategy(mode='zip')
        size = len(x_strategy.shapes)
        reduction_strategy = self.get_value_strategy(["none", "mean", "sum"], size=size, rand_func=random.choice)
        self.strategys = [
            x_strategy,
            x_strategy,
            x_strategy,
            reduction_strategy,
            self.gen_output_strategies(x_strategy, reduction_strategy)
        ]

    def gen_output_strategies(self, x_strategy, reduction_strategy):
        x_shapes = x_strategy.shapes
        x_dtypes = x_strategy.dtypes
        reductions = reduction_strategy.get()
        output_shapes = []
        output_dtypes = []
        for x_shape, x_dtype, reduction in zip(x_shapes, x_dtypes, reductions):
            if reduction == "none":
                output_shape = x_shape
                output_shapes.append(output_shape)
            elif reduction == "mean" or reduction == "sum":
                output_shapes.append([1])
            output_dtypes.append(x_dtype)
        formats = ['ND'] * len(output_shapes)
        output_strategy = TensorStrategy()
        if output_shapes is not None:
            output_strategy.format(formats).shape(output_shapes).dtype(output_dtypes)
            output_strategy.set_mode('zip')
        return output_strategy

    def get_shape_strategy(self, mode='product'):
        x_shapes = []
        x_formats = []
        x_dtypes = []
        for i in range(self.n_sample):
            for x_dtype in self.dtype:
                dim = random.randint(1, 2)
                shape = [random.randint(2, 2048) for _ in range(dim)]
                while self.get_size(shape) * 2 > self.size_of_half_gb:
                    index = random.randint(0, len(shape) - 1)
                    shape[index] = max(int(shape[index] / 2), 1)
                x_shapes.append(shape)
                x_formats.append("ND")
                x_dtypes.append(x_dtype)
        x_strategy = TensorStrategy()
        if x_shapes is not None:
            x_strategy.set_mode(mode)
            x_strategy.format(x_formats).shape(x_shapes).dtype(x_dtypes)
        return x_strategy

    def get_io_strategys(self, *assemble_strategy):
        if random.random() < 0.5:
            return assemble_strategy
        [x, y, weight, reduction, output] = assemble_strategy
        return [x, y, None, reduction, output]