from framework.op_base import OpBase
from template.generator.random_shape_generator import RandomShapeValueGenerator
import random
from copy import deepcopy


class L2LossOp(OpBase):
    def __init__(self, device, x, y):
        super().__init__(device, "L2Loss")
        self.input("x", x).output("y", y)

class L2LossIOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)

        random.seed(seed)
        self.format = ['ND']
        x_strategy = self.get_shape_strategy(mode='product')
        y_strategy = deepcopy(x_strategy)
        y_strategy.shapes = [[1]] * len(x_strategy.shapes)

        self.strategys = [
            x_strategy,
            y_strategy
        ]

    def get_io_strategys(self, *assemble_strategy):
        [x, y] = assemble_strategy
        return [x, y]
