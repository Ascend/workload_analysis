from framework.op_base import OpBase
from template.generator.random_shape_generator import RandomShapeSingleInOutGenerator


class Relu6Op(OpBase):
    def __init__(self, device, x, y):
        super().__init__(device, "Relu6")
        self.input("x", x).output("y", y)


class Relu6IOGenerator(RandomShapeSingleInOutGenerator):

    def get_io_strategys(self, *assemble_strategy):
        x, = assemble_strategy
        y = x
        return [x, y]
