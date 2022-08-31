import random

from template.generator.random_shape_generator import RandomShapeValueGenerator
from framework.tensor_strategy import TensorStrategy, ValueStrategy
from framework.op_base import OpBase
import numpy as np
import acl
import acl_ext
class IouOp(OpBase):
    def __init__(self, device, bboxes, gtboxes, overlap, mode, eps):
        super(IouOp, self).__init__(device, "Iou")
        self.input("bboxes", bboxes) \
            .input("gtboxes", gtboxes) \
            .output("overlap", overlap) \
            .attr("mode", mode) \
            .attr("eps", eps)

class IouIOGenerator(RandomShapeValueGenerator):
    def __init__(self, sample_dtype, sample_format, n_sample=0, seed=0):
        super().__init__(sample_dtype, sample_format, n_sample)
        random.seed(seed)
        bboxes_strategy = self.get_shape_strategy(mode='zip')
        gtboxes_strategy = self.get_shape_strategy(mode='zip')
        overlap_strategy = self.gen_strategies(bboxes_strategy, gtboxes_strategy)
        size = len(bboxes_strategy.shapes)
        mode_strategy = self.get_value_strategy(["iou", "iof"], size=size, rand_func=random.choice)
        eps_strategy = self.get_value_strategy(size=size, rand_func=random.random)


        self.strategys = [
            bboxes_strategy,
            gtboxes_strategy,
            overlap_strategy,
            mode_strategy,
            eps_strategy,
        ]

    def get_shape_strategy(self, mode='product'):
        x_shapes = []
        x_formats = []
        x_dtypes = []
        for i in range(self.n_sample):
            for x_dtype in self.dtype:
                x_format = self.format[0] # 只能float16 和 ND
                dim = 4
                shape = [random.randint(1, 100)] + [dim]
                while self.get_size(shape) * 2 > self.size_of_half_gb:
                    index = random.randint(0, len(shape) - 1)
                    shape[index] = max(int(shape[index] / 2), 1)
                x_shapes.append(shape)
                x_formats.append(x_format)
                x_dtypes.append(x_dtype)
        x_strategy = TensorStrategy()
        if x_shapes is not None:
            x_strategy.set_mode(mode)
            x_strategy.format(x_formats).shape(x_shapes).dtype(x_dtypes)
        return x_strategy

    def gen_strategies(self, x_strategy, y_strategy):
        dtypes = x_strategy.dtypes
        x_shapes = x_strategy.shapes
        y_shapes = y_strategy.shapes

        overlap_shapes = []
        overlap_dtypes = []

        for dtype, x_shape, y_shape in zip(dtypes, x_shapes, y_shapes):
            overlap_dtype = dtype
            overlap_shape = [y_shape[0]] + [x_shape[0]]

            overlap_shapes.append(overlap_shape)
            overlap_dtypes.append(overlap_dtype)

        formats = ["ND"] * len(overlap_shapes)
        overlap_strategy = TensorStrategy()
        if overlap_shapes is not None:
            overlap_strategy.format(formats).shape(overlap_shapes).dtype(overlap_dtypes)
            overlap_strategy.set_mode('zip')

        return overlap_strategy

    def get_io_strategys(self, *assemble_strategy):
        return assemble_strategy

            
