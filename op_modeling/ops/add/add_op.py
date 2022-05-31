from workload.framework.op_base import OpBase
from workload.framework.op_register import RegisterOfOp


@RegisterOfOp('Add')
class AddOp(OpBase):
    """
    Sub 算子
    """

    def __init__(self, device, x, y, z):
        super().__init__(device, "Add")
        self.input("x", x).input("y", y).output("z", z)
