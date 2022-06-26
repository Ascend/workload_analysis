from framework.op_base import OpBase


class AddOp(OpBase):
    def __init__(self, device, x, y, z):
        super().__init__(device, "Add")
        self.input("x", x).input("y", y).output("z", z)
