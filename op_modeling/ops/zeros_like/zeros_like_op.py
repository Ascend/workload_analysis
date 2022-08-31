from framework.op_base import OpBase


class ZerosLikeOp(OpBase):
    def __init__(self, device, x, y):
        super().__init__(device, "ZerosLike")
        self.input("x", x).output("y", y)
