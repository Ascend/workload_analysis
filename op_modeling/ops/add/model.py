from framework.model_save import OperateModelBase


class AddOpModel(OperateModelBase):
    def __init__(self):
        super().__init__()

    def param_check(self: any, inputs: list, outputs: list, attr: dict) -> tuple:
        return self.SUCCESS, ""

    def param_adapter(self, inputs, outputs, attr):
        return inputs, outputs, attr

    def generate_key(self, inputs, outputs, attr):
        return inputs[0]['dtype']
