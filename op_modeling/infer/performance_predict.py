from infer.model_manager import ModelManager


class PerformancePredict:
    """
    对外接口
    """

    def __init__(self, env):
        soc_version = env.get("soc_version", None)
        if soc_version is None:
            raise Exception("No soc_version set")
        self.model_manager = ModelManager(soc_version)

    def op_performance_predict(self: any, op_type: str, inputs: list, outputs: list, attrs: dict) -> tuple:
        """
        预测算子性能
        :param op_type: 算子类型
        :param inputs: 输入tensor，格式为[{"format":xx, "dtype":xx, "shape":xx, "origin_format":xx,...},{...}]
        :param outputs: 输出tensor，格式为[{"format":xx, "dtype":xx, "shape":xx, "origin_format":xx,...},{...}]
        :param attrs: 算子属性，格式为{”origin_ops":[{},{},...], "attr1":xx, "attr2":xx,...}
        :return: ret_code, msg, {"predict_time":xxx}
        """
        return self.model_manager.predict_single_op_perf(op_type, inputs, outputs, attrs)