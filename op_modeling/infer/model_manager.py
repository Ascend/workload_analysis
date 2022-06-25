import os

from framework.model_packing import deserialization


class ModelManager:
    """
    该类作为最终交付算子模型的管理类，主要功能为
    1. 提供外部调用单算子、融合算子模型能力
    """
    MODEL_PACK_DIR = 'model'
    SUCCESS = 0
    UN_SUCCESS = -1

    def __init__(self, soc_version):
        self.soc_version = soc_version

    @classmethod
    def check_env(cls: any, env: dict) -> tuple:
        soc_version = env.get("soc_version", None)
        if soc_version is None:
            return cls.UN_SUCCESS, "No soc_version found in env"
        return cls.SUCCESS, ""

    @classmethod
    def get_pack_dir(cls: any) -> str:
        return os.path.join(cls.get_project_dir(), cls.MODEL_PACK_DIR)

    @classmethod
    def get_pack_model_path(cls, pack_dir, soc_version, name):
        return os.path.join(pack_dir, name + "_" + soc_version + '.pkl')

    @classmethod
    def get_project_dir(cls: any) -> str:
        infer_dir = os.path.dirname(__file__)
        return os.path.dirname(infer_dir)

    def predict_single_op_perf(self: any, op_type: str, inputs: list, outputs: list,
                               attrs: dict) -> tuple:
        pack_dir = self.get_pack_dir()
        ret, msg = self._check_pack_model(pack_dir, op_type)
        if ret != self.SUCCESS:
            return ret, msg, {}

        model_path = self.get_pack_model_path(pack_dir, self.soc_version, op_type)
        model = deserialization(model_path)
        ret_code, msg, predict_time = model.predict(inputs, outputs, attrs)
        return ret_code, msg, {'predict_time': predict_time}

    def _check_pack_model(self, pack_dir, name):
        if not os.path.exists(pack_dir):
            return self.UN_SUCCESS, "No dir: {} found".format(pack_dir)

        if self.soc_version is None:
            return self.UN_SUCCESS, "Please set soc_version first"
        model_path = self.get_pack_model_path(pack_dir, self.soc_version, name)
        if not os.path.exists(model_path):
            return self.UN_SUCCESS, "No model: {} found for: {}".format(model_path, name)
        return self.SUCCESS, ""
