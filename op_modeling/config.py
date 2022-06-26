import logging
import configparser
import os
import subprocess
import json
from pprint import pformat


class BaseContainer:
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [x for x in v])
            else:
                setattr(self, k, v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()

    def add_attr(self, k, v):
        setattr(self, k, v)


class Config(BaseContainer):
    def __init__(self, config_path=""):
        """
        获取配置文件的信息作为默认配置
        :param config_path: 配置文件路径
        """
        conf = configparser.ConfigParser()
        conf.read(config_path, encoding="utf-8")
        default = conf.items('default')
        configs = {}
        for item in default:
            configs[item[0]] = json.loads(item[1])
        super().__init__(configs)


class Env(BaseContainer):
    def __init__(self):
        """
        获取当前环境信息
        """
        envs = {}
        soc_version = self.read_soc_version()
        if len(soc_version) == 0:
            logging.warning("Get soc_version error")
        envs['soc_version'] = soc_version
        super().__init__(envs)

    @classmethod
    def grep_first_line(cls, input_, pattern):
        """
        从多行的字符串中获取匹配pattern的第一行
        """
        start_index = input_.find(pattern)
        end_index = input_.find('\n', start_index)
        line = input_[start_index: end_index]
        return line

    @classmethod
    def read_soc_version(cls):
        def convert_soc_version(name):
            soc_version_table = {
                "310": "Ascend310",
                "710": "Ascend310P",
                "910A": "Ascend910",
                "310P3": "Ascend310P"
            }
            return soc_version_table.get(name, None)

        # 获取NPU ID
        statement = ['npu-smi', 'info', '-l']
        try:
            output = subprocess.check_output(statement, shell=False).decode(encoding='utf-8')
        except Exception as e:
            logging.error(f"Exec cmd error: {e}")
            return ""
        npu_id_line = cls.grep_first_line(output, "NPU ID")
        npu_id = npu_id_line.split(":")[1].lstrip()

        # 读取第一个NPU的第0个芯片的信息，获取芯片名
        statement = ["npu-smi", "info", "-t", "board", "-i", npu_id, "-c", "0"]
        try:
            output = subprocess.check_output(statement, shell=False).decode(encoding='utf-8')
        except Exception as e:
            logging.error(f"Exec cmd error: {e}")
            return ""
        chip_name_line = cls.grep_first_line(output, "Chip Name")
        chip_name = chip_name_line.split(":")[1].lstrip()

        # 转换chip name为soc_version
        soc_version = convert_soc_version(chip_name)
        if soc_version is None:
            logging.error(f"unsupported chip type: {chip_name}")
            return ""
        return soc_version


config = Config(os.path.join(os.path.dirname(__file__), "config.ini"))
env = Env()
