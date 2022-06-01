import os
import yaml
from pprint import pformat


class Config:
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


def parse_yaml(yaml_path):
    """
    Parse the yaml config file
    """
    with open(yaml_path, 'r') as fin:
        cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
        cfgs = [x for x in cfgs]
    return cfgs[0]


def get_config():
    """
    Get Config obj according to the yaml file
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "./default_config.yaml")

    default = parse_yaml(config_path)

    return Config(default)


config = get_config()
