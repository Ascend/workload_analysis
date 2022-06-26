#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os


class RegisterOf:
    """
    用于外部注册算子Builder的方法
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, target):
        self.set(target, self.name)
        BuilderManager.register(self.name, target)

    @staticmethod
    def set(target, name):
        target.op_type = name


class BuilderManager:
    """
    Builder管理器
    """

    class OpRegister:
        """
        该类提供单算子的信息与行为的能力关联
        """

        def __init__(self):
            self._dicts = {}

        def __setitem__(self, key, value):
            if not callable(value):
                raise Exception(f"value must be callabe. now: {value}")
            if key is None:
                key = value.__name__
            self._dicts[key] = value

        def __getitem__(self, key):
            if key not in self._dicts.keys():
                raise Exception(f"not find {key}")
            return self._dicts.get(key, None)

        def keys(self):
            return self._dicts.keys()

        def register(self, key, target):
            self[key] = target
    _register = OpRegister()

    def __init__(self):
        raise Exception("not support register.")

    @classmethod
    def register(cls, key, target):
        cls._register.register(key, target)

    @classmethod
    def keys(cls):
        return cls._register.keys()

    @classmethod
    def get(cls, op_name):
        return cls._register[op_name]


def import_all_ops():
    import importlib
    module_names = []
    op_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ops')
    for fname in os.listdir(op_dir):
        if os.path.isdir(os.path.join(op_dir, fname)) and not fname.startswith('__'):
            module_names.append(fname)
    for module_name in module_names:
        importlib.import_module(f'.{module_name}.builder', package='ops')
        importlib.import_module(f'.{module_name}.{module_name}_op', package='ops')
