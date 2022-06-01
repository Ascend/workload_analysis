#!/usr/bin/python
import json

import acl
import numpy as np
from framework.util import check
from framework.converter import Converter


class Tensor:
    """
    该类主要提供由Tensor的基本信息生成一个完整Tensor内容的方法。
    该类的输入是具体的策略实例化信息
    """

    def __init__(self, strategy: dict = None, mode="random", is_host=False):
        self.dtype = 'float'
        self.format = 'ND'
        self.origin_format = 'ND'
        self.shape = (8, 16)
        self.max_num = 1
        self.mode = mode
        self.is_host = is_host

        self.buffer = None
        self.ptr = None
        self.desc = None
        self.value = None
        self.strategy = strategy

        if strategy:
            self.set_strategy(strategy)

    @staticmethod
    def from_strategy(shape, dformat="ND", dtype="float16", mode="random", is_host=False):
        tensor = Tensor(dict(shape=shape, format=dformat, dtype=dtype), mode=mode, is_host=is_host)
        return tensor

    @staticmethod
    def from_numpy(data, dtype="int32", is_host=True):
        tensor = Tensor(dict(shape=list(data.shape), format="ND", dtype=dtype, value=data.tolist()), mode="const",
                        is_host=is_host)
        return tensor

    def _release_source(self):
        if self.buffer:
            ret = acl.destroy_data_buffer(self.buffer)
            check(ret, "acl.destroy_data_buffer (addr:{})".format(self.buffer))
            self.buffer = None
            if not self.is_host:
                ret = acl.rt.free(self.ptr)
                check(ret, "acl.rt.free (addr:{})".format(self.ptr))
            else:
                ret = acl.rt.free_host(self.ptr)
                check(ret, "acl.rt.free_host (addr:{})".format(self.ptr))
            self.ptr = None

        if self.desc:
            acl.destroy_tensor_desc(self.desc)
            check(0, "acl.destroy_tensor_desc (addr:{})".format(self.desc))
            self.desc = None

    def force_del(self):
        self._release_source()

    def __del__(self):
        self._release_source()

    def set_strategy(self, strategy: dict):
        for key, value in strategy.items():
            self.__setattr__(key, value)

        # self.dtype = strategy.get("dtype")
        # self.format = strategy.get("format")
        # self.shape = strategy.get("shape")
        # self.max_num = strategy.get("max_num")
        # if "origin_format" in strategy.keys():
        #     self.origin_format = strategy.get("origin_format")
        # if "value" in strategy.keys():
        #     self.value = strategy.get("value")

    def get_strategy(self) -> dict:
        if self.strategy:
            return self.strategy
        else:
            fmt = {"format": self.format,
                    "dtype": self.dtype,
                    "shape": self.shape,
                    "origin_format": self.origin_format,
                    "value": self.value}
        return fmt

    def __dict__(self):
        return self.get_strategy()

    def get_desc(self):
        """
        对于Tensor而言，desc中将存储其自身的描述信息，用于CANN中对buffer进行解读
        :return:
        """
        if not self.desc:
            acl_dtype = Converter.dtype(self.dtype)
            acl_format = Converter.format(self.format)
            self.desc = acl.create_tensor_desc(acl_dtype, list(self.shape), acl_format)
            check(0, "acl.create_tensor_desc (addr:{})".format(self.desc))
        return self.desc

    def get_buffer(self):
        """
        该函数将会创建device侧的buffer,此时会将中间内容也传递出去
        依次为：aclDataBuffer*, device_ptr, buff_size
        :param desc:
        :return: {"buffer"}
        """
        if not self.buffer:
            if not self.desc:
                self.get_desc()
            size = acl.get_tensor_desc_size(self.desc)
            if self.is_host:
                self.ptr, ret = acl.rt.malloc_host(size)
                check(ret, "acl.rt.malloc_host (addr:{}, size:{})".format(self.ptr, size))
                # 对于host侧的tensor无需创建buffer
                self.buffer = acl.create_data_buffer(self.ptr, size)
                check(0, "acl.create_data_buffer (addr:{})".format(self.buffer))
            else:
                self.ptr, ret = acl.rt.malloc(size, Converter.malloc_mode("huge_first"))
                check(ret, "acl.rt.malloc (addr:{}, size:{})".format(self.ptr, size))
                self.buffer = acl.create_data_buffer(self.ptr, size)
                check(0, "acl.create_data_buffer (addr:{})".format(self.buffer))
        return self.buffer

    def __str__(self):
        if self.is_host and self.value:
            fmt = "dtype:{}, format:{}, shape:{}, value:{}"
            return fmt.format(self.dtype, self.format, self.shape, self.value)
        else:
            fmt = "dtype:{}, format:{}, shape:{}"
            return fmt.format(self.dtype, self.format, self.shape)

    def get_value(self):
        """
        将Tensor的描述中的信息生成具体的
        :return:
        """

        # TODO: 需要后续支持numpy的内容
        def dtype_to_numpy(value):
            mapping_table = {
                "float": np.float32,
                "float16": np.float16,
                "int8": np.int8,
                "int32": np.int32,
                "uint8": np.uint8,
                "int16": np.int16,
                "uint16": np.uint16,
                "uint32": np.uint32,
                "int64": np.int64,
                "double": np.float64,
                "bool": np.bool_
            }
            return mapping_table.get(value)

        # TODO: 需要扩展或者针对性地验证
        if self.mode == "random":
            if self.dtype in set(["float", "float16", "double"]):
                return self.max_num * np.random.rand(*self.shape).astype(dtype_to_numpy(self.dtype))
            return np.random.randint(self.max_num, size=self.shape).astype(dtype_to_numpy(self.dtype))
        else:
            return np.array(self.value).astype(dtype_to_numpy(self.dtype))

    def get_size(self):
        """
        获取Tensor的元素大小
        """
        return np.prod(self.shape)
