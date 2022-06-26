#!/usr/bin/python
import json
from abc import abstractmethod

import acl
import numpy as np
from framework.util import check
from framework.converter import Converter


class Tensor:
    """
    该类主要提供由Tensor的基本信息生成一个完整Tensor内容的方法。
    该类的输入是具体的策略实例化信息
    """

    def __init__(self, strategy: dict = None):
        self.dtype = 'float'
        self.format = 'ND'
        self.origin_format = 'ND'
        self.shape = (8, 16)
        self.max_num = 1
        self.mode = "random"
        self.is_host = False

        self.buffer = None
        self.ptr = None
        self.desc = None
        self.value = None
        self.strategy = strategy

        if strategy:
            self.set_strategy(strategy)

    def __del__(self):
        self._release_source()

    def __str__(self):
        fmt = "dtype:{}, format:{}, shape:{}"
        return fmt.format(self.dtype, self.format, self.shape)

    @abstractmethod
    def malloc(self, size):
        """
        调用acl接口申请tensor内存
        """
        raise Exception("To be implement")

    @abstractmethod
    def free(self, ptr):
        """
        调用acl接口释放tensor内存
        """
        raise Exception("To be implement")

    def force_del(self):
        self._release_source()

    def set_strategy(self, strategy: dict):
        self.dtype = strategy.get("dtype")
        self.format = strategy.get("format")
        self.shape = strategy.get("shape")
        self.is_host = strategy.get("is_host")
        if "value" in strategy.keys():
            self.value = strategy.get("value")
            self.mode = "const"

    def get_strategy(self) -> dict:
        return self.strategy

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

            self.ptr = self.malloc(size)
            self.buffer = acl.create_data_buffer(self.ptr, size)
            check(0, "acl.create_data_buffer (addr:{})".format(self.buffer))
        return self.buffer

    def get_value(self):
        """
        将Tensor的描述中的信息生成具体的
        :return:
        """

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

        if self.mode == "random":
            if self.dtype in {"float", "float16", "double"}:
                return self.max_num * np.random.rand(*self.shape).astype(dtype_to_numpy(self.dtype))
            return np.random.randint(self.max_num, size=self.shape).astype(dtype_to_numpy(self.dtype))
        # const
        elif self.mode == "const":
            return np.array(self.value).astype(dtype_to_numpy(self.dtype))
        else:
            raise Exception(f"Unsupported tensor mode: {self.mode}")

    def get_size(self):
        """
        获取Tensor的元素大小
        """
        return np.prod(self.shape)

    def _release_source(self):
        if self.buffer:
            ret = acl.destroy_data_buffer(self.buffer)
            check(ret, "acl.destroy_data_buffer (addr:{})".format(self.buffer))
            self.buffer = None
            self.free(self.ptr)
            self.ptr = None

        if self.desc:
            acl.destroy_tensor_desc(self.desc)
            check(0, "acl.destroy_tensor_desc (addr:{})".format(self.desc))
            self.desc = None


class DeviceTensor(Tensor):
    def malloc(self, size):
        ptr, ret = acl.rt.malloc(size, Converter.malloc_mode("huge_first"))
        check(ret, "acl.rt.malloc (addr:{}, size:{})".format(ptr, size))
        return ptr

    def free(self, ptr):
        ret = acl.rt.free(ptr)
        check(ret, "acl.rt.free (addr:{})".format(ptr))


class HostTensor(Tensor):
    def malloc(self, size):
        ptr, ret = acl.rt.malloc_host(size)
        check(ret, "acl.rt.malloc_host (addr:{}, size:{})".format(ptr, size))
        return ptr

    def free(self, ptr):
        ret = acl.rt.free_host(ptr)
        check(ret, "acl.rt.free_host (addr:{})".format(ptr))
