#!/usr/bin/python
import json

from profile import Profile
from typing import List

import acl
import acl_ext

import numpy as np

from workload.framework.util import check
from workload.framework.tensor import Tensor
from workload.framework.converter import Converter


class OpBase:
    """
    该类中本身需要为算子的输入做准备，此处本身需要与算子的特征相关内容：
    1. 算子的输入输出以及属性定义
    2.
    """

    def __init__(self, device, op_type, op_mode=None):
        if device:
            self.stream = device.stream()
            device.register(self)

        self.op_info = {'type': op_type, 'attr': acl.op.create_attr(), "attributes": None}
        self.is_custom = False
        self.inputs = list()
        self.outputs = list()

    def _release_source_on_device(self):
        if self.op_info['attr']:
            acl.op.destroy_attr(self.op_info['attr'])
            self.op_info['attr'] = None

    def force_del(self):
        tensors = self.inputs + self.outputs
        for tensor in tensors:
            tensor.force_del()
        self._release_source_on_device()

    def __del__(self):
        self._release_source_on_device()

    def __str__(self):
        fmt = "input: "
        for i in self.inputs:
            fmt += str(i.shape)
            fmt += ", "
        fmt += "output: "
        for i in self.outputs:
            fmt += str(i.shape)
            fmt += ", "
        fmt += "attr: "
        fmt += str(self.op_info["attributes"])
        return fmt

    def attr(self, name, value):
        if self.op_info["attributes"] is None:
            self.op_info['attributes'] = {name: value}
        else:
            self.op_info['attributes'][name] = value

        if isinstance(value, int):
            if type(value) is bool:
                acl.op.set_attr_bool(self.op_info['attr'], name, value)
            else:
                acl.op.set_attr_int(self.op_info['attr'], name, value)
            return self

        if isinstance(value, float):
            acl.op.set_attr_float(self.op_info['attr'], name, value)
            return self

        if isinstance(value, str):
            acl.op.set_attr_string(self.op_info['attr'], name, value)
            return self

        if isinstance(value, list) or isinstance(value, tuple):
            if isinstance(value[0], int):
                if type(value[0]) is bool:
                    acl.op.set_attr_list_bool(self.op_info['attr'], name, np.array(value))
                else:
                    acl.op.set_attr_list_int(self.op_info['attr'], name, np.array(value))
                return self

            if isinstance(value[0], float):
                acl.op.set_attr_list_float(self.op_info['attr'], name, np.array(value))
                return self

            if isinstance(value[0], str):
                acl.op.set_attr_list_string(self.op_info['attr'], name, np.array(value))
                return self

            if isinstance(value[0], list):
                if isinstance(value[0][0], int):
                    value = [np.array(x) for x in value]
                    acl.op.set_attr_list_list_int(self.op_info['attr'], name, np.array(value))
                    return self

        raise Exception("unsupport attr (name:{}, val:{})".format(name, value))

    def input(self, name, tensor: Tensor):
        self.inputs.append(tensor)
        return self

    def output(self, name, tensor: Tensor):
        self.outputs.append(tensor)
        return self

    def _execute(self, inputs, outputs):
        op_type = self.op_info['type']
        op_attr = self.op_info['attr']
        in_descs = inputs.get('descs')
        in_buffers = inputs.get('buffers')
        out_descs = outputs.get('descs')
        out_buffers = outputs.get('buffers')
        ret = acl_ext.op.compile_and_execute(op_type, len(in_descs), in_descs, in_buffers, \
                                         len(out_descs), out_descs, out_buffers, \
                                         op_attr, Converter.engine_type("sys"), \
                                         Converter.compile_type("sys"), "no use", self.stream)
        check(ret, "acl_ext.op.compile_and_execute")
        ret = acl.rt.synchronize_stream(self.stream)
        check(ret, "acl.rt.synchronize_stream")

    def _pretreatment_and_exec(self):
        def _pretreatment(in_out):
            """将op的输入输出转换为acl接口的格式"""
            result = {"descs": list(), "buffers": list()}
            for tensor in in_out:
                desc, buffer = tensor.get_desc(), tensor.get_buffer()
                result['descs'].append(desc)
                result['buffers'].append(buffer)
            return result

        inputs = _pretreatment(self.inputs)
        outputs = _pretreatment(self.outputs)
        self._execute(inputs, outputs)

    def prepare(self):
        """
        1.依据策略生成合理的输入内容
        2.数据搬移到device侧
        :return:
        """
        for tensor in self.inputs:
            np_data = tensor.get_value()
            host_ptr = acl.util.numpy_to_ptr(np_data)

            device_buffer = tensor.get_buffer()
            if not tensor.is_host:
                device_ptr = acl.get_data_buffer_addr(device_buffer)
                size = acl.get_data_buffer_size_v2(device_buffer)
                ret = acl.rt.memcpy(device_ptr, size, host_ptr, size, Converter.copy_mode("h2d"))
                check(ret, "acl.rt.memcpy host->device (addr:{}->{})".format(host_ptr, device_ptr))
            else:
                desc = tensor.get_desc()
                size = acl.get_tensor_desc_size(desc)
                ret = acl.rt.memcpy(tensor.ptr, size, host_ptr, size, Converter.copy_mode("h2h"))
                check(ret, "acl.rt.memcpy host->host (addr:{}->{})".format(host_ptr, tensor.ptr))
                ret = acl.set_tensor_const(desc, tensor.ptr, size)
                check(ret, "acl.set_tensor_const")

    def get_op_type(self):
        return self.op_info['type']

    def get_attributes(self):
        if "attributes" in self.op_info:
            return self.op_info["attributes"]
        else:
            return None

    def run(self, prof: Profile = None):
        # 准备工作：将输入搬移到设备侧
        self.prepare()

        # 执行
        self._pretreatment_and_exec()

        if prof:
            prof.inc()

    @staticmethod
    def _get_json(inputs):
        outputs = []
        for item in inputs:
            if type(item) is Tensor:
                outputs.append(item.__dict__())
            elif type(item) is list:
                for v in item:
                    outputs.append(v.__dict__())
            else:
                outputs.append(item)
        return json.dumps(outputs)

    def get_unique_desc(self):
        attr = self.get_attributes()
        if attr:
            attr = json.dumps(attr)
        else:
            attr = str(attr)

        return self._get_json(self.inputs), self._get_json(self.outputs), attr
