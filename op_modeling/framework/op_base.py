#!/usr/bin/python
import json
import numpy as np

import acl
import acl_ext

from framework.util import check
from framework.tensor import Tensor
from framework.converter import Converter
from framework.profile import Profile


class OpBase:
    """
    该类中本身需要为算子的输入做准备，此处本身需要与算子的特征相关内容：
    1. 算子的输入输出以及属性定义
    2. 算子的内存准备和执行
    """

    def __init__(self, device, op_type):
        if device:
            self.stream = device.stream()
            device.register(self)

        self.op_info = {'type': op_type, 'attr': acl.op.create_attr(), "attributes": {}}
        self.is_custom = False
        self.inputs = list()
        self.outputs = list()

    def __del__(self):
        self._release_source_on_device()

    def __str__(self):
        def shape_desc(tensors):
            shapes = [str(tensor.shape) for tensor in tensors]
            return ", ".join(shapes)

        fmt = "input: {}, output: {}, attr: {}"
        str_ = fmt.format(shape_desc(self.inputs), shape_desc(self.outputs), str(self.op_info.get("attributes")))
        return str_

    @classmethod
    def get_json(cls, inputs):
        outputs = []
        for item in inputs:
            outputs.append(item.get_strategy())
        return json.dumps(outputs)

    @classmethod
    def memcpy_data_to_host_tensor(cls, data_ptr, tensor):
        tensor.get_buffer()
        desc = tensor.get_desc()
        size = acl.get_tensor_desc_size(desc)
        ret = acl.rt.memcpy(tensor.ptr, size, data_ptr, size, Converter.copy_mode("h2h"))
        check(ret, "acl.rt.memcpy host->host (addr:{}->{})".format(data_ptr, tensor.ptr))
        ret = acl.set_tensor_const(desc, tensor.ptr, size)
        check(ret, "acl.set_tensor_const")

    @classmethod
    def memcpy_data_to_device_tensor(cls, data_ptr, tensor):
        device_buffer = tensor.get_buffer()
        device_ptr = acl.get_data_buffer_addr(device_buffer)
        size = acl.get_data_buffer_size_v2(device_buffer)
        ret = acl.rt.memcpy(device_ptr, size, data_ptr, size, Converter.copy_mode("h2d"))
        check(ret, "acl.rt.memcpy host->device (addr:{}->{})".format(data_ptr, device_ptr))

    def force_del(self):
        tensors = self.inputs + self.outputs
        for tensor in tensors:
            tensor.force_del()
        self._release_source_on_device()

    def attr(self, name, value):
        attribute_desc = self.op_info.get("attributes")
        if attribute_desc is None:
            self.op_info["attributes"] = {name: value}
        else:
            attribute_desc[name] = value

        op_attr = self.op_info.get("attr")

        if isinstance(value, int):
            if type(value) is bool:
                acl.op.set_attr_bool(op_attr, name, value)
            else:
                acl.op.set_attr_int(op_attr, name, value)
            return self

        if isinstance(value, float):
            acl.op.set_attr_float(op_attr, name, value)
            return self

        if isinstance(value, str):
            acl.op.set_attr_string(op_attr, name, value)
            return self

        if isinstance(value, list) or isinstance(value, tuple):
            if isinstance(value[0], int):
                if type(value[0]) is bool:
                    acl.op.set_attr_list_bool(op_attr, name, np.array(value))
                else:
                    acl.op.set_attr_list_int(op_attr, name, np.array(value))
                return self

            if isinstance(value[0], float):
                acl.op.set_attr_list_float(op_attr, name, np.array(value))
                return self

            if isinstance(value[0], str):
                acl.op.set_attr_list_string(op_attr, name, np.array(value))
                return self

            if isinstance(value[0], list):
                if isinstance(value[0][0], int):
                    value = [np.array(x) for x in value]
                    acl.op.set_attr_list_list_int(op_attr, name, np.array(value))
                    return self

        raise Exception("unsupported attr (name:{}, val:{})".format(name, value))

    def input(self, name, tensor: Tensor):
        self.inputs.append(tensor)
        return self

    def output(self, name, tensor: Tensor):
        self.outputs.append(tensor)
        return self

    def prepare(self):
        """
        1.依据策略生成合理的输入内容
        2.数据搬移到device侧
        :return:
        """
        for tensor in self.inputs:
            np_data = tensor.get_value()
            data_ptr = acl.util.numpy_to_ptr(np_data)

            if not tensor.is_host:
                self.memcpy_data_to_device_tensor(data_ptr, tensor)
            else:
                self.memcpy_data_to_host_tensor(data_ptr, tensor)

    def get_op_type(self):
        return self.op_info.get('type')

    def get_attributes(self):
        return self.op_info.get("attributes")

    def run(self, prof: Profile = None):
        # 准备工作：将输入搬移到设备侧
        self.prepare()

        # 执行
        self._pretreatment_and_exec()

        if prof:
            prof.inc()

    def get_unique_desc(self):
        """
        将op的input,output,desc的json字符串作为op的唯一描述
        """
        attr = self.get_attributes()
        return {
            'Original Inputs': self.get_json(self.inputs),
            'Original Outputs': self.get_json(self.outputs),
            'Attributes': json.dumps(attr)
        }

    def _release_source_on_device(self):
        op_attr = self.op_info.get('attr')
        if op_attr is not None:
            acl.op.destroy_attr(op_attr)
            self.op_info['attr'] = None

    def _execute(self, inputs, outputs):
        op_type = self.op_info.get('type')
        op_attr = self.op_info.get('attr')
        in_descs = inputs.get('descs')
        in_buffers = inputs.get('buffers')
        out_descs = outputs.get('descs')
        out_buffers = outputs.get('buffers')
        ret = acl_ext.op.compile_and_execute(op_type, len(in_descs), in_descs, in_buffers,
                                             len(out_descs), out_descs, out_buffers,
                                             op_attr, Converter.engine_type("sys"),
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
                result.get('descs').append(desc)
                result.get('buffers').append(buffer)
            return result

        inputs = _pretreatment(self.inputs)
        outputs = _pretreatment(self.outputs)
        self._execute(inputs, outputs)
