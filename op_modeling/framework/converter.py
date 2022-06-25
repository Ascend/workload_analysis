#!/usr/bin/python

class Converter:
    """
    该类将各种属性的值转换为ACL接口需要的值
    """

    @staticmethod
    def dtype(value):
        """
        将此处定义的类型转换为
        :param dtype:
        :return:
        """
        dtype_mapping_table = {
            "float": 0,
            "float16": 1,
            "int8": 2,
            "int32": 3,
            "uint8": 4,
            "int16": 6,
            "uint16": 7,
            "uint32": 8,
            "int64": 9,
            "double": 11,
            "bool": 12
        }
        return dtype_mapping_table.get(value, -1)

    @staticmethod
    def format(value):
        """
        将普通格式转换为ACL的format格式
        :param value:
        :return:
        """
        format_mapping_table = {
            "NCHW": 0,
            "NHWC": 1,
            "ND": 2,
            "NC1HWC0": 3,
            "FRACTAL_Z": 4,
            "HWCN": 16,
            "FRACTAL_NZ": 29,
            "NDHWC": 27,
            "NCDHW": 30,
            "DHWCN": 31,
            "NDC1HWC0": 32,
            "FRACTAL_Z_3D": 33
        }
        return format_mapping_table.get(value, -1)

    @staticmethod
    def malloc_mode(value):
        """
        定义了device侧内存申请的模式
        :param value:
        :return:
        """
        malloc_mode_mapping_table = {
            "huge_first": 0,
            "huge_only": 1,
            "normal_only": 2
        }
        return malloc_mode_mapping_table.get(value, -1)

    @staticmethod
    def copy_mode(value):
        """
        定义了拷贝函数的模式（含方向）
        :param value:
        :return:
        """
        copy_mode_mapping_table = {
            "h2h": 0,  # host -> host
            "h2d": 1,  # host -> device
            "d2h": 2,  # device -> device
            "d2d": 3  # device -> device
        }
        return copy_mode_mapping_table.get(value, -1)

    @staticmethod
    def engine_type(value):
        """
        :param value:
        :return:
        """
        engine_type_mapping_table = {
            "sys": 0,
            "aicore": 1,
            "vector": 2
        }
        return engine_type_mapping_table.get(value, -1)

    @staticmethod
    def compile_type(value):
        compile_type_mapping_table = {
            "sys": 0,
        }
        return compile_type_mapping_table.get(value, -1)
